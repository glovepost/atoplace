"""
AtoPlace Lock File Handler

Provides persistence for component placements via an `atoplace.lock` sidecar file.
This allows placement decisions to survive atopile rebuilds (which regenerate
the KiCad board) and enables user-locked positions to take precedence over
physics-based placement.

File Format (YAML):
```yaml
version: 1
created: 2026-01-14T10:30:00
modified: 2026-01-14T11:45:00
build: default
components:
  U1:
    x: 125.5
    y: 80.0
    rotation: 0.0
    locked: true
    module: power
  R1:
    x: 130.0
    y: 85.5
    rotation: 90.0
    locked: false
    module: power
```

The `locked` field indicates whether the position was explicitly approved by
the user (true) or is just from the last placement run (false). Locked
positions always take precedence; unlocked positions are used as initial
hints but can be re-optimized.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging

# Use yaml if available, fall back to basic serialization
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .abstraction import Board, Component

logger = logging.getLogger(__name__)

# Lock file version for format compatibility
LOCK_FILE_VERSION = 1


@dataclass
class ComponentPosition:
    """Stored position for a single component."""
    reference: str
    x: float
    y: float
    rotation: float = 0.0
    locked: bool = False  # True = user-approved, False = auto-placed
    module: Optional[str] = None  # atopile module name if known
    layer: Optional[str] = None  # "F.Cu" or "B.Cu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "rotation": round(self.rotation, 2),
            "locked": self.locked,
        }
        if self.module:
            d["module"] = self.module
        if self.layer:
            d["layer"] = self.layer
        return d

    @classmethod
    def from_dict(cls, reference: str, data: Dict[str, Any]) -> "ComponentPosition":
        """Create from dictionary."""
        return cls(
            reference=reference,
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            rotation=float(data.get("rotation", 0.0)),
            locked=bool(data.get("locked", False)),
            module=data.get("module"),
            layer=data.get("layer"),
        )


@dataclass
class AtoplaceLock:
    """
    Represents an atoplace.lock sidecar file.

    This file stores component positions that should be preserved across
    atopile builds and placement runs.
    """
    version: int = LOCK_FILE_VERSION
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    build: str = "default"  # atopile build name

    # Component positions keyed by reference
    components: Dict[str, ComponentPosition] = field(default_factory=dict)

    # Source board file this lock is associated with
    source_file: Optional[Path] = None

    def __post_init__(self):
        """Initialize timestamps if not set."""
        now = datetime.now()
        if self.created is None:
            self.created = now
        if self.modified is None:
            self.modified = now

    def add_component(
        self,
        reference: str,
        x: float,
        y: float,
        rotation: float = 0.0,
        locked: bool = False,
        module: Optional[str] = None,
        layer: Optional[str] = None,
    ):
        """Add or update a component position."""
        self.components[reference] = ComponentPosition(
            reference=reference,
            x=x,
            y=y,
            rotation=rotation,
            locked=locked,
            module=module,
            layer=layer,
        )
        self.modified = datetime.now()

    def get_position(self, reference: str) -> Optional[ComponentPosition]:
        """Get stored position for a component."""
        return self.components.get(reference)

    def get_locked_refs(self) -> Set[str]:
        """Get references of all locked components."""
        return {ref for ref, pos in self.components.items() if pos.locked}

    def lock_component(self, reference: str):
        """Mark a component as locked (user-approved position)."""
        if reference in self.components:
            self.components[reference].locked = True
            self.modified = datetime.now()

    def unlock_component(self, reference: str):
        """Mark a component as unlocked (can be re-optimized)."""
        if reference in self.components:
            self.components[reference].locked = False
            self.modified = datetime.now()

    def lock_all(self):
        """Lock all component positions."""
        for pos in self.components.values():
            pos.locked = True
        self.modified = datetime.now()

    def unlock_all(self):
        """Unlock all component positions."""
        for pos in self.components.values():
            pos.locked = False
        self.modified = datetime.now()

    def remove_component(self, reference: str) -> bool:
        """Remove a component from the lock file."""
        if reference in self.components:
            del self.components[reference]
            self.modified = datetime.now()
            return True
        return False

    def update_from_board(
        self,
        board: Board,
        refs: Optional[List[str]] = None,
        lock: bool = False,
        preserve_locked: bool = True,
    ):
        """
        Update positions from a board.

        Args:
            board: Board to extract positions from
            refs: Optional list of refs to update (all if None)
            lock: Whether to mark updated positions as locked
            preserve_locked: If True, don't overwrite locked positions
        """
        refs_to_update = refs if refs else list(board.components.keys())

        for ref in refs_to_update:
            comp = board.get_component(ref)
            if not comp:
                continue

            # Skip locked positions if preserve_locked is True
            if preserve_locked and ref in self.components:
                if self.components[ref].locked:
                    continue

            # Get module info if available
            module = comp.properties.get("ato_module")

            self.add_component(
                reference=ref,
                x=comp.x,
                y=comp.y,
                rotation=comp.rotation,
                locked=lock,
                module=module,
                layer=comp.layer.value if comp.layer else None,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "created": self.created.isoformat() if self.created else None,
            "modified": self.modified.isoformat() if self.modified else None,
            "build": self.build,
            "components": {
                ref: pos.to_dict() for ref, pos in self.components.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtoplaceLock":
        """Create from dictionary."""
        components = {}
        comp_data = data.get("components", {})
        for ref, pos_data in comp_data.items():
            if isinstance(pos_data, dict):
                components[ref] = ComponentPosition.from_dict(ref, pos_data)

        # Parse timestamps
        created = None
        modified = None
        created_value = data.get("created")
        if isinstance(created_value, datetime):
            created = created_value
        elif isinstance(created_value, str):
            try:
                created = datetime.fromisoformat(created_value)
            except ValueError:
                pass

        modified_value = data.get("modified")
        if isinstance(modified_value, datetime):
            modified = modified_value
        elif isinstance(modified_value, str):
            try:
                modified = datetime.fromisoformat(modified_value)
            except ValueError:
                pass

        return cls(
            version=data.get("version", LOCK_FILE_VERSION),
            created=created,
            modified=modified,
            build=data.get("build", "default"),
            components=components,
        )

    def __repr__(self) -> str:
        locked_count = len(self.get_locked_refs())
        return (
            f"AtoplaceLock(build={self.build!r}, "
            f"components={len(self.components)}, "
            f"locked={locked_count})"
        )


def get_lock_file_path(board_path: Path) -> Path:
    """
    Get the path to the lock file for a board.

    The lock file is placed next to the board file with `.atoplace.lock` extension.
    For example: `board.kicad_pcb` -> `board.atoplace.lock`
    """
    board_path = Path(board_path)
    return board_path.parent / f"{board_path.stem}.atoplace.lock"


def parse_lock_file(path: Path) -> Optional[AtoplaceLock]:
    """
    Parse an atoplace.lock file.

    Args:
        path: Path to the lock file

    Returns:
        AtoplaceLock instance or None if file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        logger.debug(f"Lock file not found: {path}")
        return None

    try:
        content = path.read_text()

        if YAML_AVAILABLE:
            data = yaml.safe_load(content)
        else:
            data = _parse_simple_yaml(content)

        if not data:
            return None

        lock = AtoplaceLock.from_dict(data)
        lock.source_file = path
        logger.debug(f"Loaded lock file: {lock}")
        return lock

    except Exception as e:
        logger.warning(f"Failed to parse lock file {path}: {e}")
        return None


def write_lock_file(lock: AtoplaceLock, path: Path) -> bool:
    """
    Write an atoplace.lock file.

    Args:
        lock: Lock data to write
        path: Path to write to

    Returns:
        True if successful
    """
    path = Path(path)

    try:
        # Update modified timestamp
        lock.modified = datetime.now()

        data = lock.to_dict()

        if YAML_AVAILABLE:
            content = yaml.dump(
                data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        else:
            content = _serialize_simple_yaml(data)

        path.write_text(content)
        lock.source_file = path
        logger.info(f"Saved lock file: {path} ({len(lock.components)} components)")
        return True

    except Exception as e:
        logger.error(f"Failed to write lock file {path}: {e}")
        return False


def _parse_simple_yaml(content: str) -> Dict[str, Any]:
    """
    Simple YAML parser fallback when PyYAML is not available.
    Handles the nested structure of atoplace.lock files.
    """
    result = {}
    stack = [(0, result)]

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        # Pop stack until we find parent at lower indent
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        current = stack[-1][1]

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if value:
                # Parse value type
                if value.lower() == "true":
                    current[key] = True
                elif value.lower() == "false":
                    current[key] = False
                elif value.lower() == "null" or value.lower() == "~":
                    current[key] = None
                else:
                    # Try numeric conversion
                    try:
                        if "." in value:
                            current[key] = float(value)
                        else:
                            current[key] = int(value)
                    except ValueError:
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        current[key] = value
            else:
                # Section header
                new_section = {}
                current[key] = new_section
                stack.append((indent, new_section))

    return result


def _serialize_simple_yaml(data: Dict[str, Any], indent: int = 0) -> str:
    """
    Simple YAML serializer fallback when PyYAML is not available.
    """
    lines = []
    prefix = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_serialize_simple_yaml(value, indent + 1))
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {str(value).lower()}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, (int, float)):
            lines.append(f"{prefix}{key}: {value}")
        else:
            # String value - quote if contains special chars
            str_value = str(value)
            if any(c in str_value for c in ":{}[]#&*!|>'\"%@`"):
                str_value = f'"{str_value}"'
            lines.append(f"{prefix}{key}: {str_value}")

    return "\n".join(lines)


def apply_lock_to_board(
    board: Board,
    lock: AtoplaceLock,
    only_locked: bool = False,
    skip_missing: bool = True,
) -> int:
    """
    Apply positions from a lock file to a board.

    This is the core merge logic where Lock > Physics: locked positions
    take precedence over the board's current positions.

    Args:
        board: Board to modify in place
        lock: Lock file with saved positions
        only_locked: If True, only apply positions marked as locked
        skip_missing: If True, skip components not in the board

    Returns:
        Number of components updated
    """
    updated = 0

    for ref, pos in lock.components.items():
        # Skip unlocked if only_locked is True
        if only_locked and not pos.locked:
            continue

        comp = board.get_component(ref)
        if not comp:
            if not skip_missing:
                logger.warning(f"Component {ref} in lock file not found on board")
            continue

        # Apply position
        comp.x = pos.x
        comp.y = pos.y
        comp.rotation = pos.rotation

        # Mark as locked in component properties for downstream use
        if pos.locked:
            comp.locked = True

        updated += 1

    logger.debug(f"Applied {updated} positions from lock file to board")
    return updated


def create_lock_from_board(
    board: Board,
    build: str = "default",
    lock_all: bool = False,
) -> AtoplaceLock:
    """
    Create a new lock file from a board's current state.

    Args:
        board: Board to extract positions from
        build: atopile build name
        lock_all: If True, mark all positions as locked

    Returns:
        New AtoplaceLock instance
    """
    lock = AtoplaceLock(build=build)

    for ref, comp in board.components.items():
        # Skip DNP components
        if comp.dnp:
            continue

        module = comp.properties.get("ato_module")

        lock.add_component(
            reference=ref,
            x=comp.x,
            y=comp.y,
            rotation=comp.rotation,
            locked=lock_all or comp.locked,
            module=module,
            layer=comp.layer.value if comp.layer else None,
        )

    return lock


def merge_lock_files(
    base: AtoplaceLock,
    overlay: AtoplaceLock,
    prefer_locked: bool = True,
) -> AtoplaceLock:
    """
    Merge two lock files, with overlay taking precedence.

    Useful for combining user-specified locks with auto-generated positions.

    Args:
        base: Base lock file (lower priority)
        overlay: Overlay lock file (higher priority)
        prefer_locked: If True, locked positions always win regardless of source

    Returns:
        Merged AtoplaceLock
    """
    result = AtoplaceLock(
        build=overlay.build or base.build,
        created=base.created,
    )

    # Start with base components
    for ref, pos in base.components.items():
        result.components[ref] = ComponentPosition(
            reference=ref,
            x=pos.x,
            y=pos.y,
            rotation=pos.rotation,
            locked=pos.locked,
            module=pos.module,
            layer=pos.layer,
        )

    # Apply overlay
    for ref, pos in overlay.components.items():
        if ref in result.components:
            existing = result.components[ref]
            # If prefer_locked and existing is locked, skip overlay unless also locked
            if prefer_locked and existing.locked and not pos.locked:
                continue

        result.components[ref] = ComponentPosition(
            reference=ref,
            x=pos.x,
            y=pos.y,
            rotation=pos.rotation,
            locked=pos.locked,
            module=pos.module,
            layer=pos.layer,
        )

    return result
