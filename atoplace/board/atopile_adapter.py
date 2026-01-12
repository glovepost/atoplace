"""
Atopile Project Adapter

Provides integration with atopile projects, enabling AtoPlace to:
1. Detect and load atopile projects from ato.yaml
2. Resolve board file paths from build entry points
3. Parse ato-lock.yaml for component metadata
4. Extract module hierarchy from .ato files for grouping

This adapter works with atopile's output files rather than embedding
into the compiler, providing version-resilient integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re
import logging

# Use yaml if available, fall back to basic parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .abstraction import Board, Component

logger = logging.getLogger(__name__)


@dataclass
class AtopileProject:
    """Represents an atopile project configuration."""
    root: Path
    ato_version: Optional[str] = None
    builds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ComponentMetadata:
    """Component metadata extracted from ato-lock.yaml."""
    reference: str
    mpn: Optional[str] = None
    value: Optional[str] = None
    package: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ModuleHierarchy:
    """Module hierarchy extracted from .ato files."""
    name: str
    module_type: Optional[str] = None  # e.g., "power_supply", "sensor"
    components: List[str] = field(default_factory=list)
    submodules: List['ModuleHierarchy'] = field(default_factory=list)
    parent: Optional[str] = None


class AtopileProjectLoader:
    """
    Load and manage atopile projects for placement optimization.

    Usage:
        loader = AtopileProjectLoader(Path("my-project"))
        board = loader.load_board()  # Loads default build
        board = loader.load_board("custom-build")  # Loads specific build
    """

    def __init__(self, project_root: Path):
        """
        Initialize loader for an atopile project.

        Args:
            project_root: Path to project directory containing ato.yaml
        """
        self.root = Path(project_root).resolve()
        self._project: Optional[AtopileProject] = None
        self._lock_data: Optional[Dict] = None
        self._module_hierarchy: Optional[Dict[str, ModuleHierarchy]] = None

        # Validate project
        if not self.is_atopile_project(self.root):
            raise ValueError(
                f"Not an atopile project: {self.root}\n"
                f"Missing ato.yaml file"
            )

    @staticmethod
    def is_atopile_project(path: Path) -> bool:
        """Check if a path is an atopile project root."""
        path = Path(path)
        if path.is_file():
            path = path.parent
        return (path / "ato.yaml").exists()

    @staticmethod
    def find_project_root(start_path: Path) -> Optional[Path]:
        """
        Find atopile project root by searching up the directory tree.

        Args:
            start_path: Starting path to search from

        Returns:
            Project root path or None if not found
        """
        path = Path(start_path).resolve()
        if path.is_file():
            path = path.parent

        while path != path.parent:
            if (path / "ato.yaml").exists():
                return path
            path = path.parent

        return None

    @property
    def project(self) -> AtopileProject:
        """Get the parsed project configuration."""
        if self._project is None:
            self._project = self._load_ato_yaml()
        return self._project

    @property
    def lock_data(self) -> Dict:
        """Get the parsed lock file data."""
        if self._lock_data is None:
            self._lock_data = self._load_lock_file()
        return self._lock_data

    def _load_ato_yaml(self) -> AtopileProject:
        """Parse ato.yaml configuration file."""
        ato_yaml_path = self.root / "ato.yaml"

        if not ato_yaml_path.exists():
            raise FileNotFoundError(f"ato.yaml not found: {ato_yaml_path}")

        content = ato_yaml_path.read_text()

        if YAML_AVAILABLE:
            data = yaml.safe_load(content)
        else:
            # Basic YAML parsing fallback
            data = self._parse_simple_yaml(content)

        return AtopileProject(
            root=self.root,
            ato_version=data.get("ato-version"),
            builds=data.get("builds", {}),
            dependencies=data.get("dependencies", []),
        )

    def _load_lock_file(self) -> Dict:
        """Parse ato-lock.yaml if present."""
        lock_path = self.root / "ato-lock.yaml"

        if not lock_path.exists():
            logger.debug(f"No lock file found: {lock_path}")
            return {}

        content = lock_path.read_text()

        if YAML_AVAILABLE:
            return yaml.safe_load(content) or {}
        else:
            return self._parse_simple_yaml(content)

    def _parse_simple_yaml(self, content: str) -> Dict:
        """
        Simple YAML parser for basic key-value structures.
        Falls back to this when PyYAML is not available.
        """
        result = {}
        current_section = None
        current_indent = 0

        for line in content.split('\n'):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Calculate indent level
            indent = len(line) - len(line.lstrip())

            # Handle key-value pairs
            if ':' in stripped:
                key, _, value = stripped.partition(':')
                key = key.strip()
                value = value.strip()

                if value:
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    result[key] = value
                else:
                    # This is a section header
                    result[key] = {}
                    current_section = key

        return result

    def get_build_names(self) -> List[str]:
        """Get list of available build configurations."""
        return list(self.project.builds.keys())

    def get_entry_point(self, build_name: str = "default") -> Optional[str]:
        """
        Get the entry point for a build configuration.

        Args:
            build_name: Name of the build (default: "default")

        Returns:
            Entry point string like "elec/src/project.ato:MainModule"
        """
        if build_name not in self.project.builds:
            available = ", ".join(self.project.builds.keys()) or "(none)"
            raise ValueError(
                f"Build '{build_name}' not found. Available: {available}"
            )

        build = self.project.builds[build_name]
        return build.get("entry")

    def get_board_path(self, build_name: str = "default") -> Path:
        """
        Get the path to the generated KiCad board file.

        Atopile generates boards in elec/layout/<build_name>/<project>.kicad_pcb

        Args:
            build_name: Name of the build configuration

        Returns:
            Path to the .kicad_pcb file
        """
        entry = self.get_entry_point(build_name)
        if not entry:
            raise ValueError(f"No entry point for build: {build_name}")

        # Parse entry: "elec/src/project.ato:MainModule"
        # -> "elec/layout/<build>/project.kicad_pcb"
        file_part, _, module_name = entry.partition(':')

        # Extract base name from the .ato file path
        ato_path = Path(file_part)
        base_name = ato_path.stem  # e.g., "project" from "project.ato"

        # Construct board path
        # Standard atopile layout: elec/layout/<build_name>/<base_name>.kicad_pcb
        board_path = self.root / "elec" / "layout" / build_name / f"{base_name}.kicad_pcb"

        # Also check for legacy layout structure
        if not board_path.exists():
            # Try: elec/layout/default/<base_name>.kicad_pcb
            alt_path = self.root / "elec" / "layout" / "default" / f"{base_name}.kicad_pcb"
            if alt_path.exists():
                return alt_path

        return board_path

    def get_ato_source_path(self, build_name: str = "default") -> Optional[Path]:
        """Get path to the main .ato source file for a build."""
        entry = self.get_entry_point(build_name)
        if not entry:
            return None

        file_part, _, _ = entry.partition(':')
        return self.root / file_part

    def load_board(self, build_name: str = "default") -> Board:
        """
        Load an atopile project board.

        This loads the generated KiCad board and enriches it with
        atopile metadata (component values, module hierarchy).

        Args:
            build_name: Build configuration to load

        Returns:
            Board instance with atopile metadata applied
        """
        board_path = self.get_board_path(build_name)

        if not board_path.exists():
            raise FileNotFoundError(
                f"Board file not found: {board_path}\n"
                f"Run 'ato build' first to generate the board."
            )

        # Load via KiCad adapter
        board = Board.from_kicad(board_path)

        # Enrich with atopile metadata
        self._apply_component_metadata(board)
        self._apply_module_hierarchy(board, build_name)

        return board

    def _apply_component_metadata(self, board: Board):
        """
        Apply component metadata from ato-lock.yaml to board.

        This adds component values, MPNs, and other metadata that
        can improve module detection and constraint inference.
        """
        if not self.lock_data:
            return

        components_data = self.lock_data.get("components", {})

        for ref, comp in board.components.items():
            if ref in components_data:
                meta = components_data[ref]

                # Update component properties
                if "value" in meta and not comp.value:
                    comp.value = meta["value"]

                # Store additional metadata in properties
                if "mpn" in meta:
                    comp.properties["mpn"] = meta["mpn"]
                if "package" in meta:
                    comp.properties["package"] = meta["package"]
                if "manufacturer" in meta:
                    comp.properties["manufacturer"] = meta["manufacturer"]

    def _apply_module_hierarchy(self, board: Board, build_name: str = "default"):
        """
        Apply module hierarchy from .ato files to board components.

        This adds module grouping information that can be used for
        automatic GroupingConstraints.
        """
        ato_path = self.get_ato_source_path(build_name)
        if not ato_path or not ato_path.exists():
            return

        # Parse module hierarchy
        parser = AtopileModuleParser()
        hierarchy = parser.parse_file(ato_path)

        # Apply to components
        for module in hierarchy.values():
            for comp_ref in module.components:
                if comp_ref in board.components:
                    comp = board.components[comp_ref]
                    comp.properties["ato_module"] = module.name
                    if module.module_type:
                        comp.properties["ato_module_type"] = module.module_type

    def get_component_metadata(self, reference: str) -> Optional[ComponentMetadata]:
        """Get metadata for a specific component."""
        if not self.lock_data:
            return None

        components = self.lock_data.get("components", {})
        if reference not in components:
            return None

        data = components[reference]
        return ComponentMetadata(
            reference=reference,
            mpn=data.get("mpn"),
            value=data.get("value"),
            package=data.get("package"),
            manufacturer=data.get("manufacturer"),
            description=data.get("description"),
        )

    def infer_constraints(self, board: Board) -> List[Tuple[str, str, float]]:
        """
        Infer placement constraints from atopile patterns.

        Looks for common patterns that imply constraints:
        - Decoupling caps near ICs
        - Crystals near MCUs
        - ESD protection near connectors

        Returns:
            List of (constraint_type, description, confidence) tuples
        """
        constraints = []

        # Look for decoupling capacitors
        for ref, comp in board.components.items():
            if ref.startswith('C'):
                value = comp.value.lower() if comp.value else ""
                # Common decoupling values
                if any(v in value for v in ['100n', '0.1u', '10u', '1u', '4.7u']):
                    # Find nearest IC
                    nearest_ic = self._find_nearest_ic(board, comp)
                    if nearest_ic:
                        constraints.append((
                            "proximity",
                            f"Keep {ref} close to {nearest_ic} (decoupling)",
                            0.9
                        ))

        # Look for crystals
        for ref, comp in board.components.items():
            if ref.startswith('Y') or ref.startswith('X'):
                # Find MCU/oscillator
                nearest_ic = self._find_nearest_ic(board, comp)
                if nearest_ic:
                    constraints.append((
                        "proximity",
                        f"Keep {ref} close to {nearest_ic} (crystal)",
                        0.95
                    ))

        return constraints

    def _find_nearest_ic(self, board: Board, comp: Component) -> Optional[str]:
        """Find the nearest IC to a component based on net connectivity."""
        # Get nets connected to this component
        comp_nets = comp.get_connected_nets()

        # Find ICs sharing nets
        for net_name in comp_nets:
            net = board.get_net(net_name)
            if not net:
                continue

            for ref in net.get_component_refs():
                if ref == comp.reference:
                    continue
                other = board.get_component(ref)
                if other and ref.startswith('U'):
                    return ref

        return None


class AtopileModuleParser:
    """
    Parse .ato files to extract module hierarchy.

    This is a simplified parser that extracts module definitions
    and component instantiations without fully parsing the atopile DSL.
    """

    # Regex patterns for parsing .ato files
    MODULE_PATTERN = re.compile(r'^(module|component)\s+(\w+):', re.MULTILINE)
    COMPONENT_PATTERN = re.compile(r'(\w+)\s*=\s*new\s+(\w+)')
    IMPORT_PATTERN = re.compile(r'from\s+"([^"]+)"\s+import\s+(.+)')

    # Module type inference from names
    TYPE_KEYWORDS = {
        'power': ['power', 'supply', 'regulator', 'dcdc', 'ldo', 'buck', 'boost'],
        'sensor': ['sensor', 'accel', 'gyro', 'temp', 'humidity', 'pressure'],
        'rf': ['rf', 'antenna', 'bluetooth', 'wifi', 'radio', 'lora'],
        'mcu': ['mcu', 'micro', 'processor', 'esp32', 'stm32', 'rp2040'],
        'connector': ['connector', 'usb', 'uart', 'i2c', 'spi', 'jtag'],
        'led': ['led', 'indicator', 'status'],
        'crystal': ['crystal', 'oscillator', 'xtal', 'clock'],
    }

    def parse_file(self, ato_path: Path) -> Dict[str, ModuleHierarchy]:
        """
        Parse an .ato file and extract module hierarchy.

        Args:
            ato_path: Path to .ato file

        Returns:
            Dictionary mapping module names to their hierarchy
        """
        if not ato_path.exists():
            return {}

        content = ato_path.read_text()
        return self.parse_content(content, ato_path.stem)

    def parse_content(self, content: str, root_name: str = "root") -> Dict[str, ModuleHierarchy]:
        """Parse .ato content string."""
        modules = {}
        current_module: Optional[ModuleHierarchy] = None
        indent_stack: List[Tuple[int, ModuleHierarchy]] = []

        lines = content.split('\n')

        for line in lines:
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            indent = len(line) - len(line.lstrip())

            # Check for module definition
            module_match = self.MODULE_PATTERN.match(stripped)
            if module_match:
                kind, name = module_match.groups()

                module = ModuleHierarchy(
                    name=name,
                    module_type=self._infer_module_type(name),
                )

                # Handle nesting
                while indent_stack and indent_stack[-1][0] >= indent:
                    indent_stack.pop()

                if indent_stack:
                    parent = indent_stack[-1][1]
                    module.parent = parent.name
                    parent.submodules.append(module)

                modules[name] = module
                current_module = module
                indent_stack.append((indent, module))
                continue

            # Check for component instantiation
            if current_module:
                comp_match = self.COMPONENT_PATTERN.search(stripped)
                if comp_match:
                    instance_name, comp_type = comp_match.groups()
                    # Store as potential component reference
                    current_module.components.append(instance_name)

        return modules

    def _infer_module_type(self, name: str) -> Optional[str]:
        """Infer module type from its name."""
        name_lower = name.lower()

        for module_type, keywords in self.TYPE_KEYWORDS.items():
            if any(kw in name_lower for kw in keywords):
                return module_type

        return None


def detect_board_source(path: Path) -> Tuple[str, Path]:
    """
    Detect whether a path is a KiCad board or atopile project.

    Args:
        path: Path to check

    Returns:
        Tuple of (source_type, resolved_path) where:
        - source_type is "kicad" or "atopile"
        - resolved_path is the path to the .kicad_pcb file
    """
    path = Path(path).resolve()

    # Direct .kicad_pcb file
    if path.suffix == ".kicad_pcb":
        return ("kicad", path)

    # Directory - check for atopile project
    if path.is_dir():
        if (path / "ato.yaml").exists():
            loader = AtopileProjectLoader(path)
            board_path = loader.get_board_path()
            return ("atopile", board_path)

        # Check for .kicad_pcb files in directory
        kicad_files = list(path.glob("*.kicad_pcb"))
        if kicad_files:
            return ("kicad", kicad_files[0])

    # Check if path is inside an atopile project
    project_root = AtopileProjectLoader.find_project_root(path)
    if project_root:
        loader = AtopileProjectLoader(project_root)
        board_path = loader.get_board_path()
        return ("atopile", board_path)

    raise ValueError(
        f"Cannot determine board source for: {path}\n"
        f"Provide a .kicad_pcb file or atopile project directory."
    )


def load_board_auto(path: Path) -> Board:
    """
    Automatically load a board from either KiCad or atopile source.

    Args:
        path: Path to .kicad_pcb file or atopile project directory

    Returns:
        Board instance
    """
    source_type, board_path = detect_board_source(path)

    if source_type == "atopile":
        project_root = AtopileProjectLoader.find_project_root(path)
        if project_root:
            loader = AtopileProjectLoader(project_root)
            return loader.load_board()

    # Fall back to direct KiCad loading
    return Board.from_kicad(board_path)
