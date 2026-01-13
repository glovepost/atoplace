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

    @property
    def instance_to_ref_map(self) -> Dict[str, str]:
        """
        Build a mapping from atopile instance paths to KiCad reference designators.

        Handles multiple lock file formats:
        1. Root-level entries with 'designator' field: { "root.r1": { "designator": "R1" } }
        2. Entries under 'components' key: { "components": { "R1": { "value": "10k" } } }
        3. Entries with 'address' field: { "components": { "R1": { "address": "root.r1" } } }

        Short instance names (last path segment) are added only if globally unique.
        This prevents mis-association when different modules reuse the same instance
        name (e.g., multiple 'c_bulk' instances).
        """
        if not hasattr(self, '_instance_to_ref_map') or self._instance_to_ref_map is None:
            self._instance_to_ref_map = {}
            # Track (short_name -> [kicad_ref, ...]) to detect collisions
            short_name_candidates: Dict[str, List[str]] = {}

            # First pass: Collect all mappings and identify short name collisions
            # Format 1: Root-level entries with 'designator' field
            for key, data in self.lock_data.items():
                if key == 'components':
                    continue  # Handle separately below
                if isinstance(data, dict) and 'designator' in data:
                    instance_path = key
                    kicad_ref = data['designator']
                    self._instance_to_ref_map[instance_path] = kicad_ref
                    # Track short name candidate
                    short_name = instance_path.split('.')[-1] if '.' in instance_path else instance_path
                    if short_name not in short_name_candidates:
                        short_name_candidates[short_name] = []
                    short_name_candidates[short_name].append(kicad_ref)

            # Format 2 & 3: Entries under 'components' key
            components_data = self.lock_data.get('components', {})
            if isinstance(components_data, dict):
                for kicad_ref, comp_data in components_data.items():
                    if not isinstance(comp_data, dict):
                        continue

                    # If there's an 'address' field, use it as the instance path
                    if 'address' in comp_data:
                        instance_path = comp_data['address']
                        self._instance_to_ref_map[instance_path] = kicad_ref
                        # Track short name candidate
                        short_name = instance_path.split('.')[-1] if '.' in instance_path else instance_path
                        if short_name not in short_name_candidates:
                            short_name_candidates[short_name] = []
                        short_name_candidates[short_name].append(kicad_ref)

                    # Also map the KiCad ref to itself for direct lookups
                    # This helps when .ato files use the ref directly
                    if kicad_ref not in self._instance_to_ref_map:
                        self._instance_to_ref_map[kicad_ref] = kicad_ref
                    # Map lowercase version for case-insensitive matching
                    kicad_ref_lower = kicad_ref.lower()
                    if kicad_ref_lower not in self._instance_to_ref_map:
                        self._instance_to_ref_map[kicad_ref_lower] = kicad_ref

            # Second pass: Add short names ONLY if they're unique (no collision)
            for short_name, refs in short_name_candidates.items():
                if len(refs) == 1 and short_name not in self._instance_to_ref_map:
                    # Unique short name - safe to add as alias
                    self._instance_to_ref_map[short_name] = refs[0]
                elif len(refs) > 1:
                    # Log collision for debugging
                    logger.debug(
                        "Short name '%s' is ambiguous (maps to %s) - not adding alias",
                        short_name, refs
                    )

        return self._instance_to_ref_map

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
        Simple YAML parser for nested key-value structures.
        Falls back to this when PyYAML is not available.

        Supports up to 3 levels of nesting (sufficient for ato.yaml builds/paths).
        """
        result = {}
        # Stack of (indent, dict_ref) to track nesting
        stack = [(0, result)]

        for line in content.split('\n'):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Calculate indent level (spaces, 2 per level typically)
            indent = len(line) - len(line.lstrip())

            # Pop stack until we find parent at lower indent
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()

            current_dict = stack[-1][1]

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
                    current_dict[key] = value
                else:
                    # This is a section header - create nested dict
                    new_section = {}
                    current_dict[key] = new_section
                    stack.append((indent, new_section))

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

        The method uses multiple strategies to map atopile instance paths to KiCad refs:
        1. Primary: Use 'atopile_address' property set by atopile on KiCad footprints
        2. Fallback: Use instance_to_ref_map from ato-lock.yaml
        3. Fallback: Heuristic matching (instance name to ref case-insensitive)
        """
        ato_path = self.get_ato_source_path(build_name)
        if not ato_path or not ato_path.exists():
            return

        # Parse module hierarchy (now follows imports)
        parser = AtopileModuleParser()
        hierarchy = parser.parse_file(ato_path)

        if not hierarchy:
            logger.debug("No modules found in .ato file hierarchy")
            return

        # Build primary mapping from atopile_address property in KiCad components
        # This is the most reliable source as it's set directly by atopile
        address_to_ref = self._build_address_map(board)

        # Get the instance-to-ref mapping from lock file as fallback
        lock_ref_map = self.instance_to_ref_map

        # Track which components we've successfully mapped
        mapped_count = 0
        total_count = 0

        # Apply module info to components
        for module_path, module in hierarchy.items():
            # module_path is now the full qualified path (e.g., 'power.regulator')
            # For each component in this module, build the full path

            for instance_name in module.components:
                total_count += 1
                kicad_ref = None

                # Build the full qualified path for this component
                # e.g., for module path 'power.regulator' with component 'c1',
                # full path is 'power.regulator.c1'
                qualified_path = f"{module_path}.{instance_name}"

                # Also try simpler paths for backward compatibility
                simple_path = f"{module.name}.{instance_name}"

                # Strategy 1: Look up atopile_address (primary - most reliable)
                # Try full path first, then simple path, then just instance name
                for path in [qualified_path, simple_path, instance_name]:
                    if path in address_to_ref:
                        kicad_ref = address_to_ref[path]
                        break

                # Strategy 2: Lock file mapping
                if not kicad_ref:
                    for path in [qualified_path, simple_path, instance_name]:
                        if path in lock_ref_map:
                            kicad_ref = lock_ref_map[path]
                            break

                    # Also search for paths containing this instance name
                    if not kicad_ref:
                        for path, ref in lock_ref_map.items():
                            if path.endswith('.' + instance_name) or path == instance_name:
                                kicad_ref = ref
                                break

                # Strategy 3: Heuristic case-insensitive match
                if not kicad_ref:
                    for board_ref in board.components:
                        if board_ref.lower() == instance_name.lower():
                            kicad_ref = board_ref
                            break

                # Apply module info to the component
                if kicad_ref and kicad_ref in board.components:
                    comp = board.components[kicad_ref]
                    # Store the full module path for proper hierarchy
                    comp.properties["ato_module"] = module_path
                    comp.properties["ato_module_name"] = module.name  # local name
                    if module.module_type:
                        comp.properties["ato_module_type"] = module.module_type
                    if module.parent:
                        comp.properties["ato_parent_module"] = module.parent
                    mapped_count += 1

        logger.debug(f"Mapped {mapped_count}/{total_count} components to atopile modules")

    def _build_address_map(self, board: Board) -> Dict[str, str]:
        """
        Build a mapping from atopile_address to KiCad reference designator.

        Atopile sets the 'atopile_address' property on each footprint with
        the full instance path (e.g., 'accel.c_bulk', 'power.l_mcu').

        Returns:
            Dict mapping instance paths to KiCad refs
        """
        address_map = {}

        for ref, comp in board.components.items():
            if 'atopile_address' in comp.properties:
                address = comp.properties['atopile_address']
                address_map[address] = ref

                # Also map the leaf name for simpler lookups
                if '.' in address:
                    leaf_name = address.split('.')[-1]
                    # Don't override if leaf name already mapped (could be ambiguous)
                    if leaf_name not in address_map:
                        address_map[leaf_name] = ref

        if address_map:
            logger.debug(f"Built atopile address map with {len(address_map)} entries")

        return address_map

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
    It follows imports to build complete module hierarchies.
    """

    # Regex patterns for parsing .ato files
    MODULE_PATTERN = re.compile(r'^(module|component)\s+(\w+):', re.MULTILINE)
    COMPONENT_PATTERN = re.compile(r'(\w+)\s*=\s*new\s+(\w+)')
    IMPORT_PATTERN = re.compile(r'from\s+"([^"]+)"\s+import\s+(.+)')
    SIMPLE_IMPORT_PATTERN = re.compile(r'^import\s+(.+)', re.MULTILINE)

    # Module type inference from names
    TYPE_KEYWORDS = {
        'power': ['power', 'supply', 'regulator', 'dcdc', 'ldo', 'buck', 'boost'],
        'sensor': ['sensor', 'accel', 'gyro', 'temp', 'humidity', 'pressure', 'lis3dh', 'qmi', 'hdc'],
        'rf': ['rf', 'antenna', 'bluetooth', 'wifi', 'radio', 'lora', 'matching'],
        'mcu': ['mcu', 'micro', 'processor', 'esp32', 'stm32', 'rp2040', 'rak'],
        'connector': ['connector', 'usb', 'uart', 'i2c', 'spi', 'jtag', 'swd', 'debug'],
        'led': ['led', 'indicator', 'status'],
        'crystal': ['crystal', 'oscillator', 'xtal', 'clock'],
        'memory': ['eeprom', 'flash', 'memory', 'storage'],
    }

    def __init__(self):
        """Initialize the parser with caches for import resolution."""
        self._parsed_files: Dict[Path, Dict[str, ModuleHierarchy]] = {}
        self._imported_modules: Dict[str, ModuleHierarchy] = {}

    def parse_file(self, ato_path: Path) -> Dict[str, ModuleHierarchy]:
        """
        Parse an .ato file and extract module hierarchy, following imports.

        Args:
            ato_path: Path to .ato file

        Returns:
            Dictionary mapping module names to their hierarchy
        """
        # Clear caches for fresh parse
        self._parsed_files = {}
        self._imported_modules = {}

        return self._parse_file_recursive(ato_path.resolve())

    def _parse_file_recursive(self, ato_path: Path, depth: int = 0) -> Dict[str, ModuleHierarchy]:
        """Recursively parse an .ato file, following imports."""
        if not ato_path.exists():
            logger.debug(f"File not found: {ato_path}")
            return {}

        # Check cache to avoid re-parsing
        if ato_path in self._parsed_files:
            return self._parsed_files[ato_path]

        # Prevent infinite recursion
        if depth > 10:
            logger.warning(f"Max import depth exceeded parsing {ato_path}")
            return {}

        content = ato_path.read_text()

        # First, process imports to make imported modules available
        imports = self._extract_imports(content)
        for import_path, imported_names in imports:
            # Resolve path relative to current file
            resolved_path = self._resolve_import_path(ato_path, import_path)
            if resolved_path and resolved_path.exists():
                # Parse imported file recursively
                imported_modules = self._parse_file_recursive(resolved_path, depth + 1)
                # Store imported module definitions
                for name in imported_names:
                    name = name.strip()
                    if name in imported_modules:
                        self._imported_modules[name] = imported_modules[name]
                        logger.debug(f"Imported module {name} from {import_path}")

        # Now parse this file's content
        modules = self._parse_content_with_imports(content, ato_path.stem, ato_path.parent)

        # Cache the result
        self._parsed_files[ato_path] = modules

        return modules

    def _extract_imports(self, content: str) -> List[Tuple[str, List[str]]]:
        """Extract import statements from content."""
        imports = []

        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                continue

            # Match: from "path/file.ato" import ModuleName, AnotherModule
            match = self.IMPORT_PATTERN.match(stripped)
            if match:
                path, names = match.groups()
                name_list = [n.strip() for n in names.split(',')]
                imports.append((path, name_list))

        return imports

    def _resolve_import_path(self, current_file: Path, import_path: str) -> Optional[Path]:
        """Resolve an import path relative to the current file."""
        # Handle relative paths
        if import_path.startswith('./') or import_path.startswith('../'):
            return (current_file.parent / import_path).resolve()

        # Handle paths relative to project root
        # Try relative to current file's directory first
        candidate = current_file.parent / import_path
        if candidate.exists():
            return candidate.resolve()

        # Try going up directories to find project root (has ato.yaml)
        search_dir = current_file.parent
        while search_dir != search_dir.parent:
            candidate = search_dir / import_path
            if candidate.exists():
                return candidate.resolve()
            # Check if we found project root
            if (search_dir / "ato.yaml").exists():
                break
            search_dir = search_dir.parent

        logger.debug(f"Could not resolve import path: {import_path} from {current_file}")
        return None

    def _parse_content_with_imports(self, content: str, root_name: str, base_dir: Path) -> Dict[str, ModuleHierarchy]:
        """Parse .ato content string, resolving imported module types."""
        modules = {}
        current_module: Optional[ModuleHierarchy] = None
        indent_stack: List[Tuple[int, ModuleHierarchy]] = []
        # Track instances and their types for later resolution
        instance_types: Dict[str, str] = {}  # instance_name -> type_name

        lines = content.split('\n')

        for line in lines:
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Skip pragma and docstrings
            if stripped.startswith('#pragma') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            indent = len(line) - len(line.lstrip())

            # Helper to build full qualified path from indent stack
            def get_full_path(stack: List[Tuple[int, ModuleHierarchy]], name: str) -> str:
                """Build full qualified path from parent chain."""
                if not stack:
                    return name
                parent_names = [m.name for _, m in stack]
                return '.'.join(parent_names + [name])

            def get_parent_path(stack: List[Tuple[int, ModuleHierarchy]]) -> Optional[str]:
                """Get full path of the parent module."""
                if not stack:
                    return None
                parent_names = [m.name for _, m in stack]
                return '.'.join(parent_names)

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
                    # Store full parent path, not just immediate parent name
                    module.parent = get_parent_path(indent_stack)
                    parent.submodules.append(module)

                # Use full qualified path as key to avoid overwrites
                full_path = get_full_path(indent_stack, name)
                modules[full_path] = module
                current_module = module
                indent_stack.append((indent, module))
                continue

            # Check for component instantiation
            if current_module:
                comp_match = self.COMPONENT_PATTERN.search(stripped)
                if comp_match:
                    instance_name, comp_type = comp_match.groups()

                    # Check if this is an imported module type
                    if comp_type in self._imported_modules:
                        # This is a module instantiation - create a submodule entry
                        imported_module = self._imported_modules[comp_type]
                        # Get full parent path for this submodule
                        parent_path = get_full_path(indent_stack, "")[:-1] if indent_stack else None  # strip trailing dot
                        if not parent_path and current_module:
                            parent_path = current_module.name
                        submodule = ModuleHierarchy(
                            name=instance_name,  # Use instance name, not type name
                            module_type=imported_module.module_type or self._infer_module_type(comp_type),
                            parent=parent_path,
                            # Copy component list from imported module definition
                            components=list(imported_module.components),
                        )
                        current_module.submodules.append(submodule)
                        # Use full qualified path as key
                        full_instance_path = get_full_path(indent_stack, instance_name)
                        modules[full_instance_path] = submodule
                        instance_types[instance_name] = comp_type
                        logger.debug(f"Found module instance: {full_instance_path} of type {comp_type} with {len(imported_module.components)} components")
                    else:
                        # Regular component instantiation - store instance name
                        current_module.components.append(instance_name)
                        instance_types[instance_name] = comp_type

        return modules

    def parse_content(self, content: str, root_name: str = "root") -> Dict[str, ModuleHierarchy]:
        """Parse .ato content string (legacy method without import resolution)."""
        # Create a temporary parser instance for backward compatibility
        self._imported_modules = {}
        return self._parse_content_with_imports(content, root_name, Path('.'))

    def _infer_module_type(self, name: str) -> Optional[str]:
        """Infer module type from its name."""
        name_lower = name.lower()

        for module_type, keywords in self.TYPE_KEYWORDS.items():
            if any(kw in name_lower for kw in keywords):
                return module_type

        return None


def detect_board_source(path: Path, build_name: Optional[str] = None) -> Tuple[str, Path]:
    """
    Detect whether a path is a KiCad board or atopile project.

    Args:
        path: Path to check
        build_name: Optional build name for atopile projects (default: first available)

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
            # Use specified build or try default, then first available
            if build_name:
                board_path = loader.get_board_path(build_name)
            else:
                builds = loader.get_build_names()
                if "default" in builds:
                    board_path = loader.get_board_path("default")
                elif builds:
                    board_path = loader.get_board_path(builds[0])
                else:
                    raise ValueError(f"No builds defined in ato.yaml for: {path}")
            return ("atopile", board_path)

        # Check for .kicad_pcb files in directory
        kicad_files = list(path.glob("*.kicad_pcb"))
        if kicad_files:
            return ("kicad", kicad_files[0])

    # Check if path is inside an atopile project
    project_root = AtopileProjectLoader.find_project_root(path)
    if project_root:
        loader = AtopileProjectLoader(project_root)
        # Use specified build or try default, then first available
        if build_name:
            board_path = loader.get_board_path(build_name)
        else:
            builds = loader.get_build_names()
            if "default" in builds:
                board_path = loader.get_board_path("default")
            elif builds:
                board_path = loader.get_board_path(builds[0])
            else:
                raise ValueError(f"No builds defined in ato.yaml for: {project_root}")
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
