"""
Macro Context Generator

Provides high-level board context for LLM understanding:
- Executive Summary: Board stats, critical metrics
- Semantic Grid: 3x3 zone mapping of components
- Module Map: Hierarchical functional block tree
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import json

from ...board.abstraction import Board, Component


class Zone(Enum):
    """3x3 semantic zones for board regions."""
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


@dataclass
class NetStats:
    """Statistics for a single net."""
    name: str
    pad_count: int
    components: List[str]
    is_power: bool = False
    is_ground: bool = False


@dataclass
class BoardSummary:
    """Executive summary of board state."""
    component_count: int
    net_count: int
    layer_count: int
    board_width: float
    board_height: float
    unplaced_count: int
    locked_count: int
    power_nets: List[str]
    ground_nets: List[str]
    high_fanout_nets: List[NetStats]
    critical_issues: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class SemanticGrid:
    """3x3 zone mapping of components."""
    zones: Dict[str, List[str]]  # zone name -> component refs
    zone_counts: Dict[str, int]
    dense_zones: List[str]  # Zones with highest density

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class ModuleNode:
    """Node in module hierarchy tree."""
    name: str
    type: str  # power, mcu, rf, sensor, etc.
    components: List[str]
    children: List["ModuleNode"] = field(default_factory=list)


@dataclass
class ModuleMap:
    """Hierarchical module structure."""
    root: ModuleNode
    flat_modules: Dict[str, List[str]]  # module_name -> components

    def to_json(self) -> str:
        def node_to_dict(node: ModuleNode) -> dict:
            return {
                "name": node.name,
                "type": node.type,
                "components": node.components,
                "children": [node_to_dict(c) for c in node.children]
            }
        return json.dumps({
            "root": node_to_dict(self.root),
            "flat_modules": self.flat_modules
        }, indent=2)


class MacroContext:
    """
    High-level board context generator.

    Provides executive summaries and semantic organization
    for LLM understanding of board structure.
    """

    def __init__(self, board: Board):
        self.board = board
        self._bounds = self._calculate_bounds()

    def get_summary(self) -> BoardSummary:
        """Generate executive summary of board state."""
        # Count stats
        component_count = len(self.board.components)
        net_count = len(self.board.nets)
        layer_count = 2  # Default, could be from board metadata

        # Calculate board dimensions
        min_x, min_y, max_x, max_y = self._bounds
        board_width = max_x - min_x
        board_height = max_y - min_y

        # Count unplaced (outside bounds) and locked
        unplaced = 0
        locked = 0
        for comp in self.board.components.values():
            if comp.locked:
                locked += 1
            if not self._is_in_bounds(comp):
                unplaced += 1

        # Analyze nets
        power_nets = []
        ground_nets = []
        high_fanout = []

        for net_name, net in self.board.nets.items():
            pad_count = len(net.connections)
            components = list(set(comp_ref for comp_ref, _ in net.connections if comp_ref))

            name_lower = net_name.lower()
            is_power = any(p in name_lower for p in ['vcc', 'vdd', '3v3', '5v', '12v', 'vin'])
            is_ground = any(g in name_lower for g in ['gnd', 'vss', 'ground'])

            if is_power:
                power_nets.append(net_name)
            if is_ground:
                ground_nets.append(net_name)

            # High fanout = more than 10 connections
            if pad_count > 10:
                high_fanout.append(NetStats(
                    name=net_name,
                    pad_count=pad_count,
                    components=components[:10],  # Limit for readability
                    is_power=is_power,
                    is_ground=is_ground
                ))

        # Sort high fanout by pad count
        high_fanout.sort(key=lambda n: n.pad_count, reverse=True)

        # Identify critical issues
        issues = []
        if unplaced > 0:
            issues.append(f"{unplaced} components outside board bounds")
        if board_width > 200:
            issues.append(f"Board width ({board_width:.1f}mm) is very large")
        if board_height > 200:
            issues.append(f"Board height ({board_height:.1f}mm) is very large")

        return BoardSummary(
            component_count=component_count,
            net_count=net_count,
            layer_count=layer_count,
            board_width=board_width,
            board_height=board_height,
            unplaced_count=unplaced,
            locked_count=locked,
            power_nets=power_nets[:5],  # Limit
            ground_nets=ground_nets[:5],
            high_fanout_nets=high_fanout[:5],
            critical_issues=issues
        )

    def get_semantic_grid(self) -> SemanticGrid:
        """
        Map components to a 3x3 semantic grid.

        Returns mapping of zones to component lists.
        """
        min_x, min_y, max_x, max_y = self._bounds
        width = max_x - min_x
        height = max_y - min_y

        # Avoid division by zero
        if width <= 0:
            width = 1
        if height <= 0:
            height = 1

        # Zone thresholds (1/3 and 2/3 of each dimension)
        x_low = min_x + width / 3
        x_high = min_x + 2 * width / 3
        y_low = min_y + height / 3
        y_high = min_y + 2 * height / 3

        zones: Dict[str, List[str]] = {z.value: [] for z in Zone}

        for ref, comp in self.board.components.items():
            zone = self._get_zone(comp.x, comp.y, x_low, x_high, y_low, y_high)
            zones[zone.value].append(ref)

        # Calculate counts and find dense zones
        zone_counts = {z: len(refs) for z, refs in zones.items()}
        avg_count = sum(zone_counts.values()) / 9 if zone_counts else 0
        dense_zones = [z for z, c in zone_counts.items() if c > avg_count * 1.5]

        return SemanticGrid(
            zones=zones,
            zone_counts=zone_counts,
            dense_zones=dense_zones
        )

    def get_module_map(self) -> ModuleMap:
        """
        Generate hierarchical module structure.

        Uses board metadata and pattern detection to organize
        components into functional groups.
        """
        flat_modules: Dict[str, List[str]] = {}

        # Try to use atopile module data if available
        if hasattr(self.board, '_atopile_modules') and self.board._atopile_modules:
            for module_name, refs in self.board._atopile_modules.items():
                flat_modules[module_name] = list(refs)
        else:
            # Fall back to pattern detection
            flat_modules = self._detect_modules()

        # Build hierarchical tree
        root = ModuleNode(
            name="Board",
            type="root",
            components=[],
            children=[]
        )

        for module_name, refs in flat_modules.items():
            module_type = self._infer_module_type(module_name, refs)
            node = ModuleNode(
                name=module_name,
                type=module_type,
                components=refs
            )
            root.children.append(node)

        # Add uncategorized components
        all_in_modules = set()
        for refs in flat_modules.values():
            all_in_modules.update(refs)

        uncategorized = [r for r in self.board.components.keys() if r not in all_in_modules]
        if uncategorized:
            root.children.append(ModuleNode(
                name="Uncategorized",
                type="misc",
                components=uncategorized
            ))

        return ModuleMap(root=root, flat_modules=flat_modules)

    def _calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate board bounds from outline or components."""
        if self.board.outline and self.board.outline.polygon:
            xs = [p[0] for p in self.board.outline.polygon]
            ys = [p[1] for p in self.board.outline.polygon]
            return min(xs), min(ys), max(xs), max(ys)

        # Fall back to component extents
        if not self.board.components:
            return 0, 0, 100, 100

        min_x = min(c.x - c.width/2 for c in self.board.components.values())
        max_x = max(c.x + c.width/2 for c in self.board.components.values())
        min_y = min(c.y - c.height/2 for c in self.board.components.values())
        max_y = max(c.y + c.height/2 for c in self.board.components.values())

        return min_x, min_y, max_x, max_y

    def _is_in_bounds(self, comp: Component) -> bool:
        """Check if component is within board bounds."""
        min_x, min_y, max_x, max_y = self._bounds
        return min_x <= comp.x <= max_x and min_y <= comp.y <= max_y

    def _get_zone(self, x: float, y: float,
                  x_low: float, x_high: float,
                  y_low: float, y_high: float) -> Zone:
        """Determine which zone a point falls in."""
        # X position: left, center, right
        if x < x_low:
            x_zone = "left"
        elif x > x_high:
            x_zone = "right"
        else:
            x_zone = "center"

        # Y position: top, center, bottom (note: PCB Y may be inverted)
        if y < y_low:
            y_zone = "top"
        elif y > y_high:
            y_zone = "bottom"
        else:
            y_zone = "center"

        # Combine
        if y_zone == "center" and x_zone == "center":
            return Zone.CENTER
        elif y_zone == "top":
            if x_zone == "left":
                return Zone.TOP_LEFT
            elif x_zone == "right":
                return Zone.TOP_RIGHT
            else:
                return Zone.TOP_CENTER
        elif y_zone == "bottom":
            if x_zone == "left":
                return Zone.BOTTOM_LEFT
            elif x_zone == "right":
                return Zone.BOTTOM_RIGHT
            else:
                return Zone.BOTTOM_CENTER
        else:  # center row
            if x_zone == "left":
                return Zone.CENTER_LEFT
            else:
                return Zone.CENTER_RIGHT

    def _detect_modules(self) -> Dict[str, List[str]]:
        """Detect modules by analyzing component patterns and connectivity."""
        modules: Dict[str, List[str]] = {}

        # Group by reference prefix patterns
        prefix_groups: Dict[str, List[str]] = {}
        for ref in self.board.components.keys():
            # Extract prefix (letters before numbers)
            prefix = ""
            for c in ref:
                if c.isalpha():
                    prefix += c
                else:
                    break
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(ref)

        # Identify ICs and their associated passives
        ics = prefix_groups.get("U", []) + prefix_groups.get("IC", [])
        for ic_ref in ics:
            # Find connected components
            connected = self._find_connected_components(ic_ref)
            if connected:
                modules[f"Module_{ic_ref}"] = [ic_ref] + connected[:10]

        return modules

    def _find_connected_components(self, ref: str) -> List[str]:
        """Find components directly connected to a reference."""
        connected: Set[str] = set()

        comp = self.board.components.get(ref)
        if not comp:
            return []

        # Find nets this component is on
        for pad in comp.pads:
            if pad.net and pad.net in self.board.nets:
                net = self.board.nets[pad.net]
                for comp_ref, _ in net.connections:
                    if comp_ref and comp_ref != ref:
                        connected.add(comp_ref)

        return list(connected)

    def _infer_module_type(self, module_name: str, refs: List[str]) -> str:
        """Infer module type from name and component references."""
        name_lower = module_name.lower()

        # Check name patterns
        if any(p in name_lower for p in ['power', 'pwr', 'supply', 'reg', 'buck', 'boost']):
            return "power"
        if any(p in name_lower for p in ['rf', 'radio', 'antenna', 'match']):
            return "rf"
        if any(p in name_lower for p in ['mcu', 'cpu', 'proc', 'micro']):
            return "mcu"
        if any(p in name_lower for p in ['sensor', 'accel', 'gyro', 'temp', 'humid']):
            return "sensor"
        if any(p in name_lower for p in ['led', 'display', 'status']):
            return "indicator"
        if any(p in name_lower for p in ['conn', 'usb', 'swd', 'debug', 'jtag']):
            return "connector"
        if any(p in name_lower for p in ['eeprom', 'flash', 'memory', 'ram']):
            return "memory"
        if any(p in name_lower for p in ['i2c', 'spi', 'uart', 'bus']):
            return "interface"

        # Check component types in the module
        has_ic = any(r.startswith('U') or r.startswith('IC') for r in refs)
        has_inductor = any(r.startswith('L') for r in refs)
        has_crystal = any(r.startswith('Y') or r.startswith('X') for r in refs)

        if has_inductor and has_ic:
            return "power"
        if has_crystal:
            return "clock"

        return "generic"
