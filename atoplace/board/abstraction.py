"""
Board Abstraction Layer

Provides a unified interface for PCB board manipulation that works with
both KiCad native files and atopile-generated boards. This abstraction
allows the placement and routing engines to work independently of the
underlying board format.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import math


class Layer(Enum):
    """PCB layer identifiers."""
    TOP_COPPER = "F.Cu"
    BOTTOM_COPPER = "B.Cu"
    INNER1 = "In1.Cu"
    INNER2 = "In2.Cu"
    INNER3 = "In3.Cu"
    INNER4 = "In4.Cu"
    TOP_SILK = "F.SilkS"
    BOTTOM_SILK = "B.SilkS"
    TOP_MASK = "F.Mask"
    BOTTOM_MASK = "B.Mask"
    TOP_PASTE = "F.Paste"
    BOTTOM_PASTE = "B.Paste"
    TOP_COURTYARD = "F.CrtYd"
    BOTTOM_COURTYARD = "B.CrtYd"
    EDGE_CUTS = "Edge.Cuts"


@dataclass
class Pad:
    """Represents a component pad."""
    number: str
    x: float  # Relative to component center (mm)
    y: float
    width: float  # mm
    height: float
    shape: str = "rect"  # "rect", "circle", "oval", "roundrect"
    layer: Layer = Layer.TOP_COPPER
    net: Optional[str] = None
    drill: Optional[float] = None  # For through-hole pads

    def absolute_position(self, comp_x: float, comp_y: float,
                          rotation: float = 0) -> Tuple[float, float]:
        """Get absolute pad position given component position and rotation."""
        # Apply rotation
        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        rx = self.x * cos_r - self.y * sin_r
        ry = self.x * sin_r + self.y * cos_r
        return (comp_x + rx, comp_y + ry)


@dataclass
class Component:
    """Represents a PCB component/footprint."""
    reference: str  # e.g., "U1", "R1", "C1"
    footprint: str  # Footprint library:name
    value: str = ""
    x: float = 0.0  # mm
    y: float = 0.0
    rotation: float = 0.0  # degrees
    layer: Layer = Layer.TOP_COPPER
    pads: List[Pad] = field(default_factory=list)

    # Bounding box (relative to component center)
    width: float = 0.0  # mm
    height: float = 0.0

    # Component properties
    locked: bool = False
    dnp: bool = False  # Do Not Populate

    # Metadata
    properties: Dict[str, str] = field(default_factory=dict)

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get axis-aligned bounding box considering rotation.

        Returns:
            (min_x, min_y, max_x, max_y) in absolute coordinates
        """
        # Calculate corners
        hw, hh = self.width / 2, self.height / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

        # Rotate corners
        rad = math.radians(self.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)

        rotated = []
        for cx, cy in corners:
            rx = cx * cos_r - cy * sin_r + self.x
            ry = cx * sin_r + cy * cos_r + self.y
            rotated.append((rx, ry))

        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]

        return (min(xs), min(ys), max(xs), max(ys))

    def overlaps(self, other: 'Component', clearance: float = 0.0) -> bool:
        """Check if this component overlaps with another."""
        bb1 = self.get_bounding_box()
        bb2 = other.get_bounding_box()

        # Add clearance
        bb1 = (bb1[0] - clearance, bb1[1] - clearance,
               bb1[2] + clearance, bb1[3] + clearance)

        # Check overlap
        return not (bb1[2] < bb2[0] or bb2[2] < bb1[0] or
                    bb1[3] < bb2[1] or bb2[3] < bb1[1])

    def distance_to(self, other: 'Component') -> float:
        """Calculate center-to-center distance to another component."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def get_pad_by_number(self, number: str) -> Optional[Pad]:
        """Get a pad by its number/name."""
        for pad in self.pads:
            if pad.number == number:
                return pad
        return None

    def get_connected_nets(self) -> Set[str]:
        """Get all nets connected to this component."""
        return {pad.net for pad in self.pads if pad.net}


@dataclass
class Net:
    """Represents a PCB net (electrical connection)."""
    name: str
    code: int = 0  # Net code for KiCad

    # Connected pads: list of (component_ref, pad_number)
    connections: List[Tuple[str, str]] = field(default_factory=list)

    # Net class properties
    net_class: str = "Default"
    trace_width: Optional[float] = None  # mm, if specified
    clearance: Optional[float] = None

    # Special net flags
    is_power: bool = False
    is_ground: bool = False
    is_differential: bool = False
    diff_pair_net: Optional[str] = None  # Name of the paired net

    def add_connection(self, component_ref: str, pad_number: str):
        """Add a pad connection to this net."""
        conn = (component_ref, pad_number)
        if conn not in self.connections:
            self.connections.append(conn)

    def get_component_refs(self) -> Set[str]:
        """Get all component references connected to this net."""
        return {ref for ref, _ in self.connections}


@dataclass
class BoardOutline:
    """Represents the board edge/outline.

    Supports both simple rectangular boards and complex polygon outlines
    with optional holes (cutouts).
    """
    # Simple rectangular board (used when polygon is None)
    width: float = 100.0  # mm
    height: float = 100.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    # For complex shapes, store as polygon vertices [(x, y), ...]
    polygon: Optional[List[Tuple[float, float]]] = None

    # For boards with cutouts, store hole polygons
    holes: List[List[Tuple[float, float]]] = field(default_factory=list)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is within the board outline.

        Uses Ray Casting algorithm (even-odd rule) for polygon containment.
        Returns False if point is in a hole.
        """
        if self.polygon:
            # Offset the polygon inward by margin for boundary checking
            # For simplicity, we check the point and then verify it's at least
            # margin distance from any edge
            if not self._point_in_polygon(x, y, self.polygon):
                return False

            # Check margin distance from polygon edges
            if margin > 0 and not self._point_inside_margin(x, y, self.polygon, margin):
                return False

            # Check if point is in any hole
            for hole in self.holes:
                if self._point_in_polygon(x, y, hole):
                    return False

            return True

        # Simple rectangle check
        return (self.origin_x + margin <= x <= self.origin_x + self.width - margin and
                self.origin_y + margin <= y <= self.origin_y + self.height - margin)

    def _point_in_polygon(self, x: float, y: float,
                          polygon: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm for point-in-polygon test.

        Casts a ray from the point to the right and counts edge crossings.
        Odd number of crossings = inside, even = outside.
        """
        n = len(polygon)
        if n < 3:
            return False

        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # Check if ray from (x, y) going right crosses edge (i, j)
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def _point_inside_margin(self, x: float, y: float,
                             polygon: List[Tuple[float, float]],
                             margin: float) -> bool:
        """Check if point is at least margin distance from all polygon edges."""
        n = len(polygon)
        if n < 2:
            return True

        for i in range(n):
            j = (i + 1) % n
            x1, y1 = polygon[i]
            x2, y2 = polygon[j]

            # Calculate distance from point to line segment
            dist = self._point_to_segment_distance(x, y, x1, y1, x2, y2)
            if dist < margin:
                return False

        return True

    def _point_to_segment_distance(self, px: float, py: float,
                                   x1: float, y1: float,
                                   x2: float, y2: float) -> float:
        """Calculate shortest distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
        # Vector from segment start to point
        dx = px - x1
        dy = py - y1

        # Vector along segment
        sx = x2 - x1
        sy = y2 - y1

        # Segment length squared
        seg_len_sq = sx * sx + sy * sy

        if seg_len_sq < 1e-10:
            # Degenerate segment (point)
            return math.sqrt(dx * dx + dy * dy)

        # Parameter t for closest point on infinite line
        t = max(0.0, min(1.0, (dx * sx + dy * sy) / seg_len_sq))

        # Closest point on segment
        closest_x = x1 + t * sx
        closest_y = y1 + t * sy

        # Distance from point to closest point
        diff_x = px - closest_x
        diff_y = py - closest_y

        return math.sqrt(diff_x * diff_x + diff_y * diff_y)

    def get_edge(self, edge: str) -> float:
        """Get coordinate of specified edge.

        For polygons, returns the bounding box edge.
        """
        if self.polygon:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            if edge == "left":
                return min(xs)
            elif edge == "right":
                return max(xs)
            elif edge == "top":
                return min(ys)
            elif edge == "bottom":
                return max(ys)
            else:
                raise ValueError(f"Unknown edge: {edge}")

        if edge == "left":
            return self.origin_x
        elif edge == "right":
            return self.origin_x + self.width
        elif edge == "top":
            return self.origin_y
        elif edge == "bottom":
            return self.origin_y + self.height
        else:
            raise ValueError(f"Unknown edge: {edge}")

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box (min_x, min_y, max_x, max_y)."""
        if self.polygon:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            return (min(xs), min(ys), max(xs), max(ys))
        return (self.origin_x, self.origin_y,
                self.origin_x + self.width, self.origin_y + self.height)


@dataclass
class Board:
    """
    Unified board representation.

    This is the main interface for board manipulation, abstracting away
    the differences between KiCad and atopile board formats.
    """

    # Board identification
    name: str = ""
    source_file: Optional[Path] = None

    # Board geometry
    outline: BoardOutline = field(default_factory=BoardOutline)
    layer_count: int = 2

    # Components and connectivity
    components: Dict[str, Component] = field(default_factory=dict)
    nets: Dict[str, Net] = field(default_factory=dict)

    # Design rules
    default_trace_width: float = 0.25  # mm
    default_clearance: float = 0.2
    default_via_drill: float = 0.3
    default_via_diameter: float = 0.6

    # --- Component Management ---

    def add_component(self, component: Component):
        """Add a component to the board."""
        self.components[component.reference] = component

    def get_component(self, reference: str) -> Optional[Component]:
        """Get a component by reference designator."""
        return self.components.get(reference)

    def remove_component(self, reference: str) -> Optional[Component]:
        """Remove and return a component."""
        return self.components.pop(reference, None)

    def get_components_by_prefix(self, prefix: str) -> List[Component]:
        """Get all components with a given reference prefix (e.g., 'R', 'C', 'U')."""
        return [c for c in self.components.values()
                if c.reference.startswith(prefix)]

    # --- Net Management ---

    def add_net(self, net: Net):
        """Add a net to the board."""
        self.nets[net.name] = net

    def get_net(self, name: str) -> Optional[Net]:
        """Get a net by name."""
        return self.nets.get(name)

    def get_power_nets(self) -> List[Net]:
        """Get all power nets."""
        return [n for n in self.nets.values() if n.is_power]

    def get_ground_nets(self) -> List[Net]:
        """Get all ground nets."""
        return [n for n in self.nets.values() if n.is_ground]

    # --- Connectivity Analysis ---

    def get_connected_components(self, net_name: str) -> List[Component]:
        """Get all components connected to a specific net."""
        net = self.nets.get(net_name)
        if not net:
            return []
        return [self.components[ref] for ref in net.get_component_refs()
                if ref in self.components]

    def get_nets_between(self, ref1: str, ref2: str) -> List[Net]:
        """Get all nets connecting two components."""
        c1 = self.components.get(ref1)
        c2 = self.components.get(ref2)
        if not c1 or not c2:
            return []

        nets1 = c1.get_connected_nets()
        nets2 = c2.get_connected_nets()
        common = nets1 & nets2

        return [self.nets[name] for name in common if name in self.nets]

    # --- Placement Utilities ---

    def find_overlaps(self, clearance: float = 0.25) -> List[Tuple[str, str, float]]:
        """
        Find all overlapping component pairs.

        Returns:
            List of (ref1, ref2, overlap_distance) tuples
        """
        overlaps = []
        refs = list(self.components.keys())

        for i, ref1 in enumerate(refs):
            c1 = self.components[ref1]
            for ref2 in refs[i+1:]:
                c2 = self.components[ref2]
                if c1.overlaps(c2, clearance):
                    # Calculate overlap distance
                    dist = c1.distance_to(c2)
                    overlaps.append((ref1, ref2, dist))

        return overlaps

    def get_placement_bounds(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of all placed components."""
        if not self.components:
            return (0, 0, 0, 0)

        all_bounds = [c.get_bounding_box() for c in self.components.values()]

        return (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        )

    def move_component(self, reference: str, x: float, y: float,
                       rotation: Optional[float] = None):
        """Move a component to a new position."""
        comp = self.components.get(reference)
        if comp:
            comp.x = x
            comp.y = y
            if rotation is not None:
                comp.rotation = rotation

    # --- Statistics ---

    def get_stats(self) -> Dict:
        """Get board statistics."""
        return {
            "component_count": len(self.components),
            "net_count": len(self.nets),
            "power_net_count": len(self.get_power_nets()),
            "layer_count": self.layer_count,
            "board_width": self.outline.width,
            "board_height": self.outline.height,
            "board_area": self.outline.width * self.outline.height,
        }

    # --- I/O Methods (to be implemented by subclasses or adapters) ---

    @classmethod
    def from_kicad(cls, path: Path) -> 'Board':
        """Load board from KiCad file."""
        from .kicad_adapter import load_kicad_board
        return load_kicad_board(path)

    def to_kicad(self, path: Path):
        """Save board to KiCad file."""
        from .kicad_adapter import save_kicad_board
        save_kicad_board(self, path)

    def __repr__(self) -> str:
        return (f"Board(name={self.name!r}, "
                f"components={len(self.components)}, "
                f"nets={len(self.nets)}, "
                f"size={self.outline.width}x{self.outline.height}mm)")
