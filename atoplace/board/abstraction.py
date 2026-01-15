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
    rotation: float = 0.0  # Pad rotation relative to component (degrees)

    def absolute_position(self, comp_x: float, comp_y: float,
                          comp_rotation: float = 0) -> Tuple[float, float]:
        """Get absolute pad position given component position and rotation."""
        # Apply component rotation to pad position
        rad = math.radians(comp_rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        rx = self.x * cos_r - self.y * sin_r
        ry = self.x * sin_r + self.y * cos_r
        return (comp_x + rx, comp_y + ry)

    def absolute_rotation(self, comp_rotation: float = 0) -> float:
        """Get absolute pad rotation considering component rotation."""
        return (self.rotation + comp_rotation) % 360

    def get_bounding_box(self, comp_x: float, comp_y: float,
                         comp_rotation: float = 0) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box for this pad in absolute coordinates.

        Returns:
            (min_x, min_y, max_x, max_y) accounting for pad and component rotation
        """
        # Get absolute position
        abs_x, abs_y = self.absolute_position(comp_x, comp_y, comp_rotation)

        # For circular pads, rotation doesn't matter
        if self.shape == "circle":
            r = max(self.width, self.height) / 2
            return (abs_x - r, abs_y - r, abs_x + r, abs_y + r)

        # For rectangular/oval pads, calculate rotated bounding box
        total_rotation = self.absolute_rotation(comp_rotation)
        rad = math.radians(total_rotation)
        cos_r = abs(math.cos(rad))
        sin_r = abs(math.sin(rad))

        # AABB half-dimensions of rotated rectangle
        half_w = (self.width / 2) * cos_r + (self.height / 2) * sin_r
        half_h = (self.width / 2) * sin_r + (self.height / 2) * cos_r

        return (abs_x - half_w, abs_y - half_h, abs_x + half_w, abs_y + half_h)


@dataclass
class RefDesText:
    """Represents reference designator text positioning.

    Stores the position and properties of the reference designator text
    relative to the component. This allows tracking and repositioning
    of ref des text during placement to avoid overlaps.
    """
    # Position relative to component center (mm)
    offset_x: float = 0.0
    offset_y: float = 0.0

    # Text properties
    rotation: float = 0.0  # Degrees, relative to component
    size: float = 1.0  # Text height in mm
    thickness: float = 0.15  # Stroke thickness in mm

    # Visibility and layer
    visible: bool = True
    layer: Layer = Layer.TOP_SILK

    def get_absolute_position(self, comp_x: float, comp_y: float,
                              comp_rotation: float = 0) -> Tuple[float, float]:
        """Get absolute text position given component position and rotation."""
        rad = math.radians(comp_rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        rx = self.offset_x * cos_r - self.offset_y * sin_r
        ry = self.offset_x * sin_r + self.offset_y * cos_r
        return (comp_x + rx, comp_y + ry)

    def get_bounding_box(self, comp_x: float, comp_y: float,
                         comp_rotation: float, text: str) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box for the text in absolute coordinates.

        Args:
            comp_x, comp_y: Component center position
            comp_rotation: Component rotation in degrees
            text: The text string (for width calculation)

        Returns:
            (min_x, min_y, max_x, max_y)
        """
        abs_x, abs_y = self.get_absolute_position(comp_x, comp_y, comp_rotation)

        # Estimate text dimensions (approximate: width ~= 0.6 * height per char)
        text_width = len(text) * self.size * 0.6
        text_height = self.size

        # Apply text rotation
        total_rotation = (self.rotation + comp_rotation) % 360
        rad = math.radians(total_rotation)
        cos_r = abs(math.cos(rad))
        sin_r = abs(math.sin(rad))

        # AABB half-dimensions of rotated text box
        half_w = (text_width / 2) * cos_r + (text_height / 2) * sin_r
        half_h = (text_width / 2) * sin_r + (text_height / 2) * cos_r

        return (abs_x - half_w, abs_y - half_h, abs_x + half_w, abs_y + half_h)


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

    # Origin offset from KiCad footprint origin to centroid (in local coordinates)
    # This allows x,y to represent the centroid while preserving KiCad's origin
    # for accurate save-back. The offset is in unrotated local coordinates.
    origin_offset_x: float = 0.0
    origin_offset_y: float = 0.0

    # Component properties
    locked: bool = False
    dnp: bool = False  # Do Not Populate

    # Reference designator text positioning
    ref_des_text: Optional[RefDesText] = None

    # Metadata
    properties: Dict[str, str] = field(default_factory=dict)

    @property
    def is_through_hole(self) -> bool:
        """Check if component has through-hole pads."""
        # Heuristic: Battery holders with "TH" in name might be SMD on bottom
        if self.footprint and ("BAT-TH" in self.footprint or "CR2032" in self.footprint):
            return False
        return any(p.drill is not None and p.drill > 0 for p in self.pads)

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get axis-aligned bounding box of component body considering rotation.

        Returns:
            (min_x, min_y, max_x, max_y) in absolute coordinates

        Note: This uses the component width/height (body dimensions). For a
        bounding box that includes pad extents, use get_bounding_box_with_pads().
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

    def get_bounding_box_with_pads(self) -> Tuple[float, float, float, float]:
        """
        Get axis-aligned bounding box including all pad extents.

        This method computes the union of the component body bounding box
        and all pad bounding boxes, accounting for pads that may protrude
        beyond the component body (e.g., edge-mounted connectors, irregular
        footprints).

        Returns:
            (min_x, min_y, max_x, max_y) in absolute coordinates
        """
        # Start with body bounding box
        body_bbox = self.get_bounding_box()
        min_x, min_y, max_x, max_y = body_bbox

        # Expand to include all pads
        for pad in self.pads:
            pad_bbox = pad.get_bounding_box(self.x, self.y, self.rotation)
            min_x = min(min_x, pad_bbox[0])
            min_y = min(min_y, pad_bbox[1])
            max_x = max(max_x, pad_bbox[2])
            max_y = max(max_y, pad_bbox[3])

        return (min_x, min_y, max_x, max_y)

    def overlaps(self, other: 'Component', clearance: float = 0.0,
                 include_pads: bool = False) -> bool:
        """Check if this component overlaps with another.

        Args:
            other: Another component to check against
            clearance: Minimum clearance to enforce (mm)
            include_pads: If True, use bounding boxes that include pad extents.
                         This catches overlaps where pads protrude beyond body.
        """
        if include_pads:
            bb1 = self.get_bounding_box_with_pads()
            bb2 = other.get_bounding_box_with_pads()
        else:
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

    When has_outline is False, boundary checks should be skipped as no
    explicit outline was defined for the board.
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

    # Flag indicating if an explicit outline was defined
    # When False, boundary checks should be skipped
    has_outline: bool = True

    # Flag indicating if outline was auto-generated from component bounds
    auto_generated: bool = False

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is within the board outline.

        Uses Ray Casting algorithm (even-odd rule) for polygon containment.
        Returns False if point is in a hole or too close to hole/boundary edges.
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
                # Also check margin distance from hole edges (cutout clearance)
                if margin > 0 and not self._point_outside_hole_margin(x, y, hole, margin):
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

    def _point_outside_hole_margin(self, x: float, y: float,
                                   hole: List[Tuple[float, float]],
                                   margin: float) -> bool:
        """Check if point is at least margin distance from all hole edges.

        This enforces cutout clearance - components must maintain distance
        from board cutouts/holes, not just stay outside them.
        """
        n = len(hole)
        if n < 2:
            return True

        for i in range(n):
            j = (i + 1) % n
            x1, y1 = hole[i]
            x2, y2 = hole[j]

            # Calculate distance from point to hole edge
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

    def find_overlaps(self, clearance: float = 0.25,
                       check_layers: bool = False,
                       include_pads: bool = False) -> List[Tuple[str, str, float]]:
        """
        Find all overlapping component pairs.

        Args:
            clearance: Minimum clearance between components (mm)
            check_layers: If True, only report overlaps between components on
                         the same layer (top vs bottom). Components on opposite
                         sides of the board are allowed to overlap.
            include_pads: If True, use bounding boxes that include pad extents.
                         This catches overlaps where pads protrude beyond body.

        Returns:
            List of (ref1, ref2, penetration_depth) tuples where penetration_depth
            is the minimum distance the components need to move apart to clear
            (negative means they need to move that far to meet clearance requirement)
        """
        overlaps = []
        refs = list(self.components.keys())

        for i, ref1 in enumerate(refs):
            c1 = self.components[ref1]
            if include_pads:
                bb1 = c1.get_bounding_box_with_pads()
            else:
                bb1 = c1.get_bounding_box()
            # Expand by clearance
            bb1 = (bb1[0] - clearance, bb1[1] - clearance,
                   bb1[2] + clearance, bb1[3] + clearance)

            for ref2 in refs[i+1:]:
                c2 = self.components[ref2]

                # Skip if on opposite layers (top vs bottom)
                if check_layers:
                    # Treat through-hole as existing on all layers
                    if not (c1.is_through_hole or c2.is_through_hole):
                        layer1_is_top = c1.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                                      Layer.TOP_MASK, Layer.TOP_PASTE,
                                                      Layer.TOP_COURTYARD)
                        layer2_is_top = c2.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                                      Layer.TOP_MASK, Layer.TOP_PASTE,
                                                      Layer.TOP_COURTYARD)
                        if layer1_is_top != layer2_is_top:
                            continue  # Components on opposite sides, skip

                if include_pads:
                    bb2 = c2.get_bounding_box_with_pads()
                else:
                    bb2 = c2.get_bounding_box()

                # Calculate overlap on each axis
                overlap_x = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
                overlap_y = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])

                # Both axes must overlap for a true overlap
                if overlap_x > 0 and overlap_y > 0:
                    # Penetration depth is the minimum of the two overlaps
                    # (minimum translation vector magnitude)
                    penetration = min(overlap_x, overlap_y)
                    overlaps.append((ref1, ref2, penetration))

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
        # Calculate area properly for polygon outlines
        board_area = self.calculate_board_area()

        return {
            "component_count": len(self.components),
            "net_count": len(self.nets),
            "power_net_count": len(self.get_power_nets()),
            "layer_count": self.layer_count,
            "board_width": self.outline.width,
            "board_height": self.outline.height,
            "board_area": board_area,
        }

    def calculate_board_area(self) -> float:
        """Calculate actual board area, accounting for polygon outlines and holes.

        Uses the shoelace formula for polygon area calculation.
        Returns area in mm².

        This is a public method for use by validation and analysis tools.
        """
        if self.outline.polygon:
            # Calculate polygon area using shoelace formula
            area = self._polygon_area(self.outline.polygon)

            # Subtract hole areas
            for hole in self.outline.holes:
                area -= self._polygon_area(hole)

            return abs(area)

        # Fall back to rectangular area
        return self.outline.width * self.outline.height

    def _polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon using shoelace formula."""
        n = len(polygon)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]

        return abs(area) / 2.0

    # --- I/O Methods (to be implemented by subclasses or adapters) ---

    @classmethod
    def from_kicad(cls, path: Path) -> 'Board':
        """Load board from KiCad file."""
        from .kicad_adapter import load_kicad_board
        return load_kicad_board(path)

    def generate_outline_from_components(self, margin: float = 5.0) -> BoardOutline:
        """Generate a rectangular board outline from component positions.

        Creates a bounding box around all components with the specified margin.
        Useful for boards without an explicit Edge.Cuts outline.

        Args:
            margin: Distance (mm) to add around components on all sides

        Returns:
            New BoardOutline based on component positions
        """
        if not self.components:
            # Return a minimal default outline
            return BoardOutline(
                width=50.0,
                height=50.0,
                origin_x=0.0,
                origin_y=0.0,
                has_outline=True,
                auto_generated=True,
            )

        # Find bounding box of all components (including pads)
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for comp in self.components.values():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            bbox = comp.get_bounding_box_with_pads()
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

        # Add margin
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin

        return BoardOutline(
            width=max_x - min_x,
            height=max_y - min_y,
            origin_x=min_x,
            origin_y=min_y,
            has_outline=True,
            auto_generated=True,
        )

    def compact_outline(
        self,
        initial_margin: float = 10.0,
        min_margin: float = 1.0,
        clearance: float = 0.25,
        shrink_step: float = 0.5,
        max_iterations: int = 100,
    ) -> BoardOutline:
        """Generate a compacted board outline by iteratively shrinking until infeasible.

        This method:
        1. Generates an initial bounding box with initial_margin
        2. Iteratively shrinks the outline
        3. Checks if placement is still feasible (no boundary violations)
        4. Reverts to last feasible size when placement becomes infeasible

        Args:
            initial_margin: Starting margin around components (mm)
            min_margin: Minimum margin to maintain (mm)
            clearance: Minimum clearance from board edge (mm)
            shrink_step: Amount to shrink each iteration (mm)
            max_iterations: Maximum shrink iterations

        Returns:
            Compacted BoardOutline at the smallest feasible size
        """
        if not self.components:
            return self.generate_outline_from_components(margin=initial_margin)

        # Find component bounds (ignoring DNP)
        comp_min_x = float('inf')
        comp_min_y = float('inf')
        comp_max_x = float('-inf')
        comp_max_y = float('-inf')

        for comp in self.components.values():
            if comp.dnp:
                continue
            bbox = comp.get_bounding_box_with_pads()
            comp_min_x = min(comp_min_x, bbox[0])
            comp_min_y = min(comp_min_y, bbox[1])
            comp_max_x = max(comp_max_x, bbox[2])
            comp_max_y = max(comp_max_y, bbox[3])

        # Minimum required dimensions (components + clearance on each side)
        min_width = (comp_max_x - comp_min_x) + 2 * clearance
        min_height = (comp_max_y - comp_min_y) + 2 * clearance

        # Start with initial margin
        current_margin = initial_margin
        last_feasible_margin = initial_margin

        for _ in range(max_iterations):
            # Calculate outline at current margin
            test_outline = BoardOutline(
                width=(comp_max_x - comp_min_x) + 2 * current_margin,
                height=(comp_max_y - comp_min_y) + 2 * current_margin,
                origin_x=comp_min_x - current_margin,
                origin_y=comp_min_y - current_margin,
                has_outline=True,
                auto_generated=True,
            )

            # Check feasibility: all components within bounds with clearance
            feasible = self._check_outline_feasibility(test_outline, clearance)

            if feasible:
                last_feasible_margin = current_margin
                # Try shrinking further
                current_margin -= shrink_step
                if current_margin < min_margin:
                    break
            else:
                # Infeasible - stop and use last feasible
                break

        # Return the last feasible outline
        return BoardOutline(
            width=(comp_max_x - comp_min_x) + 2 * last_feasible_margin,
            height=(comp_max_y - comp_min_y) + 2 * last_feasible_margin,
            origin_x=comp_min_x - last_feasible_margin,
            origin_y=comp_min_y - last_feasible_margin,
            has_outline=True,
            auto_generated=True,
        )

    def _check_outline_feasibility(
        self, outline: BoardOutline, clearance: float
    ) -> bool:
        """Check if all components fit within the given outline with clearance.

        Args:
            outline: The board outline to test
            clearance: Minimum distance from board edge

        Returns:
            True if all non-DNP components fit within the outline
        """
        for comp in self.components.values():
            if comp.dnp:
                continue

            bbox = comp.get_bounding_box_with_pads()

            # Check if component is within outline bounds with clearance
            if bbox[0] < outline.origin_x + clearance:
                return False
            if bbox[1] < outline.origin_y + clearance:
                return False
            if bbox[2] > outline.origin_x + outline.width - clearance:
                return False
            if bbox[3] > outline.origin_y + outline.height - clearance:
                return False

        return True

    def reposition_ref_des_text(self, clearance: float = 0.2, pad_clearance: float = 0.15) -> int:
        """Reposition reference designator text to avoid overlaps.

        This method repositions ref des text for each component to:
        1. Avoid overlapping with pads on the same component
        2. Avoid overlapping with pads on nearby components
        3. Stay within a reasonable distance from the component

        The algorithm tries multiple candidate positions around the component
        and selects the one with the least overlap with pads/other text.

        Args:
            clearance: Minimum clearance between text and component body (mm)
            pad_clearance: Minimum clearance between text and pads (mm)

        Returns:
            Number of ref des texts that were repositioned
        """
        repositioned_count = 0

        for comp in self.components.values():
            if comp.dnp or not comp.ref_des_text:
                continue

            # Get current text position and bounding box
            ref_text = comp.ref_des_text
            current_bbox = ref_text.get_bounding_box(
                comp.x, comp.y, comp.rotation, comp.reference
            )

            # Check if current position has conflicts
            has_conflict = self._check_text_conflicts(
                comp, current_bbox, pad_clearance
            )

            if not has_conflict:
                continue

            # Try candidate positions around the component
            best_position = None
            best_score = float('inf')

            candidates = self._generate_text_candidates(comp, ref_text, clearance)

            for offset_x, offset_y in candidates:
                # Create temporary text with new position
                test_text = RefDesText(
                    offset_x=offset_x,
                    offset_y=offset_y,
                    rotation=ref_text.rotation,
                    size=ref_text.size,
                    thickness=ref_text.thickness,
                    visible=ref_text.visible,
                    layer=ref_text.layer,
                )

                test_bbox = test_text.get_bounding_box(
                    comp.x, comp.y, comp.rotation, comp.reference
                )

                # Score this position (lower is better)
                score = self._score_text_position(
                    comp, test_bbox, pad_clearance, offset_x, offset_y
                )

                if score < best_score:
                    best_score = score
                    best_position = (offset_x, offset_y)

            # Apply best position if found and better than current
            if best_position and best_score < float('inf'):
                ref_text.offset_x, ref_text.offset_y = best_position
                repositioned_count += 1

        return repositioned_count

    def _generate_text_candidates(self, comp: Component, ref_text: RefDesText,
                                  clearance: float) -> List[Tuple[float, float]]:
        """Generate candidate positions for ref des text.

        Returns positions around the component where text could be placed.
        """
        candidates = []
        comp_hw, comp_hh = comp.width / 2, comp.height / 2

        # Estimate text dimensions
        text_width = len(comp.reference) * ref_text.size * 0.6
        text_height = ref_text.size

        # Try positions: above, below, left, right of component
        # Above (centered horizontally)
        candidates.append((0, -(comp_hh + text_height / 2 + clearance)))
        # Below
        candidates.append((0, comp_hh + text_height / 2 + clearance))
        # Left
        candidates.append((-(comp_hw + text_width / 2 + clearance), 0))
        # Right
        candidates.append((comp_hw + text_width / 2 + clearance, 0))

        # Try corners
        for x_mult in [-1, 1]:
            for y_mult in [-1, 1]:
                candidates.append((
                    x_mult * (comp_hw + text_width / 2 + clearance) * 0.7,
                    y_mult * (comp_hh + text_height / 2 + clearance) * 0.7
                ))

        # Try edge-aligned positions (for small components, text to the side)
        # Left edge, above
        candidates.append((-(comp_hw + text_width / 2 + clearance), -comp_hh * 0.5))
        # Right edge, below
        candidates.append((comp_hw + text_width / 2 + clearance, comp_hh * 0.5))

        return candidates

    def _check_text_conflicts(self, comp: Component, text_bbox: Tuple[float, float, float, float],
                              pad_clearance: float) -> bool:
        """Check if text bounding box conflicts with pads."""
        # Check against own component pads
        for pad in comp.pads:
            pad_bbox = pad.get_bounding_box(comp.x, comp.y, comp.rotation)
            # Expand pad bbox by clearance
            pad_bbox = (
                pad_bbox[0] - pad_clearance,
                pad_bbox[1] - pad_clearance,
                pad_bbox[2] + pad_clearance,
                pad_bbox[3] + pad_clearance,
            )
            if self._bboxes_overlap(text_bbox, pad_bbox):
                return True

        # Check against nearby component pads
        for other in self.components.values():
            if other.reference == comp.reference or other.dnp:
                continue
            # Skip if components are far apart
            dist = comp.distance_to(other)
            if dist > 10.0:  # Only check nearby components
                continue
            for pad in other.pads:
                pad_bbox = pad.get_bounding_box(other.x, other.y, other.rotation)
                pad_bbox = (
                    pad_bbox[0] - pad_clearance,
                    pad_bbox[1] - pad_clearance,
                    pad_bbox[2] + pad_clearance,
                    pad_bbox[3] + pad_clearance,
                )
                if self._bboxes_overlap(text_bbox, pad_bbox):
                    return True

        return False

    def _score_text_position(self, comp: Component,
                             text_bbox: Tuple[float, float, float, float],
                             pad_clearance: float, offset_x: float, offset_y: float) -> float:
        """Score a text position (lower is better).

        Considers:
        - Pad overlap (very bad)
        - Distance from component center (prefer closer)
        - Out of board bounds (very bad)
        """
        score = 0.0

        # Heavy penalty for pad overlap
        overlap_penalty = 0.0
        for pad in comp.pads:
            pad_bbox = pad.get_bounding_box(comp.x, comp.y, comp.rotation)
            if self._bboxes_overlap(text_bbox, pad_bbox):
                overlap_penalty += 100.0

        # Check nearby components' pads
        for other in self.components.values():
            if other.reference == comp.reference or other.dnp:
                continue
            dist = comp.distance_to(other)
            if dist > 10.0:
                continue
            for pad in other.pads:
                pad_bbox = pad.get_bounding_box(other.x, other.y, other.rotation)
                if self._bboxes_overlap(text_bbox, pad_bbox):
                    overlap_penalty += 50.0

        score += overlap_penalty

        # Prefer positions closer to component
        distance = math.sqrt(offset_x ** 2 + offset_y ** 2)
        score += distance * 2.0

        # Penalty for being outside board bounds
        if self.outline.has_outline:
            text_center_x = (text_bbox[0] + text_bbox[2]) / 2
            text_center_y = (text_bbox[1] + text_bbox[3]) / 2
            if not self.outline.contains_point(text_center_x, text_center_y):
                score += 200.0

        return score

    def _bboxes_overlap(self, bb1: Tuple[float, float, float, float],
                        bb2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap."""
        return not (bb1[2] < bb2[0] or bb2[2] < bb1[0] or
                    bb1[3] < bb2[1] or bb2[3] < bb1[1])

    def to_kicad(self, path: Path):
        """Save board to KiCad file."""
        from .kicad_adapter import save_kicad_board
        save_kicad_board(self, path)

    def __repr__(self) -> str:
        return (f"Board(name={self.name!r}, "
                f"components={len(self.components)}, "
                f"nets={len(self.nets)}, "
                f"size={self.outline.width}x{self.outline.height}mm)")
