"""Obstacle map builder for routing.

Pre-computes all routing obstacles from a board before routing begins.
This follows @seveibar's lesson #4:
"Effective Spatial Partitioning + Caching is 1000x more important than algorithm performance"

The obstacle map is built once and reused for all net routing.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging
import hashlib

from .spatial_index import SpatialHashIndex, Obstacle, auto_calibrate_cell_size
from ..board.abstraction import Layer

logger = logging.getLogger(__name__)


def _deterministic_hash(s: str) -> int:
    """Create a deterministic integer hash from a string.

    Uses MD5 for fast, consistent hashing across Python processes and sessions.
    Python's built-in hash() is randomized by PYTHONHASHSEED for security,
    which makes routing results non-reproducible.
    """
    # Use MD5 (fast, 128-bit) and take first 8 bytes as int64
    digest = hashlib.md5(s.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='little', signed=True)


@dataclass
class NetPads:
    """Pads to connect for a single net."""
    net_name: str
    net_id: int
    pads: List[Obstacle]  # Pad obstacles with positions
    is_power: bool = False
    is_ground: bool = False


class ObstacleMapBuilder:
    """Build comprehensive obstacle map from board for routing.

    Extracts all routing obstacles:
    - Component bodies (blocks both layers if through-hole)
    - Component pads (with net association for same-net filtering)
    - Board edge keepout zones
    - Existing traces (if any)
    - Keepout areas

    The resulting spatial index enables O(~1) collision detection.
    """

    def __init__(self, board, dfm_profile):
        """
        Args:
            board: Board instance from atoplace.board.abstraction
            dfm_profile: DFMProfile with clearance rules
        """
        self.board = board
        self.dfm = dfm_profile

    def build(self) -> SpatialHashIndex:
        """Build complete obstacle index from board.

        Returns:
            SpatialHashIndex populated with all routing obstacles
        """
        # Collect obstacle sizes for cell calibration
        obstacle_sizes = []

        # First pass: collect sizes from pads only (not component bodies)
        for comp in self.board.components.values():
            for pad in comp.pads:
                obstacle_sizes.append((pad.width, pad.height))

        # Calibrate cell size
        cell_size = auto_calibrate_cell_size(obstacle_sizes, default=1.0)
        logger.debug(f"Calibrated spatial index cell size: {cell_size}mm")

        index = SpatialHashIndex(cell_size=cell_size)

        # Add obstacles - NOTE: Component bodies are NOT added as obstacles
        # because pads are inside component bounds, and adding bodies as
        # obstacles would block routes starting from their own pads.
        # Pads provide the actual routing constraints.
        pad_count = self._add_pad_obstacles(index)
        edge_count = self._add_edge_keepout(index)

        logger.info(
            f"Built obstacle map: "
            f"{pad_count} pads, {edge_count} edge segments"
        )

        stats = index.get_stats()
        logger.debug(
            f"Spatial index stats: {stats['total_cells']} cells, "
            f"avg {stats['avg_obstacles_per_cell']:.1f} obstacles/cell"
        )

        return index

    def _add_component_obstacles(self, index: SpatialHashIndex) -> int:
        """Add component bodies as obstacles.

        Note: Clearance is NOT embedded in obstacles. The router is responsible
        for passing the full clearance (trace_width/2 + dfm_clearance) to
        collision checks. This avoids double-counting clearance.
        """
        count = 0

        for ref, comp in self.board.components.items():
            # Skip DNP components
            if comp.dnp:
                continue

            bbox = comp.get_bounding_box()

            # Determine layers blocked - use Layer enum for comparison
            if comp.is_through_hole:
                layers = [-1]  # All layers
            else:
                # Compare against Layer enum values, not strings
                is_top = comp.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                       Layer.TOP_MASK, Layer.TOP_PASTE, Layer.TOP_COURTYARD)
                layer = 0 if is_top else 1
                layers = [layer]

            for layer in layers:
                index.add(Obstacle(
                    min_x=bbox[0],
                    min_y=bbox[1],
                    max_x=bbox[2],
                    max_y=bbox[3],
                    layer=layer,
                    clearance=0,  # Router handles clearance to avoid double-counting
                    net_id=None,  # Blocks all nets
                    obstacle_type="component",
                    ref=ref
                ))
                count += 1

        return count

    def _add_pad_obstacles(self, index: SpatialHashIndex) -> int:
        """Add pads as obstacles (with net association for filtering).

        Note: Clearance is NOT embedded in obstacles. The router is responsible
        for passing the full clearance (trace_width/2 + dfm_clearance) to
        collision checks. This avoids double-counting clearance.
        """
        count = 0

        for ref, comp in self.board.components.items():
            if comp.dnp:
                continue

            for pad in comp.pads:
                # Get pad bounding box in board coordinates
                # This properly handles both pad rotation and component rotation
                bbox = pad.get_bounding_box(comp.x, comp.y, comp.rotation)

                # Get net ID (hash of net name for fast comparison)
                net_id = None
                if pad.net:
                    net_id = _deterministic_hash(pad.net)

                # Determine layers - check for through-hole based on drill attribute
                # (Pad class uses drill, not is_through_hole)
                is_through_hole = pad.drill is not None and pad.drill > 0
                if is_through_hole:
                    layers = [-1]  # All layers
                elif hasattr(pad, 'layer') and pad.layer is not None:
                    # Compare against Layer enum values, not strings
                    is_top = pad.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                          Layer.TOP_MASK, Layer.TOP_PASTE, Layer.TOP_COURTYARD)
                    layers = [0 if is_top else 1]
                else:
                    # Default based on component - use Layer enum comparison
                    is_top = comp.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                           Layer.TOP_MASK, Layer.TOP_PASTE, Layer.TOP_COURTYARD)
                    layer = 0 if is_top else 1
                    layers = [layer]

                for layer in layers:
                    index.add(Obstacle(
                        min_x=bbox[0],
                        min_y=bbox[1],
                        max_x=bbox[2],
                        max_y=bbox[3],
                        layer=layer,
                        clearance=0,  # Router handles clearance to avoid double-counting
                        net_id=net_id,
                        obstacle_type="pad",
                        ref=f"{ref}.{pad.name}" if hasattr(pad, 'name') else ref
                    ))
                    count += 1

        return count

    def _transform_pad_position(self, comp, pad) -> Tuple[float, float]:
        """Transform pad position from component-relative to board coordinates."""
        import math

        # Pad position relative to component center
        px, py = pad.x, pad.y

        # Apply component rotation
        if comp.rotation != 0:
            rad = math.radians(comp.rotation)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            px, py = (
                px * cos_r - py * sin_r,
                px * sin_r + py * cos_r
            )

        # Translate to board coordinates
        return (comp.x + px, comp.y + py)

    def _add_edge_keepout(self, index: SpatialHashIndex) -> int:
        """Add board edge keepout zones.

        Only generates keepouts when an explicit board outline is defined
        (has_outline=True). This prevents artificial obstacles on boards
        without explicit Edge.Cuts boundaries.

        Handles both clockwise and counter-clockwise winding by testing
        which direction is "inside" the board using contains_point.
        """
        if not self.board.outline:
            return 0

        # Skip if no explicit outline is defined
        # This prevents artificial obstacles on outline-less boards
        if not self.board.outline.has_outline:
            return 0

        # Get outline points - BoardOutline uses 'polygon' attribute
        points = None
        if hasattr(self.board.outline, 'polygon') and self.board.outline.polygon:
            points = self.board.outline.polygon
        elif hasattr(self.board.outline, 'points') and self.board.outline.points:
            points = self.board.outline.points

        if not points:
            # Fall back to bounding box as rectangle
            if hasattr(self.board.outline, 'get_bounding_box'):
                bbox = self.board.outline.get_bounding_box()
                if bbox:
                    min_x, min_y, max_x, max_y = bbox
                    points = [
                        (min_x, min_y), (max_x, min_y),
                        (max_x, max_y), (min_x, max_y)
                    ]

        if not points:
            return 0

        count = 0
        edge_clearance = self.dfm.min_trace_to_edge

        # Create keepout rectangles along each edge
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            # Create thin rectangle along edge
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = (dx*dx + dy*dy) ** 0.5

            if length < 0.001:
                continue

            # Normalize direction
            dx /= length
            dy /= length

            # Calculate both possible normals and test which points inward
            # Normal options: (dy, -dx) or (-dy, dx)
            import math

            # Test point slightly offset from edge midpoint in each direction
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            test_dist = 0.1  # Small offset for testing

            # Try normal (dy, -dx)
            test_x1 = mid_x + dy * test_dist
            test_y1 = mid_y - dx * test_dist

            # Try normal (-dy, dx)
            test_x2 = mid_x - dy * test_dist
            test_y2 = mid_y + dx * test_dist

            # Determine which side is "inside" the board
            inside1 = self.board.outline.contains_point(test_x1, test_y1)
            inside2 = self.board.outline.contains_point(test_x2, test_y2)

            # Choose the normal pointing INWARD (inside the board)
            if inside1 and not inside2:
                nx, ny = dy, -dx  # Normal 1 points inward
            elif inside2 and not inside1:
                nx, ny = -dy, dx  # Normal 2 points inward
            else:
                # Both or neither - fall back to CCW assumption
                nx, ny = dy, -dx

            # Four corners of keepout rectangle
            # Outer edge (outside board, at board boundary)
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            # Inner edge (inside board, at keepout boundary)
            x3 = p2[0] + nx * edge_clearance
            y3 = p2[1] + ny * edge_clearance
            x4 = p1[0] + nx * edge_clearance
            y4 = p1[1] + ny * edge_clearance

            # Bounding box of the keepout zone
            min_x = min(x1, x2, x3, x4)
            max_x = max(x1, x2, x3, x4)
            min_y = min(y1, y2, y3, y4)
            max_y = max(y1, y2, y3, y4)

            # Add as obstacle on all layers
            index.add(Obstacle(
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                layer=-1,  # All layers
                clearance=0,  # Already includes clearance
                net_id=None,
                obstacle_type="keepout",
                ref=f"edge_{i}"
            ))
            count += 1

        return count

    def get_net_pads(self) -> List[NetPads]:
        """Extract all nets with their pad positions for routing.

        Returns:
            List of NetPads, one per net that needs routing
        """
        # Group pads by net
        net_pads: Dict[str, List[Obstacle]] = {}

        for ref, comp in self.board.components.items():
            if comp.dnp:
                continue

            for pad in comp.pads:
                if not pad.net:
                    continue

                # Get pad bounding box in board coordinates
                # This properly handles both pad rotation and component rotation
                bbox = pad.get_bounding_box(comp.x, comp.y, comp.rotation)

                # Determine layer - check for through-hole based on drill attribute
                is_through_hole = pad.drill is not None and pad.drill > 0
                if is_through_hole:
                    layer = -1  # All layers
                elif hasattr(pad, 'layer') and pad.layer is not None:
                    # Compare against Layer enum values, not strings
                    is_top = pad.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                          Layer.TOP_MASK, Layer.TOP_PASTE, Layer.TOP_COURTYARD)
                    layer = 0 if is_top else 1
                else:
                    # Default based on component - use Layer enum comparison
                    is_top = comp.layer in (Layer.TOP_COPPER, Layer.TOP_SILK,
                                           Layer.TOP_MASK, Layer.TOP_PASTE, Layer.TOP_COURTYARD)
                    layer = 0 if is_top else 1

                pad_obs = Obstacle(
                    min_x=bbox[0],
                    min_y=bbox[1],
                    max_x=bbox[2],
                    max_y=bbox[3],
                    layer=layer,
                    clearance=0,
                    net_id=_deterministic_hash(pad.net),
                    obstacle_type="pad",
                    ref=f"{ref}.{pad.name}" if hasattr(pad, 'name') else ref
                )

                if pad.net not in net_pads:
                    net_pads[pad.net] = []
                net_pads[pad.net].append(pad_obs)

        # Convert to NetPads list
        result = []
        for net_name, pads in net_pads.items():
            if len(pads) < 2:
                continue  # Skip single-pad nets

            # Check if power/ground
            net_obj = self.board.nets.get(net_name)
            is_power = net_obj.is_power if net_obj else False
            is_ground = net_obj.is_ground if net_obj else False

            result.append(NetPads(
                net_name=net_name,
                net_id=_deterministic_hash(net_name),
                pads=pads,
                is_power=is_power,
                is_ground=is_ground
            ))

        return result

    def get_routing_stats(self) -> Dict:
        """Get statistics about the board for routing planning.

        For polygon outlines, uses actual polygon area (Shoelace formula)
        instead of bounding box area, giving accurate density for irregular boards.
        """
        nets = self.get_net_pads()
        total_pads = sum(len(n.pads) for n in nets)

        # Estimate routing complexity
        # More pads in small area = harder
        area = 1000  # Default
        if self.board.outline:
            # Prefer actual polygon area over bounding box area
            if hasattr(self.board.outline, 'polygon') and self.board.outline.polygon:
                # Calculate polygon area using Shoelace formula
                polygon = self.board.outline.polygon
                n = len(polygon)
                if n >= 3:
                    area = 0.0
                    for i in range(n):
                        j = (i + 1) % n
                        area += polygon[i][0] * polygon[j][1]
                        area -= polygon[j][0] * polygon[i][1]
                    area = abs(area) / 2.0
            elif hasattr(self.board.outline, 'get_bounding_box'):
                # Fall back to bounding box only if no polygon
                bbox = self.board.outline.get_bounding_box()
                if bbox:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if area <= 0:
            area = self.board.width * self.board.height if hasattr(self.board, 'width') else 1000

        density = total_pads / max(area, 1)

        return {
            "net_count": len(nets),
            "total_pads": total_pads,
            "power_nets": sum(1 for n in nets if n.is_power),
            "ground_nets": sum(1 for n in nets if n.is_ground),
            "board_area_mm2": area,
            "pad_density": density,
            "estimated_difficulty": "high" if density > 0.1 else "medium" if density > 0.05 else "low"
        }


def build_obstacle_map(board, dfm_profile) -> Tuple[SpatialHashIndex, List[NetPads]]:
    """Convenience function to build obstacle map and get nets.

    Args:
        board: Board instance
        dfm_profile: DFM profile for clearances

    Returns:
        Tuple of (obstacle_index, net_pads_list)
    """
    builder = ObstacleMapBuilder(board, dfm_profile)
    index = builder.build()
    nets = builder.get_net_pads()
    return index, nets
