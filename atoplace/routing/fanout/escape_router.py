"""Escape routing for BGA fanout.

Routes traces from fanout vias outward to the edge of the BGA
courtyard where they can connect to the main routing field.

The escape router uses a vector field approach:
1. Create a vector field pointing outward from BGA center
2. Project rays from vias along the field
3. Detect collisions with other vias/pads
4. Generate trace segments for valid paths
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .patterns import FanoutTrace, FanoutVia

logger = logging.getLogger(__name__)


class EscapeDirection(Enum):
    """Primary escape direction from BGA."""
    NORTH = "north"      # Escape upward (negative Y)
    SOUTH = "south"      # Escape downward (positive Y)
    EAST = "east"        # Escape rightward (positive X)
    WEST = "west"        # Escape leftward (negative X)
    NE = "northeast"     # Diagonal escape
    NW = "northwest"
    SE = "southeast"
    SW = "southwest"
    RADIAL = "radial"    # Direct outward from center


@dataclass
class EscapeResult:
    """Result of escape routing for a single via.

    Attributes:
        success: Whether escape routing succeeded
        traces: List of trace segments forming the escape path
        end_point: Final endpoint of escape path (where routing field begins)
        escape_length: Total escape path length
        layer: Layer the escape is routed on
        direction: Primary escape direction used
        failure_reason: If failed, reason for failure
    """
    success: bool
    traces: List[FanoutTrace] = field(default_factory=list)
    end_point: Optional[Tuple[float, float]] = None
    escape_length: float = 0.0
    layer: int = 0
    direction: Optional[EscapeDirection] = None
    failure_reason: str = ""


class ObstacleGrid:
    """Simple grid-based obstacle tracking for escape routing.

    Uses a coarse grid to quickly identify occupied areas during
    escape ray casting.
    """

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        cell_size: float = 0.1,
    ):
        """
        Args:
            min_x, min_y, max_x, max_y: Bounding box of routing area
            cell_size: Grid cell size in mm
        """
        self.min_x = min_x
        self.min_y = min_y
        self.cell_size = cell_size
        self.cols = max(1, int(math.ceil((max_x - min_x) / cell_size)))
        self.rows = max(1, int(math.ceil((max_y - min_y) / cell_size)))

        # Dict of (col, row) -> set of (net_name, layer)
        self.occupied: Dict[Tuple[int, int], Set[Tuple[Optional[str], int]]] = {}

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to grid cell."""
        col = int((x - self.min_x) / self.cell_size)
        row = int((y - self.min_y) / self.cell_size)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return col, row

    def add_obstacle(
        self,
        x: float,
        y: float,
        radius: float,
        layer: int,
        net_name: Optional[str] = None,
    ):
        """Mark an area as occupied.

        Args:
            x, y: Center of obstacle
            radius: Radius of obstacle (including clearance)
            layer: Layer the obstacle is on (-1 for all layers)
            net_name: Net name (for same-net checking)
        """
        # Get cells that the obstacle touches
        cells_radius = int(math.ceil(radius / self.cell_size)) + 1
        center_col, center_row = self._get_cell(x, y)

        for dc in range(-cells_radius, cells_radius + 1):
            for dr in range(-cells_radius, cells_radius + 1):
                col = center_col + dc
                row = center_row + dr
                if 0 <= col < self.cols and 0 <= row < self.rows:
                    key = (col, row)
                    if key not in self.occupied:
                        self.occupied[key] = set()
                    self.occupied[key].add((net_name, layer))

    def is_clear(
        self,
        x: float,
        y: float,
        layer: int,
        net_name: Optional[str] = None,
    ) -> bool:
        """Check if a point is clear of obstacles.

        Same-net obstacles don't block.

        Args:
            x, y: Point to check
            layer: Layer to check
            net_name: Net name (same-net doesn't block)

        Returns:
            True if point is clear
        """
        col, row = self._get_cell(x, y)
        key = (col, row)
        if key not in self.occupied:
            return True

        for obs_net, obs_layer in self.occupied[key]:
            # Same net doesn't block
            if obs_net is not None and obs_net == net_name:
                continue
            # Different layer doesn't block (unless layer is -1)
            if obs_layer != -1 and obs_layer != layer:
                continue
            return False

        return True


class EscapeRouter:
    """Routes escape traces from fanout vias to the BGA edge.

    Uses ray casting to find clear paths from each via outward
    to where main routing can begin.
    """

    def __init__(
        self,
        trace_width: float = 0.2,
        clearance: float = 0.15,
        escape_margin: float = 1.0,
    ):
        """
        Args:
            trace_width: Width of escape traces (mm)
            clearance: Required clearance from obstacles (mm)
            escape_margin: Extra distance past BGA courtyard (mm)
        """
        self.trace_width = trace_width
        self.clearance = clearance
        self.escape_margin = escape_margin

    def route_escapes(
        self,
        vias: List[FanoutVia],
        component_center: Tuple[float, float],
        courtyard_bounds: Tuple[float, float, float, float],
        existing_obstacles: Optional[List[Tuple[float, float, float, int]]] = None,
    ) -> Dict[str, EscapeResult]:
        """Route escape traces for all fanout vias.

        Args:
            vias: List of FanoutVia objects to route escapes for
            component_center: (x, y) center of BGA component
            courtyard_bounds: (min_x, min_y, max_x, max_y) of BGA courtyard
            existing_obstacles: List of (x, y, radius, layer) obstacles

        Returns:
            Dict mapping pad_number to EscapeResult
        """
        results = {}
        cx, cy = component_center
        min_x, min_y, max_x, max_y = courtyard_bounds

        # Build obstacle grid
        grid = ObstacleGrid(
            min_x - self.escape_margin * 2,
            min_y - self.escape_margin * 2,
            max_x + self.escape_margin * 2,
            max_y + self.escape_margin * 2,
            cell_size=self.trace_width * 2,
        )

        # Add existing obstacles
        if existing_obstacles:
            for ox, oy, radius, layer in existing_obstacles:
                grid.add_obstacle(ox, oy, radius + self.clearance, layer)

        # Add vias as obstacles
        for via in vias:
            grid.add_obstacle(
                via.x, via.y,
                via.pad_diameter / 2 + self.clearance,
                layer=-1,  # Vias block all layers
                net_name=via.net_name,
            )

        # Route each via
        for via in vias:
            result = self._route_single_escape(
                via, cx, cy, courtyard_bounds, grid
            )
            if via.pad_number:
                results[via.pad_number] = result

            # Add routed traces as obstacles for subsequent routing
            if result.success:
                for trace in result.traces:
                    # Add trace as line obstacle
                    self._add_trace_obstacle(trace, grid)

        return results

    def _route_single_escape(
        self,
        via: FanoutVia,
        cx: float,
        cy: float,
        courtyard_bounds: Tuple[float, float, float, float],
        grid: ObstacleGrid,
    ) -> EscapeResult:
        """Route escape for a single via.

        Uses ray casting to find a clear path outward.
        """
        min_x, min_y, max_x, max_y = courtyard_bounds

        # Determine primary escape direction (radial from center)
        direction = self._get_escape_direction(via.x, via.y, cx, cy)

        # Calculate target point (outside courtyard with margin)
        target = self._get_escape_target(
            via.x, via.y, direction, courtyard_bounds
        )

        if target is None:
            return EscapeResult(
                success=False,
                failure_reason="Could not determine escape target"
            )

        target_x, target_y = target

        # Try direct path first
        if self._path_is_clear(via.x, via.y, target_x, target_y, via.end_layer, via.net_name, grid):
            trace = FanoutTrace(
                start=(via.x, via.y),
                end=(target_x, target_y),
                width=self.trace_width,
                layer=via.end_layer,
                net_name=via.net_name,
            )
            length = math.sqrt((target_x - via.x)**2 + (target_y - via.y)**2)
            return EscapeResult(
                success=True,
                traces=[trace],
                end_point=(target_x, target_y),
                escape_length=length,
                layer=via.end_layer,
                direction=direction,
            )

        # Try alternate directions
        alternate_dirs = self._get_alternate_directions(direction)
        for alt_dir in alternate_dirs:
            alt_target = self._get_escape_target(
                via.x, via.y, alt_dir, courtyard_bounds
            )
            if alt_target is None:
                continue

            target_x, target_y = alt_target
            if self._path_is_clear(via.x, via.y, target_x, target_y, via.end_layer, via.net_name, grid):
                trace = FanoutTrace(
                    start=(via.x, via.y),
                    end=(target_x, target_y),
                    width=self.trace_width,
                    layer=via.end_layer,
                    net_name=via.net_name,
                )
                length = math.sqrt((target_x - via.x)**2 + (target_y - via.y)**2)
                return EscapeResult(
                    success=True,
                    traces=[trace],
                    end_point=(target_x, target_y),
                    escape_length=length,
                    layer=via.end_layer,
                    direction=alt_dir,
                )

        # Try L-shaped path as last resort
        l_result = self._try_l_shaped_escape(
            via, cx, cy, courtyard_bounds, grid
        )
        if l_result.success:
            return l_result

        return EscapeResult(
            success=False,
            layer=via.end_layer,
            failure_reason="No clear escape path found"
        )

    def _get_escape_direction(
        self, x: float, y: float, cx: float, cy: float
    ) -> EscapeDirection:
        """Determine primary escape direction based on position."""
        dx = x - cx
        dy = y - cy

        # Calculate angle
        angle = math.atan2(dy, dx)  # Range: -pi to pi

        # Map angle to direction
        # Using 8 cardinal + diagonal directions
        if -math.pi/8 <= angle < math.pi/8:
            return EscapeDirection.EAST
        elif math.pi/8 <= angle < 3*math.pi/8:
            return EscapeDirection.SE
        elif 3*math.pi/8 <= angle < 5*math.pi/8:
            return EscapeDirection.SOUTH
        elif 5*math.pi/8 <= angle < 7*math.pi/8:
            return EscapeDirection.SW
        elif angle >= 7*math.pi/8 or angle < -7*math.pi/8:
            return EscapeDirection.WEST
        elif -7*math.pi/8 <= angle < -5*math.pi/8:
            return EscapeDirection.NW
        elif -5*math.pi/8 <= angle < -3*math.pi/8:
            return EscapeDirection.NORTH
        else:  # -3*math.pi/8 <= angle < -math.pi/8
            return EscapeDirection.NE

    def _get_alternate_directions(
        self, primary: EscapeDirection
    ) -> List[EscapeDirection]:
        """Get alternate escape directions to try."""
        # Adjacent directions
        dir_order = [
            EscapeDirection.NORTH, EscapeDirection.NE,
            EscapeDirection.EAST, EscapeDirection.SE,
            EscapeDirection.SOUTH, EscapeDirection.SW,
            EscapeDirection.WEST, EscapeDirection.NW,
        ]
        try:
            idx = dir_order.index(primary)
        except ValueError:
            return dir_order

        # Return adjacent directions first, then others
        alternates = []
        for offset in [1, -1, 2, -2, 3, -3, 4]:
            alt_idx = (idx + offset) % len(dir_order)
            alternates.append(dir_order[alt_idx])
        return alternates

    def _get_escape_target(
        self,
        x: float,
        y: float,
        direction: EscapeDirection,
        courtyard_bounds: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float]]:
        """Calculate escape target point based on direction."""
        min_x, min_y, max_x, max_y = courtyard_bounds
        margin = self.escape_margin

        direction_vectors = {
            EscapeDirection.NORTH: (0, -1),
            EscapeDirection.SOUTH: (0, 1),
            EscapeDirection.EAST: (1, 0),
            EscapeDirection.WEST: (-1, 0),
            EscapeDirection.NE: (1, -1),
            EscapeDirection.NW: (-1, -1),
            EscapeDirection.SE: (1, 1),
            EscapeDirection.SW: (-1, 1),
        }

        if direction == EscapeDirection.RADIAL:
            # Use center-relative radial direction
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            dx = x - cx
            dy = y - cy
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.001:
                return None
            dx /= length
            dy /= length
        else:
            vec = direction_vectors.get(direction)
            if vec is None:
                return None
            dx, dy = vec
            # Normalize diagonal vectors
            length = math.sqrt(dx*dx + dy*dy)
            dx /= length
            dy /= length

        # Project to courtyard edge plus margin
        # Find intersection with courtyard boundary
        if abs(dx) > 0.001:
            if dx > 0:
                t_x = (max_x + margin - x) / dx
            else:
                t_x = (min_x - margin - x) / dx
        else:
            t_x = float('inf')

        if abs(dy) > 0.001:
            if dy > 0:
                t_y = (max_y + margin - y) / dy
            else:
                t_y = (min_y - margin - y) / dy
        else:
            t_y = float('inf')

        t = min(t_x, t_y)
        if t <= 0 or t == float('inf'):
            return None

        target_x = x + dx * t
        target_y = y + dy * t

        return (target_x, target_y)

    def _path_is_clear(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        layer: int,
        net_name: Optional[str],
        grid: ObstacleGrid,
    ) -> bool:
        """Check if a straight path is clear of obstacles."""
        # Sample points along the path
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.001:
            return True

        num_samples = max(2, int(length / (self.trace_width / 2)))
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not grid.is_clear(x, y, layer, net_name):
                return False

        return True

    def _try_l_shaped_escape(
        self,
        via: FanoutVia,
        cx: float,
        cy: float,
        courtyard_bounds: Tuple[float, float, float, float],
        grid: ObstacleGrid,
    ) -> EscapeResult:
        """Try an L-shaped escape path."""
        min_x, min_y, max_x, max_y = courtyard_bounds
        margin = self.escape_margin

        # Determine quadrant
        dx = via.x - cx
        dy = via.y - cy

        # Try horizontal then vertical
        if abs(dx) > abs(dy):
            # Go horizontal first
            if dx > 0:
                mid_x = max_x + margin
            else:
                mid_x = min_x - margin
            mid_y = via.y

            # Then vertical to final target
            if dy > 0:
                final_y = max_y + margin
            else:
                final_y = min_y - margin
            final_x = mid_x
        else:
            # Go vertical first
            if dy > 0:
                mid_y = max_y + margin
            else:
                mid_y = min_y - margin
            mid_x = via.x

            # Then horizontal to final target
            if dx > 0:
                final_x = max_x + margin
            else:
                final_x = min_x - margin
            final_y = mid_y

        # Check both segments
        if (self._path_is_clear(via.x, via.y, mid_x, mid_y, via.end_layer, via.net_name, grid) and
            self._path_is_clear(mid_x, mid_y, final_x, final_y, via.end_layer, via.net_name, grid)):

            traces = [
                FanoutTrace(
                    start=(via.x, via.y),
                    end=(mid_x, mid_y),
                    width=self.trace_width,
                    layer=via.end_layer,
                    net_name=via.net_name,
                ),
                FanoutTrace(
                    start=(mid_x, mid_y),
                    end=(final_x, final_y),
                    width=self.trace_width,
                    layer=via.end_layer,
                    net_name=via.net_name,
                ),
            ]
            length = (
                math.sqrt((mid_x - via.x)**2 + (mid_y - via.y)**2) +
                math.sqrt((final_x - mid_x)**2 + (final_y - mid_y)**2)
            )
            return EscapeResult(
                success=True,
                traces=traces,
                end_point=(final_x, final_y),
                escape_length=length,
                layer=via.end_layer,
            )

        return EscapeResult(success=False, failure_reason="L-shaped path blocked")

    def _add_trace_obstacle(self, trace: FanoutTrace, grid: ObstacleGrid):
        """Add a routed trace as an obstacle."""
        x1, y1 = trace.start
        x2, y2 = trace.end
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.001:
            return

        # Sample points along trace
        num_samples = max(2, int(length / (self.trace_width / 2)))
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            grid.add_obstacle(
                x, y,
                self.trace_width / 2 + self.clearance,
                trace.layer,
                trace.net_name,
            )
