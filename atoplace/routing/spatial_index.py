"""Spatial hash index for O(~1) collision detection.

Based on @seveibar's autorouter lesson #3:
"Spatial Hash Indexing > Tree Data Structures"

Key insight: Any time you're using a tree (QuadTree, R-Tree) you're ignoring
an O(~1) hash algorithm for a more complicated O(log(N)) algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import math


@dataclass
class Obstacle:
    """A routing obstacle (trace segment, pad, via, keepout, component body).

    Obstacles are axis-aligned bounding boxes (AABBs) that block routing.
    """
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    layer: int  # 0=F.Cu, 1=B.Cu, -1=all layers (through-hole)
    clearance: float = 0.0  # Additional clearance beyond bounds
    net_id: Optional[int] = None  # None = blocks all nets, else blocks other nets
    obstacle_type: str = "generic"  # "pad", "via", "trace", "component", "keepout"
    ref: Optional[str] = None  # Component reference if applicable

    def __hash__(self):
        return hash((self.min_x, self.min_y, self.max_x, self.max_y, self.layer))

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of obstacle."""
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def expanded(self, margin: float) -> "Obstacle":
        """Return new obstacle expanded by margin on all sides."""
        return Obstacle(
            min_x=self.min_x - margin,
            min_y=self.min_y - margin,
            max_x=self.max_x + margin,
            max_y=self.max_y + margin,
            layer=self.layer,
            clearance=self.clearance,
            net_id=self.net_id,
            obstacle_type=self.obstacle_type,
            ref=self.ref
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside obstacle (including clearance)."""
        return (self.min_x - self.clearance <= x <= self.max_x + self.clearance and
                self.min_y - self.clearance <= y <= self.max_y + self.clearance)

    def intersects(self, other: "Obstacle") -> bool:
        """Check if this obstacle intersects another."""
        return not (self.max_x < other.min_x or other.max_x < self.min_x or
                    self.max_y < other.min_y or other.max_y < self.min_y)


@dataclass
class SpatialHashIndex:
    """Grid-based spatial hash for O(~1) collision queries.

    Instead of tree structures (QuadTree, R-Tree) that give O(log N),
    we hash object locations into grid cells for O(~1) lookup.

    Cell size selection is critical:
    - Too small: many cells, high memory, many edge cases
    - Too large: many obstacles per cell, slow queries
    - Rule of thumb: 2-3x the average obstacle size
    """
    cell_size: float = 1.0  # Size of each hash cell in mm
    cells: Dict[Tuple[int, int], List[Obstacle]] = field(default_factory=dict)
    all_obstacles: List[Obstacle] = field(default_factory=list)
    _obstacle_count: int = 0

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Hash position to cell coordinates."""
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def _get_cells_for_rect(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> Set[Tuple[int, int]]:
        """Get all cells that a rectangle overlaps."""
        cells = set()
        start_x = int(math.floor(min_x / self.cell_size))
        end_x = int(math.floor(max_x / self.cell_size))
        start_y = int(math.floor(min_y / self.cell_size))
        end_y = int(math.floor(max_y / self.cell_size))

        for cx in range(start_x, end_x + 1):
            for cy in range(start_y, end_y + 1):
                cells.add((cx, cy))
        return cells

    def add(self, obstacle: Obstacle):
        """Add obstacle to index."""
        # Include clearance in cell placement
        cells = self._get_cells_for_rect(
            obstacle.min_x - obstacle.clearance,
            obstacle.min_y - obstacle.clearance,
            obstacle.max_x + obstacle.clearance,
            obstacle.max_y + obstacle.clearance
        )
        for cell in cells:
            if cell not in self.cells:
                self.cells[cell] = []
            self.cells[cell].append(obstacle)

        self.all_obstacles.append(obstacle)
        self._obstacle_count += 1

    def remove(self, obstacle: Obstacle):
        """Remove obstacle from index."""
        cells = self._get_cells_for_rect(
            obstacle.min_x - obstacle.clearance,
            obstacle.min_y - obstacle.clearance,
            obstacle.max_x + obstacle.clearance,
            obstacle.max_y + obstacle.clearance
        )
        for cell in cells:
            if cell in self.cells:
                try:
                    self.cells[cell].remove(obstacle)
                except ValueError:
                    pass

        try:
            self.all_obstacles.remove(obstacle)
            self._obstacle_count -= 1
        except ValueError:
            pass

    def query_point(
        self,
        x: float,
        y: float,
        layer: int,
        net_id: Optional[int] = None
    ) -> List[Obstacle]:
        """
        Query obstacles near a point.

        Args:
            x, y: Query position
            layer: Layer to check (0=F.Cu, 1=B.Cu, -1=all)
            net_id: If provided, exclude obstacles on same net (they don't block)

        Returns:
            List of potentially colliding obstacles (candidates for detailed check)
        """
        cell = self._get_cell(x, y)
        candidates = []

        # Check cell and 8 neighbors for robustness
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in self.cells:
                    for obs in self.cells[neighbor]:
                        # Layer check: -1 means all layers
                        if obs.layer != -1 and layer != -1 and obs.layer != layer:
                            continue
                        # Net check: same net doesn't block
                        if net_id is not None and obs.net_id == net_id:
                            continue
                        candidates.append(obs)

        return candidates

    def query_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        layer: int,
        net_id: Optional[int] = None
    ) -> List[Obstacle]:
        """Query obstacles that might intersect a rectangle."""
        cells = self._get_cells_for_rect(min_x, min_y, max_x, max_y)
        candidates = []
        seen = set()

        for cell in cells:
            if cell in self.cells:
                for obs in self.cells[cell]:
                    if id(obs) in seen:
                        continue
                    seen.add(id(obs))

                    # Layer check
                    if obs.layer != -1 and layer != -1 and obs.layer != layer:
                        continue
                    # Net check
                    if net_id is not None and obs.net_id == net_id:
                        continue
                    candidates.append(obs)

        return candidates

    def check_collision(
        self,
        x: float,
        y: float,
        layer: int,
        clearance: float = 0.0,
        net_id: Optional[int] = None
    ) -> bool:
        """
        Check if a point collides with any obstacle.

        Args:
            x, y: Point to check
            layer: Layer to check
            clearance: Additional clearance around point
            net_id: Net ID (same-net obstacles don't collide)

        Returns:
            True if collision detected
        """
        for obs in self.query_point(x, y, layer, net_id):
            # AABB check with clearance
            total_clearance = obs.clearance + clearance
            if (x >= obs.min_x - total_clearance and
                x <= obs.max_x + total_clearance and
                y >= obs.min_y - total_clearance and
                y <= obs.max_y + total_clearance):
                return True
        return False

    def check_segment_collision(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        layer: int,
        width: float = 0.0,
        net_id: Optional[int] = None
    ) -> bool:
        """
        Check if a line segment collides with any obstacle.

        Uses fast AABB pre-filter, then precise segment-rect intersection.

        Args:
            x1, y1, x2, y2: Segment endpoints
            layer: Layer to check
            width: Trace width (adds to clearance)
            net_id: Net ID for same-net filtering

        Returns:
            True if collision detected
        """
        # Segment bounding box
        min_x = min(x1, x2) - width / 2
        max_x = max(x1, x2) + width / 2
        min_y = min(y1, y2) - width / 2
        max_y = max(y1, y2) + width / 2

        for obs in self.query_rect(min_x, min_y, max_x, max_y, layer, net_id):
            if self._segment_intersects_rect(
                x1, y1, x2, y2, width / 2 + obs.clearance,
                obs.min_x, obs.min_y, obs.max_x, obs.max_y
            ):
                return True
        return False

    def _segment_intersects_rect(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        half_width: float,
        rect_min_x: float, rect_min_y: float,
        rect_max_x: float, rect_max_y: float
    ) -> bool:
        """Check if line segment with width intersects rectangle.

        Uses the separating axis theorem for efficiency.
        """
        # Expand rect by half_width
        rect_min_x -= half_width
        rect_min_y -= half_width
        rect_max_x += half_width
        rect_max_y += half_width

        # Quick rejection: segment bbox vs rect
        seg_min_x, seg_max_x = (x1, x2) if x1 < x2 else (x2, x1)
        seg_min_y, seg_max_y = (y1, y2) if y1 < y2 else (y2, y1)

        if seg_max_x < rect_min_x or seg_min_x > rect_max_x:
            return False
        if seg_max_y < rect_min_y or seg_min_y > rect_max_y:
            return False

        # Check if either endpoint is inside rect
        if rect_min_x <= x1 <= rect_max_x and rect_min_y <= y1 <= rect_max_y:
            return True
        if rect_min_x <= x2 <= rect_max_x and rect_min_y <= y2 <= rect_max_y:
            return True

        # Check segment intersection with rect edges
        # This is simplified - for a production router, use proper
        # Liang-Barsky or Cohen-Sutherland clipping

        # Check if segment crosses any rect edge
        rect_edges = [
            (rect_min_x, rect_min_y, rect_max_x, rect_min_y),  # bottom
            (rect_max_x, rect_min_y, rect_max_x, rect_max_y),  # right
            (rect_max_x, rect_max_y, rect_min_x, rect_max_y),  # top
            (rect_min_x, rect_max_y, rect_min_x, rect_min_y),  # left
        ]

        for ex1, ey1, ex2, ey2 in rect_edges:
            if self._segments_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
                return True

        return False

    def _segments_intersect(
        self,
        ax1: float, ay1: float, ax2: float, ay2: float,
        bx1: float, by1: float, bx2: float, by2: float
    ) -> bool:
        """Check if two line segments intersect using cross products."""
        def cross(ox: float, oy: float, ax: float, ay: float, bx: float, by: float) -> float:
            return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

        d1 = cross(bx1, by1, bx2, by2, ax1, ay1)
        d2 = cross(bx1, by1, bx2, by2, ax2, ay2)
        d3 = cross(ax1, ay1, ax2, ay2, bx1, by1)
        d4 = cross(ax1, ay1, ax2, ay2, bx2, by2)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Collinear cases (point on segment)
        def on_segment(px, py, qx, qy, rx, ry):
            return (min(px, rx) <= qx <= max(px, rx) and
                    min(py, ry) <= qy <= max(py, ry))

        if d1 == 0 and on_segment(bx1, by1, ax1, ay1, bx2, by2):
            return True
        if d2 == 0 and on_segment(bx1, by1, ax2, ay2, bx2, by2):
            return True
        if d3 == 0 and on_segment(ax1, ay1, bx1, by1, ax2, ay2):
            return True
        if d4 == 0 and on_segment(ax1, ay1, bx2, by2, ax2, ay2):
            return True

        return False

    def get_stats(self) -> Dict:
        """Get index statistics for debugging."""
        cell_counts = [len(obs) for obs in self.cells.values()]
        return {
            "total_obstacles": self._obstacle_count,
            "total_cells": len(self.cells),
            "cell_size": self.cell_size,
            "avg_obstacles_per_cell": sum(cell_counts) / max(len(cell_counts), 1),
            "max_obstacles_per_cell": max(cell_counts) if cell_counts else 0,
            "min_obstacles_per_cell": min(cell_counts) if cell_counts else 0,
        }


def auto_calibrate_cell_size(
    obstacle_sizes: List[Tuple[float, float]],
    default: float = 1.0
) -> float:
    """
    Determine optimal cell size based on obstacle sizes.

    Rule of thumb: cell_size = 2-3x median obstacle size.
    This ensures most obstacles fit in 1-4 cells while keeping
    cell occupancy reasonable.

    Args:
        obstacle_sizes: List of (width, height) tuples
        default: Default cell size if no obstacles

    Returns:
        Optimal cell size in mm
    """
    if not obstacle_sizes:
        return default

    # Get max dimension of each obstacle
    max_dims = [max(w, h) for w, h in obstacle_sizes]
    max_dims.sort()

    # Use median
    median_size = max_dims[len(max_dims) // 2]

    # Cell size = 2.5x median, with reasonable bounds
    cell_size = median_size * 2.5
    cell_size = max(0.5, min(5.0, cell_size))  # Clamp to 0.5-5mm

    return round(cell_size, 2)
