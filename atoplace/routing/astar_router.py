"""A* pathfinding router with greedy multiplier.

Based on @seveibar's autorouter lessons:
- Lesson #1: Know A* like the back of your hand
- Lesson #7: Never use recursive functions (iterative with explicit queue)
- Lesson #13: The "Greedy Multiplier" - secret hack to 100x A* performance

The greedy multiplier trades optimality for speed:
- Standard A*: f(n) = g(n) + h(n)
- Weighted A*: f(n) = g(n) + w * h(n) where w > 1

For PCB routing, w = 2-3 is often ideal - much faster with acceptable paths.
"""

import heapq
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum

from .spatial_index import SpatialHashIndex, Obstacle
from .visualizer import RouteVisualizer, RouteSegment, Via

logger = logging.getLogger(__name__)


class RouteDirection(Enum):
    """Allowed routing directions."""
    MANHATTAN = "manhattan"  # Only horizontal/vertical
    DIAGONAL = "diagonal"    # Allow 45-degree angles
    ANY = "any"              # Any angle (gridless)


@dataclass(frozen=True)
class RouteNode:
    """A node in the A* search space.

    Immutable and hashable for use in sets and priority queues.
    """
    x: float
    y: float
    layer: int

    def __hash__(self):
        # Round to grid precision for consistent hashing
        return hash((round(self.x, 3), round(self.y, 3), self.layer))

    def __eq__(self, other):
        if not isinstance(other, RouteNode):
            return False
        return (round(self.x, 3) == round(other.x, 3) and
                round(self.y, 3) == round(other.y, 3) and
                self.layer == other.layer)

    def __lt__(self, other):
        # For heapq tie-breaking
        return (self.x, self.y, self.layer) < (other.x, other.y, other.layer)

    def distance_to(self, other: "RouteNode") -> float:
        """Euclidean distance to another node."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def manhattan_distance_to(self, other: "RouteNode") -> float:
        """Manhattan distance to another node."""
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class RoutingResult:
    """Result of routing a single net."""
    success: bool
    net_name: str = ""
    segments: List[RouteSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    iterations: int = 0
    explored_count: int = 0
    total_length: float = 0.0
    via_count: int = 0
    failure_reason: str = ""


@dataclass
class RouterConfig:
    """Configuration for the A* router."""
    # A* parameters
    greedy_weight: float = 2.0  # Heuristic multiplier (1=optimal, 2-3=fast)
    grid_size: float = 0.1     # Routing grid resolution in mm
    max_iterations: int = 50000  # Max A* iterations per net

    # Routing style
    direction: RouteDirection = RouteDirection.DIAGONAL
    prefer_layer: Optional[int] = None  # Preferred routing layer (0=top, 1=bottom)

    # Via parameters
    via_cost: float = 5.0      # Cost multiplier for layer changes
    min_via_spacing: float = 0.5  # Minimum spacing between vias

    # Trace parameters
    trace_width: float = 0.2   # Default trace width in mm
    clearance: float = 0.15    # Clearance from obstacles

    # Goal tolerance
    goal_tolerance: float = 0.2  # Distance to consider "at goal"


class AStarRouter:
    """A* pathfinding router for PCB traces.

    Uses weighted A* (greedy multiplier) for fast routing with
    acceptable path quality. Supports:
    - Multi-layer routing with vias
    - Manhattan, diagonal, or any-angle routing
    - Same-net obstacle filtering
    - Integration with SpatialHashIndex for O(~1) collision detection
    """

    def __init__(
        self,
        obstacle_index: SpatialHashIndex,
        config: RouterConfig = None,
        visualizer: Optional[RouteVisualizer] = None
    ):
        """
        Args:
            obstacle_index: Pre-built spatial hash of obstacles
            config: Router configuration
            visualizer: Optional visualizer for debugging
        """
        self.obstacles = obstacle_index
        self.config = config or RouterConfig()
        self.viz = visualizer

        # Precompute neighbor offsets based on routing direction
        self._setup_neighbor_offsets()

        # Track routed traces (added as obstacles for subsequent nets)
        self.routed_segments: List[RouteSegment] = []
        self.routed_vias: List[Via] = []

    def _setup_neighbor_offsets(self):
        """Precompute valid movement directions based on config."""
        g = self.config.grid_size
        self.offsets = []

        # Manhattan directions (always included)
        self.offsets.extend([
            (g, 0, 1.0),      # Right
            (-g, 0, 1.0),     # Left
            (0, g, 1.0),      # Up
            (0, -g, 1.0),     # Down
        ])

        if self.config.direction in (RouteDirection.DIAGONAL, RouteDirection.ANY):
            # 45-degree diagonals
            d = g * math.sqrt(2)
            self.offsets.extend([
                (g, g, d),     # Up-right
                (g, -g, d),    # Down-right
                (-g, g, d),    # Up-left
                (-g, -g, d),   # Down-left
            ])

    def route_net(
        self,
        pads: List[Obstacle],
        net_name: str = "",
        net_id: Optional[int] = None,
        trace_width: Optional[float] = None
    ) -> RoutingResult:
        """
        Route a net connecting multiple pads.

        Uses iterative Steiner-tree approach:
        1. Start from first pad
        2. Route to nearest unconnected pad
        3. Mark routed path as "connected"
        4. Repeat until all pads connected

        Args:
            pads: List of pad obstacles to connect
            net_name: Name of the net (for logging)
            net_id: Net ID for same-net filtering
            trace_width: Trace width override

        Returns:
            RoutingResult with success status and trace segments
        """
        if len(pads) < 2:
            return RoutingResult(
                success=True,
                net_name=net_name,
                segments=[],
                vias=[],
            )

        trace_width = trace_width or self.config.trace_width
        all_segments = []
        all_vias = []
        total_iterations = 0
        total_explored = 0
        total_length = 0.0

        # Convert pads to RouteNodes (use pad centers)
        pad_nodes = []
        for pad in pads:
            cx, cy = pad.center
            # Use pad layer, or default to layer 0 for through-hole (-1)
            layer = pad.layer if pad.layer >= 0 else 0
            pad_nodes.append(RouteNode(cx, cy, layer))

        # Start with first pad as connected
        connected_nodes = {pad_nodes[0]}
        connected_set = {pad_nodes[0]}  # For fast lookup
        remaining_nodes = set(pad_nodes[1:])

        while remaining_nodes:
            # Find nearest unconnected pad to any connected point
            best_start = None
            best_end = None
            best_dist = float('inf')

            for conn in connected_nodes:
                for rem in remaining_nodes:
                    dist = conn.distance_to(rem)
                    if dist < best_dist:
                        best_start = conn
                        best_end = rem
                        best_dist = dist

            if best_start is None or best_end is None:
                return RoutingResult(
                    success=False,
                    net_name=net_name,
                    segments=all_segments,
                    vias=all_vias,
                    iterations=total_iterations,
                    explored_count=total_explored,
                    failure_reason="No valid start/end pair found"
                )

            # Route between best pair
            # Try on original layers first
            result = self._route_two_points(
                best_start,
                best_end,
                net_id,
                trace_width,
                net_name
            )

            # If failed quickly, try starting on alternate layer
            # This helps when one layer is congested but the other is clear
            if not result.success and result.iterations < 1000:
                alt_start = RouteNode(best_start.x, best_start.y, 1 - best_start.layer)
                alt_result = self._route_two_points(
                    alt_start,
                    best_end,
                    net_id,
                    trace_width,
                    net_name
                )
                if alt_result.success or alt_result.iterations > result.iterations:
                    # Alternate layer found a path or explored more - use it
                    if alt_result.success:
                        # Add via to switch to alternate layer at start
                        from .visualizer import Via
                        all_vias.append(Via(
                            x=best_start.x, y=best_start.y,
                            drill_diameter=0.3, pad_diameter=0.6,
                            net_id=net_id
                        ))
                    result = alt_result

            total_iterations += result.iterations
            total_explored += result.explored_count

            if not result.success:
                logger.warning(
                    f"Failed to route {net_name}: {best_start} -> {best_end}: "
                    f"{result.failure_reason}"
                )
                return RoutingResult(
                    success=False,
                    net_name=net_name,
                    segments=all_segments,
                    vias=all_vias,
                    iterations=total_iterations,
                    explored_count=total_explored,
                    failure_reason=f"Failed segment: {result.failure_reason}"
                )

            all_segments.extend(result.segments)
            all_vias.extend(result.vias)
            total_length += result.total_length

            # Mark end as connected
            connected_nodes.add(best_end)
            connected_set.add(best_end)
            remaining_nodes.remove(best_end)

            # Add intermediate path points as connected (for star routing)
            for seg in result.segments:
                connected_nodes.add(RouteNode(seg.end[0], seg.end[1], seg.layer))

        return RoutingResult(
            success=True,
            net_name=net_name,
            segments=all_segments,
            vias=all_vias,
            iterations=total_iterations,
            explored_count=total_explored,
            total_length=total_length,
            via_count=len(all_vias)
        )

    def _route_two_points(
        self,
        start: RouteNode,
        goal: RouteNode,
        net_id: Optional[int],
        trace_width: float,
        net_name: str = ""
    ) -> RoutingResult:
        """Route between two points using A* with greedy multiplier.

        This is the core A* implementation - iterative, not recursive.
        """
        clearance = trace_width / 2 + self.config.clearance

        # Priority queue: (f_score, g_score, node, parent_map_key)
        # Using g_score as secondary sort for tie-breaking
        start_h = self._heuristic(start, goal)
        start_f = start_h * self.config.greedy_weight

        # open_set: heap of (f, g, node)
        open_set = [(start_f, 0.0, start)]
        heapq.heapify(open_set)

        # Track best g_score and parent for path reconstruction
        g_scores: Dict[RouteNode, float] = {start: 0.0}
        came_from: Dict[RouteNode, RouteNode] = {}

        # Closed set for already-explored nodes
        closed: Set[RouteNode] = set()

        iterations = 0
        viz_interval = 500  # Capture frame every N iterations

        while open_set:
            iterations += 1

            if iterations > self.config.max_iterations:
                return RoutingResult(
                    success=False,
                    net_name=net_name,
                    iterations=iterations,
                    explored_count=len(closed),
                    failure_reason=f"Max iterations ({self.config.max_iterations}) exceeded"
                )

            # Pop node with lowest f_score
            _, current_g, current = heapq.heappop(open_set)

            # Skip if already processed with better score
            if current in closed:
                continue

            # Visualization hook
            if self.viz and iterations % viz_interval == 0:
                self._capture_viz_frame(
                    closed, open_set, current, goal, net_name, iterations
                )

            # Goal check
            if self._is_at_goal(current, goal):
                # Reconstruct path
                path = self._reconstruct_path(came_from, current, start)
                return self._build_result(
                    path, trace_width, iterations, len(closed), net_name
                )

            closed.add(current)

            # Expand neighbors
            for neighbor, move_cost in self._get_neighbors(
                current, goal, net_id, clearance
            ):
                if neighbor in closed:
                    continue

                tentative_g = current_g + move_cost

                # Only process if this is a better path
                if neighbor in g_scores and tentative_g >= g_scores[neighbor]:
                    continue

                # Update path
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current

                # Calculate f with greedy multiplier
                h = self._heuristic(neighbor, goal)
                f = tentative_g + self.config.greedy_weight * h

                heapq.heappush(open_set, (f, tentative_g, neighbor))

        # No path found
        return RoutingResult(
            success=False,
            net_name=net_name,
            iterations=iterations,
            explored_count=len(closed),
            failure_reason="No path found (open set exhausted)"
        )

    def _get_neighbors(
        self,
        node: RouteNode,
        goal: RouteNode,
        net_id: Optional[int],
        clearance: float
    ) -> List[Tuple[RouteNode, float]]:
        """Get valid neighboring nodes with movement costs.

        Returns list of (neighbor_node, cost) tuples.
        """
        neighbors = []

        # Same-layer moves
        for dx, dy, base_cost in self.offsets:
            nx = node.x + dx
            ny = node.y + dy

            # Check collision
            if not self._check_path_clear(
                node.x, node.y, nx, ny, node.layer, clearance, net_id
            ):
                continue

            neighbor = RouteNode(nx, ny, node.layer)
            neighbors.append((neighbor, base_cost))

        # Layer change (via) - only if we have 2 layers
        other_layer = 1 - node.layer
        if self._can_place_via(node.x, node.y, clearance, net_id):
            # Check if destination layer is clear
            if not self.obstacles.check_collision(
                node.x, node.y, other_layer, clearance, net_id
            ):
                via_cost = self.config.via_cost * self.config.grid_size
                neighbor = RouteNode(node.x, node.y, other_layer)
                neighbors.append((neighbor, via_cost))

        return neighbors

    def _check_path_clear(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        layer: int,
        clearance: float,
        net_id: Optional[int]
    ) -> bool:
        """Check if path between two points is clear of obstacles."""
        # For short moves, just check endpoint
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if dist < self.config.grid_size * 1.5:
            return not self.obstacles.check_collision(x2, y2, layer, clearance, net_id)

        # For longer moves, check segment
        return not self.obstacles.check_segment_collision(
            x1, y1, x2, y2, layer, clearance * 2, net_id
        )

    def _can_place_via(
        self,
        x: float, y: float,
        clearance: float,
        net_id: Optional[int]
    ) -> bool:
        """Check if a via can be placed at this location."""
        via_clearance = self.config.min_via_spacing / 2 + clearance

        # Check both layers
        if self.obstacles.check_collision(x, y, 0, via_clearance, net_id):
            return False
        if self.obstacles.check_collision(x, y, 1, via_clearance, net_id):
            return False

        return True

    def _heuristic(self, node: RouteNode, goal: RouteNode) -> float:
        """Estimate cost to reach goal (admissible heuristic).

        Uses Manhattan distance plus layer change penalty.
        Must be admissible (never overestimate) for A* optimality,
        but with greedy multiplier we trade optimality for speed.
        """
        # Manhattan distance (admissible for grid routing)
        dist = node.manhattan_distance_to(goal)

        # Add layer change penalty if on different layer
        if node.layer != goal.layer:
            dist += self.config.via_cost * self.config.grid_size

        return dist

    def _is_at_goal(self, node: RouteNode, goal: RouteNode) -> bool:
        """Check if node is close enough to goal."""
        if node.layer != goal.layer:
            return False
        return node.distance_to(goal) <= self.config.goal_tolerance

    def _reconstruct_path(
        self,
        came_from: Dict[RouteNode, RouteNode],
        current: RouteNode,
        start: RouteNode
    ) -> List[RouteNode]:
        """Reconstruct path from start to current using came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _build_result(
        self,
        path: List[RouteNode],
        trace_width: float,
        iterations: int,
        explored: int,
        net_name: str
    ) -> RoutingResult:
        """Convert node path to trace segments and vias."""
        segments = []
        vias = []
        total_length = 0.0

        for i in range(len(path) - 1):
            curr = path[i]
            next_ = path[i + 1]

            if curr.layer != next_.layer:
                # Via (layer change)
                vias.append(Via(
                    x=curr.x,
                    y=curr.y,
                    drill_diameter=0.3,  # Standard via
                    pad_diameter=0.6,
                    net_id=None  # TODO: pass net_id
                ))
            else:
                # Trace segment
                seg = RouteSegment(
                    start=(curr.x, curr.y),
                    end=(next_.x, next_.y),
                    layer=curr.layer,
                    width=trace_width
                )
                segments.append(seg)
                total_length += curr.distance_to(next_)

        return RoutingResult(
            success=True,
            net_name=net_name,
            segments=segments,
            vias=vias,
            iterations=iterations,
            explored_count=explored,
            total_length=total_length,
            via_count=len(vias)
        )

    def _capture_viz_frame(
        self,
        closed: Set[RouteNode],
        open_set: List,
        current: RouteNode,
        goal: RouteNode,
        net_name: str,
        iterations: int
    ):
        """Capture visualization frame for debugging."""
        if not self.viz:
            return

        explored = {(n.x, n.y, n.layer) for n in closed}
        frontier = {(item[2].x, item[2].y, item[2].layer) for item in open_set}

        self.viz.capture_frame(
            obstacles=[],  # Already rendered
            pads=[],
            explored_nodes=explored,
            frontier_nodes=frontier,
            current_path=[(current.x, current.y, current.layer)],
            current_net=net_name,
            label=f"Iter {iterations}"
        )

    def add_routed_trace(self, segment: RouteSegment):
        """Add a routed trace segment as an obstacle for future routing."""
        self.routed_segments.append(segment)

        # Add to obstacle index
        min_x = min(segment.start[0], segment.end[0]) - segment.width / 2
        max_x = max(segment.start[0], segment.end[0]) + segment.width / 2
        min_y = min(segment.start[1], segment.end[1]) - segment.width / 2
        max_y = max(segment.start[1], segment.end[1]) + segment.width / 2

        self.obstacles.add(Obstacle(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            layer=segment.layer,
            clearance=self.config.clearance,
            net_id=segment.net_id,
            obstacle_type="trace"
        ))

    def add_routed_via(self, via: Via):
        """Add a routed via as an obstacle for future routing."""
        self.routed_vias.append(via)

        # Add to obstacle index (blocks all layers)
        r = via.pad_diameter / 2
        self.obstacles.add(Obstacle(
            min_x=via.x - r,
            min_y=via.y - r,
            max_x=via.x + r,
            max_y=via.y + r,
            layer=-1,  # All layers
            clearance=self.config.clearance,
            net_id=via.net_id,
            obstacle_type="via"
        ))


class NetOrderer:
    """Determine optimal net routing order.

    Based on @seveibar's lesson #12: Measure spatial probability of failure.
    Route difficult nets first to maximize success rate.
    """

    def __init__(self, obstacle_index: SpatialHashIndex):
        self.obstacles = obstacle_index

    def order_nets(self, nets: List) -> List:
        """
        Order nets by routing difficulty (hardest first).

        Factors:
        1. Congestion at pad locations (more obstacles = harder)
        2. Net length (longer = harder)
        3. Number of pads (more = harder)
        4. Critical nets (power/ground) get priority

        Args:
            nets: List of NetPads objects

        Returns:
            Sorted list with hardest nets first
        """
        scored_nets = []

        for net in nets:
            score = self._calculate_difficulty(net)
            scored_nets.append((score, net))

        # Sort by difficulty descending (hardest first)
        scored_nets.sort(key=lambda x: -x[0])

        return [net for _, net in scored_nets]

    def _calculate_difficulty(self, net) -> float:
        """Calculate routing difficulty score for a net."""
        score = 0.0
        pads = net.pads

        if len(pads) < 2:
            return 0.0

        # 1. Congestion: count obstacles near each pad
        for pad in pads:
            cx, cy = pad.center
            layer = pad.layer if pad.layer >= 0 else 0
            nearby = self.obstacles.query_point(cx, cy, layer)
            score += len(nearby) * 10

        # 2. Net length estimate (bounding box diagonal)
        xs = [p.center[0] for p in pads]
        ys = [p.center[1] for p in pads]
        if xs and ys:
            diagonal = math.sqrt(
                (max(xs) - min(xs))**2 + (max(ys) - min(ys))**2
            )
            score += diagonal

        # 3. Pad count (more pads = more complex routing)
        score += len(pads) * 5

        # 4. Critical nets bonus (route first)
        if hasattr(net, 'is_power') and net.is_power:
            score += 1000
        if hasattr(net, 'is_ground') and net.is_ground:
            score += 1000

        return score


def route_board(
    board,
    dfm_profile,
    config: RouterConfig = None,
    visualize: bool = False
) -> Dict[str, RoutingResult]:
    """
    Convenience function to route all nets on a board.

    Args:
        board: Board instance
        dfm_profile: DFM profile for clearances
        config: Router configuration
        visualize: Whether to generate visualization

    Returns:
        Dictionary mapping net name to RoutingResult
    """
    from .obstacle_map import ObstacleMapBuilder
    from .visualizer import create_visualizer_from_board

    # Build obstacle map
    builder = ObstacleMapBuilder(board, dfm_profile)
    obstacle_index = builder.build()
    nets = builder.get_net_pads()

    # Create visualizer if requested
    viz = create_visualizer_from_board(board) if visualize else None

    # Configure router
    config = config or RouterConfig(
        trace_width=dfm_profile.min_trace_width,
        clearance=dfm_profile.min_spacing,
    )

    # Create router
    router = AStarRouter(obstacle_index, config, viz)

    # Order nets by difficulty
    orderer = NetOrderer(obstacle_index)
    ordered_nets = orderer.order_nets(nets)

    # Route each net
    results = {}
    success_count = 0

    for net in ordered_nets:
        result = router.route_net(
            pads=net.pads,
            net_name=net.net_name,
            net_id=net.net_id,
            trace_width=config.trace_width
        )
        results[net.net_name] = result

        if result.success:
            success_count += 1
            # Add routed traces as obstacles for subsequent nets
            for seg in result.segments:
                seg.net_id = net.net_id
                router.add_routed_trace(seg)
            for via in result.vias:
                via.net_id = net.net_id
                router.add_routed_via(via)

            logger.debug(
                f"Routed {net.net_name}: {len(result.segments)} segments, "
                f"{result.via_count} vias, {result.total_length:.1f}mm"
            )
        else:
            logger.warning(f"Failed to route {net.net_name}: {result.failure_reason}")

    logger.info(f"Routing complete: {success_count}/{len(nets)} nets routed")

    # Export visualization if enabled
    if viz:
        viz.export_html_report("routing_result.html")

    return results
