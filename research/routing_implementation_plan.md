# AtoPlace Routing Implementation Plan

**Based on**: @seveibar's autorouter lessons and tscircuit implementation patterns
**Target**: Phase 3 of AtoPlace development roadmap
**Date**: January 2026

---

## Executive Summary

This plan outlines the implementation of AtoPlace's routing engine using lessons learned from tscircuit's autorouter development. Key principles:

1. **A* with Greedy Multiplier** as the core pathfinding algorithm
2. **Spatial Hash Indexing** for O(~1) collision detection
3. **Visualization-First Development** - build debug tools before algorithms
4. **Aggressive Caching** of obstacle maps and routing patterns
5. **Iterative, Non-Recursive Design** for debuggability and animation

---

## Phase 3A: Foundation (Visualization + Data Structures)

### 3A.1 Routing Visualization System

**Priority**: CRITICAL - Build before any routing algorithm

**Rationale**: "If you do not have a visualization for a problem, you will never solve it."

**Components**:

```
atoplace/routing/
├── visualizer.py          # Core visualization engine
├── svg_renderer.py        # SVG export for static frames
├── html_report.py         # Interactive HTML report generator
└── animation.py           # Iteration animation support
```

**`RouteVisualizer` Class**:
```python
class RouteVisualizer:
    """Real-time visualization of routing progress."""

    def __init__(self, board: Board, output_dir: Path = None):
        self.board = board
        self.output_dir = output_dir or Path("./route_debug")
        self.frames: List[VisualizationFrame] = []

    def render_state(self, state: RoutingState, label: str = ""):
        """Capture current routing state as a frame."""
        frame = VisualizationFrame(
            obstacles=state.obstacles,
            routed_traces=state.completed_traces,
            current_net=state.active_net,
            explored_nodes=state.astar_explored,
            frontier=state.astar_frontier,
            label=label
        )
        self.frames.append(frame)

    def export_svg(self, frame: VisualizationFrame, path: Path):
        """Export single frame as SVG."""

    def export_animation(self, fps: int = 10):
        """Export all frames as animated GIF or video."""

    def export_html_report(self):
        """Generate interactive HTML with layer toggles."""
```

**Visualization Layers**:
- Board outline and keepouts
- Component pads (color by net)
- Obstacles (existing traces, vias, copper pours)
- Current net being routed
- A* explored nodes (heat map)
- A* frontier (current candidates)
- Completed traces
- Failed routing attempts (red)

---

### 3A.2 Spatial Hash Index

**File**: `atoplace/routing/spatial_index.py`

**Rationale**: O(~1) collision detection vs O(log N) for trees

```python
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class Obstacle:
    """Routing obstacle (trace segment, pad, via, keepout)."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    layer: int
    clearance: float
    net_id: Optional[int] = None  # None = blocks all nets

class SpatialHashIndex:
    """Grid-based spatial hash for O(~1) collision queries."""

    def __init__(self, cell_size: float = 1.0):
        """
        Args:
            cell_size: Size of each hash cell in mm.
                      Too small = many cells, high memory
                      Too large = many obstacles per cell, slow queries
                      Rule of thumb: 2-3x average obstacle size
        """
        self.cell_size = cell_size
        self.cells: Dict[Tuple[int, int], List[Obstacle]] = {}

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Hash position to cell coordinates."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _get_cells_for_rect(self, min_x, min_y, max_x, max_y) -> Set[Tuple[int, int]]:
        """Get all cells that a rectangle overlaps."""
        cells = set()
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                cells.add(self._get_cell(x, y))
                y += self.cell_size
            x += self.cell_size
        return cells

    def add(self, obstacle: Obstacle):
        """Add obstacle to index."""
        for cell in self._get_cells_for_rect(
            obstacle.min_x, obstacle.min_y,
            obstacle.max_x, obstacle.max_y
        ):
            if cell not in self.cells:
                self.cells[cell] = []
            self.cells[cell].append(obstacle)

    def query(self, x: float, y: float, layer: int,
              net_id: Optional[int] = None) -> List[Obstacle]:
        """
        Query obstacles near a point.

        Args:
            x, y: Query position
            layer: Layer to check
            net_id: If provided, exclude obstacles on same net

        Returns:
            List of potentially colliding obstacles
        """
        cell = self._get_cell(x, y)
        candidates = []

        # Check cell and 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in self.cells:
                    for obs in self.cells[neighbor]:
                        if obs.layer == layer:
                            if net_id is None or obs.net_id != net_id:
                                candidates.append(obs)
        return candidates

    def check_collision(self, x: float, y: float, layer: int,
                        clearance: float, net_id: Optional[int] = None) -> bool:
        """Check if point collides with any obstacle."""
        for obs in self.query(x, y, layer, net_id):
            # Fast AABB check with clearance
            if (x >= obs.min_x - clearance and x <= obs.max_x + clearance and
                y >= obs.min_y - clearance and y <= obs.max_y + clearance):
                return True
        return False
```

**Calibration Strategy**:
```python
def auto_calibrate_cell_size(board: Board) -> float:
    """Determine optimal cell size based on board characteristics."""
    # Collect obstacle sizes
    sizes = []
    for comp in board.components.values():
        w, h = comp.width, comp.height
        sizes.append(max(w, h))

    # Cell size = 2-3x median obstacle size
    median_size = sorted(sizes)[len(sizes) // 2]
    return median_size * 2.5
```

---

### 3A.3 Obstacle Map Builder

**File**: `atoplace/routing/obstacle_map.py`

Pre-compute all obstacles before routing starts:

```python
class ObstacleMapBuilder:
    """Build comprehensive obstacle map from board."""

    def __init__(self, board: Board, dfm_profile: DFMProfile):
        self.board = board
        self.dfm = dfm_profile

    def build(self) -> SpatialHashIndex:
        """Build complete obstacle index."""
        index = SpatialHashIndex(
            cell_size=auto_calibrate_cell_size(self.board)
        )

        # 1. Component bodies (all layers they occupy)
        for ref, comp in self.board.components.items():
            self._add_component_obstacles(index, comp)

        # 2. Pads (with net association)
        for ref, comp in self.board.components.items():
            for pad in comp.pads:
                self._add_pad_obstacle(index, pad, comp)

        # 3. Board edge keepout
        self._add_edge_keepout(index)

        # 4. Existing traces (if any)
        # TODO: Add when trace model exists

        return index

    def _add_component_obstacles(self, index: SpatialHashIndex, comp: Component):
        """Add component body as obstacle."""
        bbox = comp.get_bounding_box()
        clearance = self.dfm.min_spacing

        # Determine layers blocked
        if comp.is_through_hole:
            layers = [0, 1]  # All layers
        else:
            layers = [0 if comp.layer == 'F.Cu' else 1]

        for layer in layers:
            index.add(Obstacle(
                min_x=bbox[0], min_y=bbox[1],
                max_x=bbox[2], max_y=bbox[3],
                layer=layer,
                clearance=clearance,
                net_id=None  # Blocks all nets
            ))

    def _add_pad_obstacle(self, index: SpatialHashIndex,
                         pad: Pad, comp: Component):
        """Add pad as obstacle (but allow same-net traces)."""
        # Transform pad position to board coordinates
        px, py = pad.x + comp.x, pad.y + comp.y
        hw, hh = pad.width / 2, pad.height / 2

        # Determine net
        net_id = None
        if pad.net:
            net_id = hash(pad.net)  # Use net name hash as ID

        layers = [0, 1] if pad.is_through_hole else [pad.layer]

        for layer in layers:
            index.add(Obstacle(
                min_x=px - hw, min_y=py - hh,
                max_x=px + hw, max_y=py + hh,
                layer=layer,
                clearance=self.dfm.min_spacing,
                net_id=net_id
            ))
```

---

## Phase 3B: Core A* Router

### 3B.1 A* with Greedy Multiplier

**File**: `atoplace/routing/astar_router.py`

```python
import heapq
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

@dataclass
class RouteNode:
    """Node in A* search space."""
    x: float
    y: float
    layer: int
    via_count: int = 0  # Number of vias to reach this node

    def __hash__(self):
        return hash((round(self.x, 3), round(self.y, 3), self.layer))

    def __eq__(self, other):
        return (round(self.x, 3) == round(other.x, 3) and
                round(self.y, 3) == round(other.y, 3) and
                self.layer == other.layer)

@dataclass
class RouteSegment:
    """A segment of a routed trace."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: int
    width: float

@dataclass
class RoutingResult:
    """Result of routing a single net."""
    success: bool
    segments: List[RouteSegment]
    vias: List[Tuple[float, float]]  # Via positions
    iterations: int
    explored_count: int

class AStarRouter:
    """A* pathfinder with greedy multiplier for PCB routing."""

    def __init__(
        self,
        obstacle_index: SpatialHashIndex,
        dfm_profile: DFMProfile,
        greedy_weight: float = 2.0,
        grid_size: float = 0.1,
        allow_45_degree: bool = True,
        visualizer: Optional[RouteVisualizer] = None
    ):
        """
        Args:
            obstacle_index: Pre-built spatial hash of obstacles
            dfm_profile: Design rules for clearances
            greedy_weight: A* heuristic multiplier (1.0=optimal, 2-3=fast)
            grid_size: Routing grid resolution in mm
            allow_45_degree: Allow 45-degree routing (vs Manhattan only)
            visualizer: Optional visualizer for debugging
        """
        self.obstacles = obstacle_index
        self.dfm = dfm_profile
        self.w = greedy_weight
        self.grid = grid_size
        self.allow_45 = allow_45_degree
        self.viz = visualizer

        # Precompute neighbor offsets
        self._setup_neighbor_offsets()

    def _setup_neighbor_offsets(self):
        """Precompute valid movement directions."""
        g = self.grid
        self.offsets = [
            (g, 0), (-g, 0), (0, g), (0, -g)  # Manhattan
        ]
        if self.allow_45:
            d = g * 0.7071  # sqrt(2)/2 for 45-degree
            self.offsets.extend([
                (d, d), (d, -d), (-d, d), (-d, -d)
            ])

    def route_net(
        self,
        start_pads: List[Tuple[float, float, int]],  # (x, y, layer)
        net_id: int,
        trace_width: float
    ) -> RoutingResult:
        """
        Route a net connecting multiple pads.

        Uses iterative Steiner-tree approach:
        1. Start from first pad
        2. Route to nearest unconnected pad
        3. Add routed trace as "connected" area
        4. Repeat until all pads connected

        Returns:
            RoutingResult with success status and trace segments
        """
        if len(start_pads) < 2:
            return RoutingResult(True, [], [], 0, 0)

        all_segments = []
        all_vias = []
        total_iterations = 0
        total_explored = 0

        # Start with first pad as connected
        connected_points = {start_pads[0]}
        remaining_pads = set(start_pads[1:])

        while remaining_pads:
            # Find nearest unconnected pad to any connected point
            best_start, best_end, best_dist = None, None, float('inf')
            for conn in connected_points:
                for rem in remaining_pads:
                    dist = self._heuristic(
                        RouteNode(conn[0], conn[1], conn[2]),
                        RouteNode(rem[0], rem[1], rem[2])
                    )
                    if dist < best_dist:
                        best_start, best_end, best_dist = conn, rem, dist

            # Route between best pair
            result = self._route_two_points(
                RouteNode(best_start[0], best_start[1], best_start[2]),
                RouteNode(best_end[0], best_end[1], best_end[2]),
                net_id,
                trace_width
            )

            if not result.success:
                return RoutingResult(False, all_segments, all_vias,
                                    total_iterations, total_explored)

            all_segments.extend(result.segments)
            all_vias.extend(result.vias)
            total_iterations += result.iterations
            total_explored += result.explored_count

            # Mark end as connected
            connected_points.add(best_end)
            remaining_pads.remove(best_end)

            # Add trace points as connected (for future routing)
            for seg in result.segments:
                connected_points.add((seg.end[0], seg.end[1], seg.layer))

        return RoutingResult(True, all_segments, all_vias,
                            total_iterations, total_explored)

    def _route_two_points(
        self,
        start: RouteNode,
        end: RouteNode,
        net_id: int,
        trace_width: float
    ) -> RoutingResult:
        """Route between two points using A*."""
        clearance = trace_width / 2 + self.dfm.min_spacing

        # Priority queue: (f_score, node, path)
        open_set = [(0, start, [start])]
        g_scores = {start: 0}
        explored = set()
        iterations = 0

        while open_set:
            iterations += 1
            _, current, path = heapq.heappop(open_set)

            # Visualization hook
            if self.viz and iterations % 100 == 0:
                self.viz.render_state(RoutingState(
                    explored_nodes=explored,
                    frontier=[n for _, n, _ in open_set],
                    current_path=path
                ), f"Iteration {iterations}")

            # Goal check (with tolerance)
            if self._is_goal(current, end):
                return self._build_result(path, trace_width, iterations, len(explored))

            if current in explored:
                continue
            explored.add(current)

            # Expand neighbors
            for neighbor in self._get_neighbors(current, end, net_id, clearance):
                if neighbor in explored:
                    continue

                # Cost: distance + via penalty
                move_cost = self._move_cost(current, neighbor)
                tentative_g = g_scores[current] + move_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h = self._heuristic(neighbor, end)
                    f = tentative_g + self.w * h  # GREEDY MULTIPLIER
                    heapq.heappush(open_set, (f, neighbor, path + [neighbor]))

        # No path found
        return RoutingResult(False, [], [], iterations, len(explored))

    def _get_neighbors(
        self,
        node: RouteNode,
        goal: RouteNode,
        net_id: int,
        clearance: float
    ) -> List[RouteNode]:
        """Get valid neighboring nodes."""
        neighbors = []

        # Same-layer moves
        for dx, dy in self.offsets:
            nx, ny = node.x + dx, node.y + dy
            if not self.obstacles.check_collision(nx, ny, node.layer, clearance, net_id):
                neighbors.append(RouteNode(nx, ny, node.layer, node.via_count))

        # Layer change (via)
        other_layer = 1 - node.layer
        if not self.obstacles.check_collision(node.x, node.y, other_layer, clearance, net_id):
            # Check via drill clearance
            via_clearance = self.dfm.min_via_diameter / 2 + self.dfm.min_spacing
            if not self._check_via_collision(node.x, node.y, via_clearance, net_id):
                neighbors.append(RouteNode(node.x, node.y, other_layer, node.via_count + 1))

        return neighbors

    def _heuristic(self, node: RouteNode, goal: RouteNode) -> float:
        """Estimate cost to reach goal (admissible heuristic)."""
        # Manhattan distance + layer change penalty
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        dist = dx + dy

        # Add via penalty if on different layer
        if node.layer != goal.layer:
            dist += self.dfm.min_via_diameter * 5  # Via penalty

        return dist

    def _move_cost(self, current: RouteNode, neighbor: RouteNode) -> float:
        """Actual cost to move from current to neighbor."""
        if current.layer != neighbor.layer:
            # Via cost: drilling expense + signal integrity
            return self.dfm.min_via_diameter * 10
        else:
            # Distance cost
            dx = neighbor.x - current.x
            dy = neighbor.y - current.y
            return (dx*dx + dy*dy) ** 0.5

    def _is_goal(self, node: RouteNode, goal: RouteNode) -> bool:
        """Check if node is close enough to goal."""
        if node.layer != goal.layer:
            return False
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        return dx <= self.grid and dy <= self.grid

    def _build_result(
        self,
        path: List[RouteNode],
        trace_width: float,
        iterations: int,
        explored: int
    ) -> RoutingResult:
        """Convert node path to trace segments and vias."""
        segments = []
        vias = []

        for i in range(len(path) - 1):
            curr, next_ = path[i], path[i + 1]

            if curr.layer != next_.layer:
                # Via
                vias.append((curr.x, curr.y))
            else:
                # Trace segment
                segments.append(RouteSegment(
                    start=(curr.x, curr.y),
                    end=(next_.x, next_.y),
                    layer=curr.layer,
                    width=trace_width
                ))

        return RoutingResult(True, segments, vias, iterations, explored)
```

---

### 3B.2 Net Ordering Strategy

**File**: `atoplace/routing/net_orderer.py`

Route difficult nets first (spatial probability of failure):

```python
class NetOrderer:
    """Determine optimal net routing order."""

    def __init__(self, board: Board, obstacle_index: SpatialHashIndex):
        self.board = board
        self.obstacles = obstacle_index

    def order_nets(self) -> List[Net]:
        """
        Order nets by routing difficulty (hardest first).

        Factors:
        1. Congestion at pad locations
        2. Net length (longer = harder)
        3. Number of pads (more = harder)
        4. Critical nets (power/ground) first
        """
        net_scores = []

        for net in self.board.nets.values():
            score = self._calculate_difficulty(net)
            net_scores.append((score, net))

        # Sort by difficulty descending (hardest first)
        net_scores.sort(key=lambda x: -x[0])
        return [net for _, net in net_scores]

    def _calculate_difficulty(self, net: Net) -> float:
        """Calculate routing difficulty score."""
        score = 0.0

        # Get pad positions
        pads = self._get_net_pads(net)
        if len(pads) < 2:
            return 0.0

        # 1. Congestion: count obstacles near each pad
        for px, py, layer in pads:
            nearby = self.obstacles.query(px, py, layer)
            score += len(nearby) * 10

        # 2. Net length estimate (bounding box diagonal)
        xs = [p[0] for p in pads]
        ys = [p[1] for p in pads]
        diagonal = ((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2) ** 0.5
        score += diagonal

        # 3. Pad count
        score += len(pads) * 5

        # 4. Critical nets bonus (route first)
        if net.is_power or net.is_ground:
            score += 1000

        return score
```

---

### 3B.3 Routing Result Caching

**File**: `atoplace/routing/cache.py`

```python
import hashlib
import pickle
from pathlib import Path

class RoutingCache:
    """Cache successful routing patterns for reuse."""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".atoplace" / "routing_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, start: Tuple, end: Tuple, obstacles_hash: str) -> str:
        """Generate cache key for routing problem."""
        data = f"{start}-{end}-{obstacles_hash}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, start, end, obstacles_hash) -> Optional[RoutingResult]:
        """Retrieve cached routing result."""
        key = self._make_key(start, end, obstacles_hash)
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def put(self, start, end, obstacles_hash, result: RoutingResult):
        """Store routing result in cache."""
        key = self._make_key(start, end, obstacles_hash)
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

---

## Phase 3C: Integration

### 3C.1 CLI Integration

**File**: `atoplace/cli.py` additions

```python
@cli.command()
@click.argument("board_path", type=click.Path(exists=True))
@click.option("--greedy", default=2.0, help="A* greedy multiplier (1=optimal, 2-3=fast)")
@click.option("--grid", default=0.1, help="Routing grid size in mm")
@click.option("--visualize", is_flag=True, help="Generate visualization")
@click.option("--animate", is_flag=True, help="Generate routing animation")
def route(board_path, greedy, grid, visualize, animate):
    """Route all nets on the board."""
    board = load_board(board_path)
    dfm = auto_select_dfm_profile(board)

    # Build obstacles
    obstacle_builder = ObstacleMapBuilder(board, dfm)
    obstacles = obstacle_builder.build()

    # Setup visualizer
    viz = RouteVisualizer(board) if visualize or animate else None

    # Create router
    router = AStarRouter(
        obstacle_index=obstacles,
        dfm_profile=dfm,
        greedy_weight=greedy,
        grid_size=grid,
        visualizer=viz
    )

    # Order nets
    orderer = NetOrderer(board, obstacles)
    ordered_nets = orderer.order_nets()

    # Route each net
    results = {}
    for net in track(ordered_nets, description="Routing"):
        pads = get_net_pads(board, net)
        result = router.route_net(pads, hash(net.name), dfm.min_trace_width)
        results[net.name] = result

        if result.success:
            # Add routed traces to obstacles for subsequent nets
            add_traces_to_obstacles(obstacles, result.segments, dfm)

    # Report
    success_count = sum(1 for r in results.values() if r.success)
    console.print(f"Routed {success_count}/{len(results)} nets")

    if viz:
        if animate:
            viz.export_animation()
        viz.export_html_report()
```

---

## Phase 3D: Freerouting Fallback

For complex boards where our A* router fails, fall back to Freerouting:

```python
class FreeroutingFallback:
    """Use Freerouting for nets that fail A* routing."""

    def __init__(self, board: Board, freerouting_jar: Path):
        self.board = board
        self.jar = freerouting_jar

    def route_failed_nets(self, failed_nets: List[str]) -> bool:
        """Export to DSN, run Freerouting, import SES."""
        # 1. Export board with only failed nets unrouted
        dsn_path = self._export_dsn(failed_nets)

        # 2. Run Freerouting
        subprocess.run([
            "java", "-jar", str(self.jar),
            "-de", str(dsn_path),
            "-do", str(dsn_path.with_suffix('.ses')),
            "-mp", "10"  # 10 routing passes
        ], check=True)

        # 3. Import results
        return self._import_ses(dsn_path.with_suffix('.ses'))
```

---

## Implementation Timeline

| Phase | Components | Estimated Effort |
|-------|------------|------------------|
| 3A.1 | Visualization System | Medium |
| 3A.2 | Spatial Hash Index | Small |
| 3A.3 | Obstacle Map Builder | Small |
| 3B.1 | A* Router Core | Large |
| 3B.2 | Net Ordering | Small |
| 3B.3 | Result Caching | Small |
| 3C.1 | CLI Integration | Medium |
| 3D | Freerouting Fallback | Medium |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Route completion (simple boards) | >95% |
| Route completion (medium boards) | >80% |
| Routing speed (<50 nets) | <10 seconds |
| Routing speed (50-100 nets) | <60 seconds |
| Via count | Within 20% of optimal |

---

## References

- [@seveibar's autorouter lessons](https://blog.autorouting.com/p/13-things-i-would-have-told-myself)
- [tscircuit autorouting repo](https://github.com/tscircuit/autorouting)
- [Freerouting](https://github.com/freerouting/freerouting)
- `research/autorouter_lessons_seveibar.md` - Detailed lesson notes
