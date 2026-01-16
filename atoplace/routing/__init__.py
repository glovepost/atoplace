"""Routing engine for AtoPlace.

Phase 0 - Pin Optimization (implemented):
- PinSwapper: Optimize pin assignments before routing
- SwapGroupDetector: Detect FPGA banks, MCU GPIO, connectors
- CrossingCounter: Measure ratsnest complexity
- BipartiteMatcher: Hungarian algorithm for optimal assignment
- ConstraintGenerator: Export XDC/QSF/TCL constraint files

Phase 3A - Foundation components:
- SpatialHashIndex: O(~1) collision detection using spatial hashing
- ObstacleMapBuilder: Pre-compute all routing obstacles from board
- RouteVisualizer: SVG/HTML visualization for debugging

Phase 3B - Core router (implemented):
- AStarRouter: A* pathfinder with greedy multiplier
- NetOrderer: Route difficult nets first
- route_board: Convenience function to route all nets

Phase 3B+ - Routing Manager (implemented):
- RoutingManager: Orchestrates the multi-phase routing pipeline
- Coordinates fanout → critical nets → general routing

Phase 3C - Integration (planned):
- FreeroutingRunner: Fallback to Freerouting for complex boards

Phase 3D - BGA Fanout (implemented):
- FanoutGenerator: Automated escape routing for high-density BGAs
- Dogbone/VIP patterns for different pitch sizes
- Onion model layer assignment
- Escape routing from vias to routing field

Phase 2 - Critical Nets (implemented):
- DiffPairDetector: Auto-detect differential pairs from net names
- DiffPairRouter: Route coupled differential pairs
- LengthMatcher: Add meanders for length matching
"""

# Phase 0 - Pin Optimization
from .pinswapper import (
    PinSwapper,
    SwapResult,
    SwapConfig,
    SwapGroupDetector,
    SwapGroup,
    SwapGroupType,
    SwappablePin,
    CrossingCounter,
    CrossingResult,
    RatsnestEdge,
    BipartiteMatcher,
    MatchingResult,
    SwapAssignment,
    ConstraintGenerator,
    ConstraintFormat,
    ConstraintUpdate,
)

# Phase 3A - Foundation
from .spatial_index import SpatialHashIndex, Obstacle, auto_calibrate_cell_size
from .obstacle_map import ObstacleMapBuilder, NetPads, build_obstacle_map
from .visualizer import (
    RouteVisualizer,
    VisualizationFrame,
    RouteSegment,
    Via,
    create_visualizer_from_board,
)

# Phase 3B - Core Router
from .astar_router import (
    AStarRouter,
    RouterConfig,
    RoutingResult,
    RouteNode,
    RouteDirection,
    NetOrderer,
    route_board,
)

# Phase 3B+ - Routing Manager
from .manager import (
    RoutingManager,
    RoutingManagerConfig,
    RoutingManagerResult,
    RoutingPhase,
    NetPriority,
    DiffPair,
    route_board_managed,
)

# Phase 2 - Critical Nets (Diff Pairs)
from .diff_pairs import (
    DiffPairDetector,
    DiffPairRouter,
    DiffPairSpec,
    DiffPairResult,
    DiffPairPattern,
    DiffPairGeometry,
    LengthMatcher,
)

# Phase 3D - BGA Fanout
from .fanout import (
    FanoutGenerator,
    FanoutResult,
    FanoutStrategy,
    DogbonePattern,
    VIPPattern,
    FanoutVia,
    FanoutTrace,
    LayerAssigner,
    EscapeRouter,
)

__all__ = [
    # Pin Optimization
    "PinSwapper",
    "SwapResult",
    "SwapConfig",
    "SwapGroupDetector",
    "SwapGroup",
    "SwapGroupType",
    "SwappablePin",
    "CrossingCounter",
    "CrossingResult",
    "RatsnestEdge",
    "BipartiteMatcher",
    "MatchingResult",
    "SwapAssignment",
    "ConstraintGenerator",
    "ConstraintFormat",
    "ConstraintUpdate",
    # Spatial indexing
    "SpatialHashIndex",
    "Obstacle",
    "auto_calibrate_cell_size",
    # Obstacle map
    "ObstacleMapBuilder",
    "NetPads",
    "build_obstacle_map",
    # Visualization
    "RouteVisualizer",
    "VisualizationFrame",
    "RouteSegment",
    "Via",
    "create_visualizer_from_board",
    # A* Router
    "AStarRouter",
    "RouterConfig",
    "RoutingResult",
    "RouteNode",
    "RouteDirection",
    "NetOrderer",
    "route_board",
    # Routing Manager
    "RoutingManager",
    "RoutingManagerConfig",
    "RoutingManagerResult",
    "RoutingPhase",
    "NetPriority",
    "DiffPair",
    "route_board_managed",
    # Differential Pairs
    "DiffPairDetector",
    "DiffPairRouter",
    "DiffPairSpec",
    "DiffPairResult",
    "DiffPairPattern",
    "DiffPairGeometry",
    "LengthMatcher",
    # BGA Fanout
    "FanoutGenerator",
    "FanoutResult",
    "FanoutStrategy",
    "DogbonePattern",
    "VIPPattern",
    "FanoutVia",
    "FanoutTrace",
    "LayerAssigner",
    "EscapeRouter",
    # Planned (lazy import)
    "FreeroutingRunner",
    "NetClassAssigner",
]


def __getattr__(name):
    """Lazy import for planned components."""
    if name == "FreeroutingRunner":
        try:
            from .freerouting import FreeroutingRunner
            return FreeroutingRunner
        except ImportError:
            raise ImportError(
                "FreeroutingRunner not yet implemented. "
                "See research/routing_implementation_plan.md for implementation plan."
            )
    elif name == "NetClassAssigner":
        try:
            from .net_classes import NetClassAssigner
            return NetClassAssigner
        except ImportError:
            raise ImportError(
                "NetClassAssigner not yet implemented. "
                "See research/routing_implementation_plan.md for implementation plan."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
