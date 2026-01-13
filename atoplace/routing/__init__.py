"""Routing engine for AtoPlace.

Phase 3A - Foundation components:
- SpatialHashIndex: O(~1) collision detection using spatial hashing
- ObstacleMapBuilder: Pre-compute all routing obstacles from board
- RouteVisualizer: SVG/HTML visualization for debugging

Phase 3B - Core router (implemented):
- AStarRouter: A* pathfinder with greedy multiplier
- NetOrderer: Route difficult nets first
- route_board: Convenience function to route all nets

Phase 3C - Integration (planned):
- FreeroutingRunner: Fallback to Freerouting for complex boards
"""

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

__all__ = [
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
    # Planned (lazy import)
    "FreeroutingRunner",
    "NetClassAssigner",
    "DiffPairDetector",
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
    elif name == "DiffPairDetector":
        try:
            from .diff_pairs import DiffPairDetector
            return DiffPairDetector
        except ImportError:
            raise ImportError(
                "DiffPairDetector not yet implemented. "
                "See research/routing_implementation_plan.md for implementation plan."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
