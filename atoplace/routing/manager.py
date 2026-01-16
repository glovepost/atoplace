"""
Routing Manager

Orchestrates the multi-phase routing pipeline:
1. Fanout Generation (BGA/High-density)
2. Critical Net Routing (Diff pairs, clocks)
3. General Net Routing (A*)
4. Post-processing (Smoothing, Teardrops - planned)

This manager ensures that critical constraints are met before bulk routing begins.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..board.abstraction import Board, Net
from ..dfm.profiles import DFMProfile
from .astar_router import AStarRouter, RouterConfig, RoutingResult, NetOrderer
from .fanout import FanoutGenerator, FanoutStrategy, FanoutResult
from .obstacle_map import ObstacleMapBuilder
from .visualizer import RouteVisualizer

logger = logging.getLogger(__name__)


class RoutingPhase(Enum):
    """Routing phases."""
    FANOUT = "fanout"
    CRITICAL = "critical"
    GENERAL = "general"
    CLEANUP = "cleanup"


class NetPriority(Enum):
    """Net routing priority."""
    CRITICAL = 100  # Diff pairs, clocks
    POWER = 80      # Power/Ground
    SIGNAL = 50     # Standard signals
    LOW = 10        # Non-critical


@dataclass
class DiffPair:
    """Differential pair definition."""
    net_p: str
    net_n: str
    gap: float = 0.15
    width: float = 0.2
    
    @property
    def name(self) -> str:
        return f"DIFF_{self.net_p}_{self.net_n}"


@dataclass
class RoutingManagerConfig:
    """Configuration for the routing manager."""
    # Fanout
    enable_fanout: bool = True
    fanout_strategy: FanoutStrategy = FanoutStrategy.AUTO
    
    # Critical nets
    enable_diff_pairs: bool = True
    diff_pairs: List[DiffPair] = field(default_factory=list)
    
    # General routing
    greedy_weight: float = 2.0
    grid_size: float = 0.1
    max_iterations: int = 50000
    
    # Visualization
    visualize: bool = False
    animate: bool = False


@dataclass
class RoutingManagerResult:
    """Aggregate result of the routing process."""
    success: bool
    results: Dict[str, RoutingResult] = field(default_factory=dict)
    fanout_results: Dict[str, FanoutResult] = field(default_factory=dict)
    unrouted_nets: List[str] = field(default_factory=list)
    stats: Dict[str, any] = field(default_factory=dict)


class RoutingManager:
    """
    Orchestrates the complete routing pipeline.
    """

    def __init__(
        self,
        board: Board,
        dfm_profile: DFMProfile,
        config: Optional[RoutingManagerConfig] = None,
        visualizer: Optional[RouteVisualizer] = None
    ):
        self.board = board
        self.dfm = dfm_profile
        self.config = config or RoutingManagerConfig()
        self.viz = visualizer
        
        # Initialize sub-components
        self.fanout_gen = FanoutGenerator(board, dfm_profile)
        self.obstacle_builder = ObstacleMapBuilder(board, dfm_profile)
        
        # State
        self.obstacle_index = None
        self.nets_to_route: Set[str] = set()
        self.routed_nets: Set[str] = set()
        self.results = RoutingManagerResult(success=True)

    def run(self) -> RoutingManagerResult:
        """Execute the full routing pipeline."""
        logger.info("Starting routing pipeline...")
        
        # 1. Initialization
        self._identify_nets()
        
        # 2. Phase 1: Fanout
        if self.config.enable_fanout:
            self._run_fanout_phase()
            
        # 3. Build initial obstacle map (including fanout)
        # Note: We rebuild here to include fanout traces/vias as obstacles
        # Ideally, we'd add incrementally, but full rebuild is safer for consistency
        self.obstacle_index = self.obstacle_builder.build(include_component_bodies=True)
        
        # Add fanout results to obstacle index manually since they aren't on the board yet
        # (unless we applied them, which we should check)
        self._add_fanout_obstacles()
        
        # 4. Phase 2: Critical Nets
        if self.config.enable_diff_pairs:
            self._run_critical_phase()
            
        # 5. Phase 3: General Routing
        self._run_general_phase()
        
        # 6. Finalize
        self._finalize_results()
        
        return self.results

    def _identify_nets(self):
        """Identify all nets requiring routing."""
        for net_name, net in self.board.nets.items():
            if len(net.connections) >= 2:
                self.nets_to_route.add(net_name)
        
        logger.info(f"Identified {len(self.nets_to_route)} nets to route")

    def _run_fanout_phase(self):
        """Run BGA/FPGA fanout generation."""
        logger.info("Phase 1: Fanout Generation")
        
        # Detect BGAs
        bga_refs = self.fanout_gen.detect_bgas()
        
        for ref in bga_refs:
            result = self.fanout_gen.fanout_component(
                ref, 
                strategy=self.config.fanout_strategy,
                include_escape=True
            )
            self.results.fanout_results[ref] = result
            
            if result.success:
                logger.info(f"Fanout successful for {ref}")
            else:
                logger.warning(f"Fanout failed for {ref}: {result.failure_reason}")

    def _add_fanout_obstacles(self):
        """Add fanout traces/vias to the obstacle index."""
        if not self.obstacle_index:
            return
            
        from .spatial_index import Obstacle
        
        for result in self.results.fanout_results.values():
            if not result.success:
                continue
                
            # Add traces
            for trace in result.traces:
                min_x = min(trace.start[0], trace.end[0]) - trace.width/2
                max_x = max(trace.start[0], trace.end[0]) + trace.width/2
                min_y = min(trace.start[1], trace.end[1]) - trace.width/2
                max_y = max(trace.start[1], trace.end[1]) + trace.width/2
                
                self.obstacle_index.add(Obstacle(
                    min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
                    layer=trace.layer,
                    clearance=0, # Router handles clearance
                    net_id=None, # Block all for now (simplification)
                    obstacle_type="fanout_trace"
                ))
                
            # Add vias
            for via in result.vias:
                r = via.pad_diameter / 2
                self.obstacle_index.add(Obstacle(
                    min_x=via.x - r, min_y=via.y - r,
                    max_x=via.x + r, max_y=via.y + r,
                    layer=-1, # All layers
                    clearance=0,
                    net_id=None,
                    obstacle_type="fanout_via"
                ))

    def _run_critical_phase(self):
        """Run routing for critical nets (diff pairs)."""
        logger.info("Phase 2: Critical Net Routing")
        
        # Identify diff pairs if not provided
        if not self.config.diff_pairs:
            from .diff_pairs import DiffPairDetector
            detector = DiffPairDetector(self.board)
            self.config.diff_pairs = detector.detect()
            
        if not self.config.diff_pairs:
            return

        from .diff_pairs import DiffPairRouter
        dp_router = DiffPairRouter(
            self.obstacle_index, 
            self.dfm,
            self.viz
        )
        
        for dp in self.config.diff_pairs:
            logger.info(f"Routing diff pair: {dp.net_p} / {dp.net_n}")
            
            # Get start/end pads for both nets
            pads_p = self._get_net_pads(dp.net_p)
            pads_n = self._get_net_pads(dp.net_n)
            
            if not pads_p or not pads_n:
                logger.warning(f"Skipping diff pair {dp.name}: missing pads")
                continue
                
            # TODO: Call dp_router.route_pair()
            # For now, just mark as unrouted so general router picks them up
            # (until DiffPairRouter is fully implemented)
            pass

    def _run_general_phase(self):
        """Run A* routing for all remaining nets."""
        logger.info("Phase 3: General Routing")
        
        # Configure router
        router_config = RouterConfig(
            greedy_weight=self.config.greedy_weight,
            grid_size=self.config.grid_size,
            max_iterations=self.config.max_iterations,
            layer_count=self.board.layer_count,
            trace_width=self.dfm.min_trace_width,
            clearance=self.dfm.min_spacing
        )
        
        router = AStarRouter(
            self.obstacle_index,
            router_config,
            self.viz
        )
        
        # Order nets
        # Only include nets not yet routed
        remaining_nets = []
        # Re-fetch net objects for orderer
        net_pads_list = self.obstacle_builder.get_net_pads()
        for np in net_pads_list:
            if np.net_name not in self.routed_nets and np.net_name in self.nets_to_route:
                remaining_nets.append(np)
                
        orderer = NetOrderer(self.obstacle_index)
        ordered_nets = orderer.order_nets(remaining_nets)
        
        # Route
        for net in ordered_nets:
            # Check if net is still unrouted (might have been routed as part of a group)
            if net.net_name in self.routed_nets:
                continue
                
            logger.debug(f"Routing {net.net_name}...")
            
            # Determine specific rules for this net
            board_net = self.board.nets.get(net.net_name)
            width = router_config.trace_width
            clearance = router_config.clearance
            
            if board_net:
                if board_net.trace_width: width = board_net.trace_width
                if board_net.clearance: clearance = board_net.clearance
            
            result = router.route_net(
                net.pads,
                net_name=net.net_name,
                net_id=net.net_id,
                trace_width=width,
                clearance=clearance
            )
            
            self.results.results[net.net_name] = result
            
            if result.success:
                self.routed_nets.add(net.net_name)
                # Add to obstacles for subsequent nets
                for seg in result.segments:
                    seg.net_id = net.net_id
                    seg.net_name = net.net_name  # Ensure net_name is set (Issue #31)
                    router.add_routed_trace(seg)
                for via in result.vias:
                    via.net_id = net.net_id
                    via.net_name = net.net_name  # Ensure net_name is set (Issue #31)
                    router.add_routed_via(via)
            else:
                logger.warning(f"Failed to route {net.net_name}")

    def _get_net_pads(self, net_name: str):
        """Helper to get pads for a net."""
        # This would normally query the obstacle map or board
        # Placeholder for DiffPair logic
        return []

    def _finalize_results(self):
        """Compile final statistics."""
        self.results.unrouted_nets = list(self.nets_to_route - self.routed_nets)
        self.results.success = len(self.results.unrouted_nets) == 0
        
        self.results.stats = {
            "total_nets": len(self.nets_to_route),
            "routed_nets": len(self.routed_nets),
            "fanout_count": len(self.results.fanout_results),
            "completion_rate": len(self.routed_nets) / max(1, len(self.nets_to_route))
        }
        
        logger.info(
            f"Routing complete. "
            f"Success: {self.results.success}. "
            f"Routed: {len(self.routed_nets)}/{len(self.nets_to_route)}"
        )


def route_board_managed(
    board: Board,
    dfm_profile: DFMProfile,
    config: Optional[RoutingManagerConfig] = None
) -> RoutingManagerResult:
    """
    Convenience entry point for managed routing.
    
    Args:
        board: Board to route
        dfm_profile: DFM rules
        config: Routing configuration
        
    Returns:
        RoutingManagerResult
    """
    manager = RoutingManager(board, dfm_profile, config)
    return manager.run()