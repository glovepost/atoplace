"""Routing Manager - orchestrates the multi-phase routing pipeline.

The RoutingManager coordinates the complete routing flow:
1. Phase 0: Pin Optimization (optional)
2. Phase 1: Fanout & Escape (for BGAs/QFNs)
3. Phase 2: Critical Nets (diff pairs, length-matched)
4. Phase 3: General Routing (A* with greedy multiplier)
5. Phase 4: Post-processing (teardrops, smoothing)

This follows the Unified Routing Strategy defined in docs/specs/ROUTING_STRATEGY.md.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from enum import Enum

from ..board import Board
from ..dfm.profiles import DFMProfile, get_profile
from .spatial_index import SpatialHashIndex, Obstacle
from .obstacle_map import ObstacleMapBuilder, NetPads
from .astar_router import AStarRouter, RouterConfig, RoutingResult, NetOrderer
from .visualizer import RouteVisualizer, RouteSegment, Via, create_visualizer_from_board

logger = logging.getLogger(__name__)


class RoutingPhase(Enum):
    """Phases of the routing pipeline."""
    PIN_OPTIMIZATION = "pin_optimization"
    FANOUT_ESCAPE = "fanout_escape"
    CRITICAL_NETS = "critical_nets"
    GENERAL_ROUTING = "general_routing"
    POST_PROCESSING = "post_processing"


class NetPriority(Enum):
    """Net priority classification."""
    CRITICAL = "critical"  # Diff pairs, clocks, high-speed
    POWER = "power"        # Power rails
    GROUND = "ground"      # Ground nets
    SIGNAL = "signal"      # Regular signal nets


@dataclass
class DiffPair:
    """A differential pair definition."""
    name: str
    positive_net: str
    negative_net: str
    impedance: float = 100.0  # Ohms
    max_skew: float = 0.1     # mm
    spacing: float = 0.15     # mm between traces


@dataclass
class RoutingManagerConfig:
    """Configuration for the routing manager."""
    # Phase enables
    enable_pin_optimization: bool = False  # Not yet implemented
    enable_fanout: bool = True
    enable_critical_nets: bool = True
    enable_general_routing: bool = True
    enable_post_processing: bool = False   # Not yet implemented

    # Router config
    router_config: RouterConfig = field(default_factory=RouterConfig)

    # Fanout config
    fanout_strategy: str = "auto"  # auto, dogbone, vip

    # Visualization
    visualize: bool = False
    output_dir: str = "."


@dataclass
class RoutingManagerResult:
    """Result of the full routing pipeline."""
    success: bool
    phases_completed: List[RoutingPhase]
    net_results: Dict[str, RoutingResult]

    # Statistics
    total_nets: int = 0
    routed_nets: int = 0
    failed_nets: int = 0
    total_length: float = 0.0
    total_vias: int = 0

    # Phase-specific results
    fanout_results: Optional[Dict] = None
    diff_pair_results: Optional[Dict] = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        """Percentage of nets successfully routed."""
        if self.total_nets == 0:
            return 100.0
        return (self.routed_nets / self.total_nets) * 100


class RoutingManager:
    """Orchestrates the multi-phase routing pipeline.

    The RoutingManager is the primary entry point for routing a PCB.
    It coordinates:
    - Obstacle map building
    - Net classification and ordering
    - Phase-by-phase routing execution
    - Result aggregation and reporting

    Example:
        >>> manager = RoutingManager(board, dfm_profile)
        >>> manager.add_diff_pair("USB", "USB_D+", "USB_D-", impedance=90)
        >>> result = manager.route_all()
        >>> print(f"Routed {result.completion_rate:.1f}% of nets")
    """

    def __init__(
        self,
        board: Board,
        dfm_profile: Optional[DFMProfile] = None,
        config: Optional[RoutingManagerConfig] = None
    ):
        """
        Initialize the routing manager.

        Args:
            board: Board to route
            dfm_profile: DFM profile for design rules (uses JLCPCB if not specified)
            config: Manager configuration
        """
        self.board = board
        self.dfm = dfm_profile or get_profile("jlcpcb_standard")
        self.config = config or RoutingManagerConfig()

        # Initialize router config with DFM rules
        if self.config.router_config.trace_width == 0.2:  # Default value
            self.config.router_config.trace_width = self.dfm.min_trace_width
        if self.config.router_config.clearance == 0.15:  # Default value
            self.config.router_config.clearance = self.dfm.min_spacing
        self.config.router_config.layer_count = board.layer_count

        # Net classification
        self._diff_pairs: List[DiffPair] = []
        self._critical_nets: Set[str] = set()
        self._power_nets: Set[str] = set()
        self._ground_nets: Set[str] = set()

        # State
        self._obstacle_index: Optional[SpatialHashIndex] = None
        self._net_pads: Optional[List[NetPads]] = None
        self._router: Optional[AStarRouter] = None
        self._visualizer: Optional[RouteVisualizer] = None

        # Results
        self._results: Dict[str, RoutingResult] = {}
        self._all_segments: List[RouteSegment] = []
        self._all_vias: List[Via] = []

        # Progress callback
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def add_diff_pair(
        self,
        name: str,
        positive_net: str,
        negative_net: str,
        impedance: float = 100.0,
        max_skew: float = 0.1,
        spacing: float = 0.15
    ) -> "RoutingManager":
        """
        Register a differential pair for routing.

        Args:
            name: Name of the diff pair (e.g., "USB", "HDMI_0")
            positive_net: Name of the positive signal net
            negative_net: Name of the negative signal net
            impedance: Target differential impedance in Ohms
            max_skew: Maximum allowed length difference in mm
            spacing: Gap between the traces in mm

        Returns:
            self for method chaining
        """
        pair = DiffPair(
            name=name,
            positive_net=positive_net,
            negative_net=negative_net,
            impedance=impedance,
            max_skew=max_skew,
            spacing=spacing
        )
        self._diff_pairs.append(pair)
        self._critical_nets.add(positive_net)
        self._critical_nets.add(negative_net)
        logger.info(f"Registered diff pair '{name}': {positive_net} / {negative_net}")
        return self

    def set_critical_nets(self, net_names: List[str]) -> "RoutingManager":
        """Mark nets as critical (routed first with higher priority)."""
        self._critical_nets.update(net_names)
        return self

    def set_power_nets(self, net_names: List[str]) -> "RoutingManager":
        """Mark nets as power nets."""
        self._power_nets.update(net_names)
        return self

    def set_ground_nets(self, net_names: List[str]) -> "RoutingManager":
        """Mark nets as ground nets."""
        self._ground_nets.update(net_names)
        return self

    def set_progress_callback(
        self,
        callback: Callable[[str, float], None]
    ) -> "RoutingManager":
        """Set callback for progress updates: callback(phase_name, progress_0_to_1)."""
        self._progress_callback = callback
        return self

    def _report_progress(self, phase: str, progress: float):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(phase, progress)

    def _build_obstacle_map(self):
        """Build the obstacle map from the board."""
        logger.info("Building obstacle map...")
        builder = ObstacleMapBuilder(self.board, self.dfm)
        self._obstacle_index = builder.build()
        self._net_pads = builder.get_net_pads()
        logger.info(f"Built obstacle map with {len(self._net_pads)} nets")

    def _classify_nets(self) -> Dict[NetPriority, List[NetPads]]:
        """Classify nets by priority for routing order."""
        classified: Dict[NetPriority, List[NetPads]] = {
            NetPriority.CRITICAL: [],
            NetPriority.POWER: [],
            NetPriority.GROUND: [],
            NetPriority.SIGNAL: [],
        }

        for net in self._net_pads:
            if net.net_name in self._critical_nets:
                classified[NetPriority.CRITICAL].append(net)
            elif net.net_name in self._power_nets:
                classified[NetPriority.POWER].append(net)
            elif net.net_name in self._ground_nets:
                classified[NetPriority.GROUND].append(net)
            else:
                classified[NetPriority.SIGNAL].append(net)

        return classified

    def _order_nets_for_routing(self) -> List[NetPads]:
        """Order all nets for routing based on priority and difficulty."""
        classified = self._classify_nets()

        # Use NetOrderer to sort within each category
        orderer = NetOrderer(self._obstacle_index)

        ordered = []

        # Critical nets first (already sorted by difficulty)
        critical = orderer.order_nets(classified[NetPriority.CRITICAL])
        ordered.extend(critical)

        # Power and ground next
        power = orderer.order_nets(classified[NetPriority.POWER])
        ground = orderer.order_nets(classified[NetPriority.GROUND])
        ordered.extend(power)
        ordered.extend(ground)

        # Finally, regular signals
        signals = orderer.order_nets(classified[NetPriority.SIGNAL])
        ordered.extend(signals)

        return ordered

    def _run_fanout_phase(self) -> Optional[Dict]:
        """Run Phase 1: Fanout & Escape for BGAs."""
        if not self.config.enable_fanout:
            return None

        try:
            from .fanout import FanoutGenerator, FanoutStrategy

            logger.info("Phase 1: Fanout & Escape")
            self._report_progress("fanout", 0.0)

            # Detect BGA components
            generator = FanoutGenerator(self.board)
            bga_refs = generator.detect_bga_components()

            if not bga_refs:
                logger.info("No BGA components detected, skipping fanout phase")
                return {"skipped": True, "reason": "no_bga_components"}

            logger.info(f"Detected {len(bga_refs)} BGA components: {bga_refs}")

            # Generate fanout for each BGA
            results = {}
            strategy = FanoutStrategy(self.config.fanout_strategy)

            for i, ref in enumerate(bga_refs):
                self._report_progress("fanout", i / len(bga_refs))

                result = generator.generate_fanout(
                    ref,
                    strategy=strategy,
                    include_escape=True
                )
                results[ref] = {
                    "success": result.success,
                    "strategy": result.strategy_used.value,
                    "via_count": len(result.vias),
                    "trace_count": len(result.traces),
                    "warnings": result.warnings
                }

                # Add fanout structures to obstacle index
                if result.success:
                    for via in result.vias:
                        self._obstacle_index.add(Obstacle(
                            min_x=via.x - via.pad_diameter / 2,
                            min_y=via.y - via.pad_diameter / 2,
                            max_x=via.x + via.pad_diameter / 2,
                            max_y=via.y + via.pad_diameter / 2,
                            layer=-1,  # All layers
                            clearance=0,
                            obstacle_type="via"
                        ))

            self._report_progress("fanout", 1.0)
            return results

        except ImportError as e:
            logger.warning(f"Fanout module not available: {e}")
            return {"skipped": True, "reason": str(e)}

    def _run_critical_nets_phase(self) -> Optional[Dict]:
        """Run Phase 2: Critical Nets (diff pairs, etc.)."""
        if not self.config.enable_critical_nets:
            return None

        if not self._diff_pairs:
            logger.info("No diff pairs registered, skipping critical nets phase")
            return {"skipped": True, "reason": "no_diff_pairs"}

        logger.info("Phase 2: Critical Nets")
        self._report_progress("critical_nets", 0.0)

        results = {}

        # Route diff pairs first
        for i, pair in enumerate(self._diff_pairs):
            self._report_progress("critical_nets", i / len(self._diff_pairs))

            # Find pads for both nets
            pos_pads = None
            neg_pads = None
            for net in self._net_pads:
                if net.net_name == pair.positive_net:
                    pos_pads = net
                elif net.net_name == pair.negative_net:
                    neg_pads = net

            if not pos_pads or not neg_pads:
                logger.warning(
                    f"Could not find pads for diff pair {pair.name}: "
                    f"pos={pair.positive_net}, neg={pair.negative_net}"
                )
                results[pair.name] = {"success": False, "reason": "pads_not_found"}
                continue

            # For now, route as separate nets (true diff pair routing TBD)
            # TODO: Implement DiffPairRouter for coupled routing
            pos_result = self._router.route_net(
                pads=pos_pads.pads,
                net_name=pos_pads.net_name,
                net_id=pos_pads.net_id
            )

            neg_result = self._router.route_net(
                pads=neg_pads.pads,
                net_name=neg_pads.net_name,
                net_id=neg_pads.net_id
            )

            results[pair.name] = {
                "success": pos_result.success and neg_result.success,
                "positive": {
                    "success": pos_result.success,
                    "length": pos_result.total_length,
                    "vias": pos_result.via_count
                },
                "negative": {
                    "success": neg_result.success,
                    "length": neg_result.total_length,
                    "vias": neg_result.via_count
                },
                "skew": abs(pos_result.total_length - neg_result.total_length)
            }

            # Store results
            self._results[pos_pads.net_name] = pos_result
            self._results[neg_pads.net_name] = neg_result

            # Add routed traces as obstacles
            if pos_result.success:
                for seg in pos_result.segments:
                    self._router.add_routed_trace(seg)
                for via in pos_result.vias:
                    self._router.add_routed_via(via)

            if neg_result.success:
                for seg in neg_result.segments:
                    self._router.add_routed_trace(seg)
                for via in neg_result.vias:
                    self._router.add_routed_via(via)

        self._report_progress("critical_nets", 1.0)
        return results

    def _run_general_routing_phase(self) -> Dict:
        """Run Phase 3: General Routing with A*."""
        if not self.config.enable_general_routing:
            return {"skipped": True, "reason": "disabled"}

        logger.info("Phase 3: General Routing")
        self._report_progress("general_routing", 0.0)

        # Get ordered nets (excluding already-routed critical nets)
        ordered = self._order_nets_for_routing()
        nets_to_route = [
            n for n in ordered
            if n.net_name not in self._results
        ]

        logger.info(f"Routing {len(nets_to_route)} general nets")

        success_count = 0
        for i, net in enumerate(nets_to_route):
            self._report_progress("general_routing", i / len(nets_to_route))

            # Get net-specific rules
            board_net = self.board.nets.get(net.net_name)
            trace_width = self.config.router_config.trace_width
            clearance = self.config.router_config.clearance

            if board_net:
                if board_net.trace_width:
                    trace_width = board_net.trace_width
                if board_net.clearance:
                    clearance = board_net.clearance

            result = self._router.route_net(
                pads=net.pads,
                net_name=net.net_name,
                net_id=net.net_id,
                trace_width=trace_width,
                clearance=clearance
            )

            self._results[net.net_name] = result

            if result.success:
                success_count += 1
                # Add as obstacles
                for seg in result.segments:
                    self._router.add_routed_trace(seg)
                for via in result.vias:
                    self._router.add_routed_via(via)

                logger.debug(
                    f"Routed {net.net_name}: {len(result.segments)} segs, "
                    f"{result.via_count} vias, {result.total_length:.1f}mm"
                )
            else:
                logger.warning(f"Failed to route {net.net_name}: {result.failure_reason}")

        self._report_progress("general_routing", 1.0)

        return {
            "nets_attempted": len(nets_to_route),
            "nets_routed": success_count,
            "nets_failed": len(nets_to_route) - success_count
        }

    def route_all(self) -> RoutingManagerResult:
        """
        Execute the complete routing pipeline.

        Returns:
            RoutingManagerResult with statistics and per-net results
        """
        phases_completed = []
        errors = []
        warnings = []

        # Build obstacle map
        self._build_obstacle_map()

        # Create visualizer if enabled
        if self.config.visualize:
            self._visualizer = create_visualizer_from_board(self.board)

        # Create router
        self._router = AStarRouter(
            self._obstacle_index,
            self.config.router_config,
            self._visualizer
        )

        # Phase 1: Fanout
        fanout_results = None
        try:
            fanout_results = self._run_fanout_phase()
            if fanout_results and not fanout_results.get("skipped"):
                phases_completed.append(RoutingPhase.FANOUT_ESCAPE)
        except Exception as e:
            logger.error(f"Fanout phase failed: {e}")
            errors.append(f"Fanout phase: {e}")

        # Phase 2: Critical Nets
        diff_pair_results = None
        try:
            diff_pair_results = self._run_critical_nets_phase()
            if diff_pair_results and not diff_pair_results.get("skipped"):
                phases_completed.append(RoutingPhase.CRITICAL_NETS)
        except Exception as e:
            logger.error(f"Critical nets phase failed: {e}")
            errors.append(f"Critical nets phase: {e}")

        # Phase 3: General Routing
        general_results = None
        try:
            general_results = self._run_general_routing_phase()
            if general_results and not general_results.get("skipped"):
                phases_completed.append(RoutingPhase.GENERAL_ROUTING)
        except Exception as e:
            logger.error(f"General routing phase failed: {e}")
            errors.append(f"General routing phase: {e}")

        # Export visualization
        if self._visualizer:
            output_path = f"{self.config.output_dir}/routing_result.html"
            self._visualizer.export_html_report(output_path)
            logger.info(f"Exported routing visualization to {output_path}")

        # Calculate statistics
        total_length = 0.0
        total_vias = 0
        routed_count = 0
        failed_count = 0

        for result in self._results.values():
            if result.success:
                routed_count += 1
                total_length += result.total_length
                total_vias += result.via_count
            else:
                failed_count += 1

        return RoutingManagerResult(
            success=failed_count == 0,
            phases_completed=phases_completed,
            net_results=self._results,
            total_nets=len(self._net_pads),
            routed_nets=routed_count,
            failed_nets=failed_count,
            total_length=total_length,
            total_vias=total_vias,
            fanout_results=fanout_results,
            diff_pair_results=diff_pair_results,
            errors=errors,
            warnings=warnings
        )

    def route_net(self, net_name: str) -> RoutingResult:
        """
        Route a single net by name.

        Args:
            net_name: Name of the net to route

        Returns:
            RoutingResult for the net
        """
        # Ensure obstacle map is built
        if not self._obstacle_index:
            self._build_obstacle_map()

        # Create router if needed
        if not self._router:
            self._router = AStarRouter(
                self._obstacle_index,
                self.config.router_config
            )

        # Find the net
        net_pads = None
        for net in self._net_pads:
            if net.net_name == net_name:
                net_pads = net
                break

        if not net_pads:
            return RoutingResult(
                success=False,
                net_name=net_name,
                failure_reason=f"Net '{net_name}' not found"
            )

        # Route it
        result = self._router.route_net(
            pads=net_pads.pads,
            net_name=net_pads.net_name,
            net_id=net_pads.net_id
        )

        # Track result
        self._results[net_name] = result

        # Add as obstacles if successful
        if result.success:
            for seg in result.segments:
                self._router.add_routed_trace(seg)
            for via in result.vias:
                self._router.add_routed_via(via)

        return result

    def get_routed_geometry(self) -> Tuple[List[RouteSegment], List[Via]]:
        """Get all routed trace segments and vias."""
        segments = []
        vias = []

        for result in self._results.values():
            if result.success:
                segments.extend(result.segments)
                vias.extend(result.vias)

        return segments, vias


def route_board_managed(
    board: Board,
    dfm_profile: Optional[DFMProfile] = None,
    diff_pairs: Optional[List[Dict]] = None,
    critical_nets: Optional[List[str]] = None,
    visualize: bool = False
) -> RoutingManagerResult:
    """
    Convenience function for managed routing.

    Args:
        board: Board to route
        dfm_profile: DFM profile (defaults to JLCPCB)
        diff_pairs: List of diff pair specs: [{"name": ..., "pos": ..., "neg": ...}]
        critical_nets: List of critical net names
        visualize: Whether to generate visualization

    Returns:
        RoutingManagerResult
    """
    config = RoutingManagerConfig(visualize=visualize)
    manager = RoutingManager(board, dfm_profile, config)

    # Add diff pairs
    if diff_pairs:
        for dp in diff_pairs:
            manager.add_diff_pair(
                name=dp["name"],
                positive_net=dp.get("pos", dp.get("positive_net")),
                negative_net=dp.get("neg", dp.get("negative_net")),
                impedance=dp.get("impedance", 100.0),
                max_skew=dp.get("max_skew", 0.1),
                spacing=dp.get("spacing", 0.15)
            )

    # Set critical nets
    if critical_nets:
        manager.set_critical_nets(critical_nets)

    return manager.route_all()
