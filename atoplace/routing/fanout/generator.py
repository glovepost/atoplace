"""Main entry point for BGA/FPGA fanout generation.

The FanoutGenerator orchestrates the fanout process:
1. Detect BGA components on the board
2. Analyze pin grid and determine pitch
3. Apply appropriate fanout pattern (dogbone/VIP)
4. Assign escape layers using onion model
5. Route escape traces to clear routing space
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ...board.abstraction import Board, Component, Pad

from .patterns import (
    DogbonePattern,
    VIPPattern,
    ChannelPattern,
    FanoutVia,
    FanoutTrace,
)
from .layer_assigner import LayerAssigner, LayerMapping, PinRing
from .escape_router import EscapeRouter, EscapeResult

logger = logging.getLogger(__name__)


class FanoutStrategy(Enum):
    """Fanout pattern strategy."""
    AUTO = "auto"          # Auto-detect based on pitch
    DOGBONE = "dogbone"    # Standard dogbone for pitch >= 0.65mm
    VIP = "vip"            # Via-in-Pad for pitch <= 0.5mm
    CHANNEL = "channel"    # Channel routing for outer rings


@dataclass
class FanoutResult:
    """Result of fanout generation for a component.

    Attributes:
        success: Whether fanout generation succeeded
        component_ref: Reference designator of the component
        strategy_used: Fanout strategy that was used
        vias: List of generated vias
        traces: List of generated traces (pad-to-via and escape)
        escape_results: Dict of pad number to escape routing results
        layer_mapping: Layer assignment for each pad
        pitch_detected: Detected BGA pitch
        ring_count: Number of pin rings detected
        stats: Summary statistics
        warnings: List of warnings during generation
        failure_reason: If failed, reason for failure
    """
    success: bool
    component_ref: str = ""
    strategy_used: FanoutStrategy = FanoutStrategy.AUTO
    vias: List[FanoutVia] = field(default_factory=list)
    traces: List[FanoutTrace] = field(default_factory=list)
    escape_results: Dict[str, EscapeResult] = field(default_factory=dict)
    layer_mapping: Optional[LayerMapping] = None
    pitch_detected: float = 0.0
    ring_count: int = 0
    stats: Dict[str, any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    failure_reason: str = ""


class FanoutGenerator:
    """Main class for generating BGA fanout patterns.

    Coordinates pattern generation, layer assignment, and escape routing
    to produce complete fanout for high-density BGA packages.

    Usage:
        generator = FanoutGenerator(board, dfm_profile)
        result = generator.fanout_component("U1")
        # or
        results = generator.fanout_all_bgas()
    """

    # Pitch thresholds for strategy selection
    VIP_THRESHOLD = 0.5      # Use VIP for pitch <= 0.5mm
    DOGBONE_THRESHOLD = 0.65  # Use dogbone for pitch >= 0.65mm

    # BGA detection parameters
    MIN_BGA_PINS = 9         # Minimum pins to be considered a BGA
    BGA_FOOTPRINT_PATTERNS = [
        "bga", "lga", "qfn", "fpga", "csp",
        "ball", "land", "grid", "array"
    ]

    def __init__(
        self,
        board: Board,
        dfm_profile=None,
        layer_count: Optional[int] = None,
    ):
        """
        Args:
            board: Board instance to generate fanout for
            dfm_profile: DFM profile for trace/via parameters
            layer_count: Override for number of routing layers
        """
        self.board = board
        self.dfm = dfm_profile
        self.layer_count = layer_count or board.layer_count

        # Set DFM parameters
        if dfm_profile:
            self.via_drill = getattr(dfm_profile, 'min_via_drill', 0.3)
            self.via_pad = getattr(dfm_profile, 'min_via_diameter', 0.6)
            self.trace_width = getattr(dfm_profile, 'min_trace_width', 0.2)
            self.clearance = getattr(dfm_profile, 'min_spacing', 0.15)
        else:
            self.via_drill = 0.3
            self.via_pad = 0.6
            self.trace_width = 0.2
            self.clearance = 0.15

        # Create pattern generators
        self._dogbone = DogbonePattern(
            via_drill=self.via_drill,
            via_pad=self.via_pad,
            trace_width=self.trace_width,
            clearance=self.clearance,
        )
        self._vip = VIPPattern(
            via_drill=min(self.via_drill, 0.2),  # Smaller for VIP
            via_pad=min(self.via_pad, 0.35),
        )
        self._channel = ChannelPattern(
            trace_width=self.trace_width,
            clearance=self.clearance,
        )

        # Create layer assigner
        self._layer_assigner = LayerAssigner(layer_count=self.layer_count)

        # Create escape router
        self._escape_router = EscapeRouter(
            trace_width=self.trace_width,
            clearance=self.clearance,
        )

    def detect_bgas(self) -> List[str]:
        """Detect BGA-like components on the board.

        Identifies components that likely need fanout routing based on:
        - Footprint name patterns (BGA, LGA, QFN, etc.)
        - Number of pads (grid-like arrangements)
        - Pad arrangement (regular grid pattern)

        Returns:
            List of component references that are likely BGAs
        """
        bga_refs = []

        for ref, comp in self.board.components.items():
            if self._is_bga_candidate(comp):
                bga_refs.append(ref)

        logger.info(f"Detected {len(bga_refs)} BGA-like components")
        return bga_refs

    def _is_bga_candidate(self, comp: Component) -> bool:
        """Check if a component is a BGA candidate."""
        # Check footprint name
        footprint_lower = comp.footprint.lower() if comp.footprint else ""
        for pattern in self.BGA_FOOTPRINT_PATTERNS:
            if pattern in footprint_lower:
                return True

        # Check pad count
        if len(comp.pads) < self.MIN_BGA_PINS:
            return False

        # Check for grid-like arrangement
        return self._has_grid_pattern(comp)

    def _has_grid_pattern(self, comp: Component) -> bool:
        """Check if component pads form a grid pattern."""
        if len(comp.pads) < self.MIN_BGA_PINS:
            return False

        # Get unique X and Y coordinates
        x_coords = sorted(set(round(p.x, 3) for p in comp.pads))
        y_coords = sorted(set(round(p.y, 3) for p in comp.pads))

        # Grid should have multiple rows and columns
        if len(x_coords) < 3 or len(y_coords) < 3:
            return False

        # Check for regular pitch in X direction
        if len(x_coords) > 1:
            x_pitches = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            # Pitch should be consistent
            if max(x_pitches) - min(x_pitches) > 0.1:
                return False

        # Check for regular pitch in Y direction
        if len(y_coords) > 1:
            y_pitches = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            if max(y_pitches) - min(y_pitches) > 0.1:
                return False

        # Likely a grid
        return True

    def measure_pitch(self, comp: Component) -> float:
        """Measure the grid pitch of a BGA component.

        Args:
            comp: Component to measure

        Returns:
            Pitch in mm (center-to-center distance between adjacent pads)
        """
        if len(comp.pads) < 2:
            return 0.0

        # Get sorted unique coordinates
        x_coords = sorted(set(round(p.x, 3) for p in comp.pads))
        y_coords = sorted(set(round(p.y, 3) for p in comp.pads))

        # Calculate average pitch
        x_pitch = 0.0
        if len(x_coords) > 1:
            x_pitches = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            x_pitch = sum(x_pitches) / len(x_pitches)

        y_pitch = 0.0
        if len(y_coords) > 1:
            y_pitches = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            y_pitch = sum(y_pitches) / len(y_pitches)

        # Use the smaller non-zero pitch
        if x_pitch > 0 and y_pitch > 0:
            return min(x_pitch, y_pitch)
        return max(x_pitch, y_pitch)

    def select_strategy(self, pitch: float) -> FanoutStrategy:
        """Select fanout strategy based on pitch.

        Args:
            pitch: BGA grid pitch in mm

        Returns:
            Appropriate FanoutStrategy
        """
        if pitch <= self.VIP_THRESHOLD:
            return FanoutStrategy.VIP
        else:
            return FanoutStrategy.DOGBONE

    def fanout_component(
        self,
        ref: str,
        strategy: FanoutStrategy = FanoutStrategy.AUTO,
        include_escape: bool = True,
    ) -> FanoutResult:
        """Generate fanout for a specific component.

        Args:
            ref: Component reference (e.g., "U1")
            strategy: Fanout strategy to use (AUTO detects from pitch)
            include_escape: Whether to include escape routing

        Returns:
            FanoutResult with generated vias and traces
        """
        comp = self.board.components.get(ref)
        if comp is None:
            return FanoutResult(
                success=False,
                component_ref=ref,
                failure_reason=f"Component {ref} not found"
            )

        if len(comp.pads) < 2:
            return FanoutResult(
                success=False,
                component_ref=ref,
                failure_reason=f"Component {ref} has too few pads ({len(comp.pads)})"
            )

        # Measure pitch
        pitch = self.measure_pitch(comp)
        if pitch <= 0:
            return FanoutResult(
                success=False,
                component_ref=ref,
                failure_reason="Could not determine pad pitch"
            )

        # Select strategy
        if strategy == FanoutStrategy.AUTO:
            strategy = self.select_strategy(pitch)

        logger.info(
            f"Generating fanout for {ref}: pitch={pitch:.3f}mm, "
            f"strategy={strategy.value}, pads={len(comp.pads)}"
        )

        # Get absolute pad positions
        pads_data = []
        for pad in comp.pads:
            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            pads_data.append((pad.number, abs_x, abs_y))

        # Analyze rings
        rings = self._layer_assigner.analyze_rings(pads_data, pitch)

        # Assign layers
        layer_mapping = self._layer_assigner.assign_layers(rings, strategy="balanced")

        # Generate fanout pattern
        vias = []
        traces = []
        warnings = []

        for pad in comp.pads:
            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            escape_layer = layer_mapping.get_escape_layer(pad.number) or 0

            # Check if outer ring can use channel routing
            ring_idx = self._get_pad_ring(pad.number, rings)
            if ring_idx == 0 and self._channel.can_channel_route(
                abs_x, abs_y, pitch, ring_idx, len(rings)
            ):
                # Channel route - no via needed for outermost ring
                channel_traces = self._channel.generate_escape_path(
                    abs_x, abs_y, comp.x, comp.y,
                    escape_distance=pitch * (len(rings) + 1),
                    layer=0,  # Top layer
                    net_name=pad.net,
                )
                traces.extend(channel_traces)
                continue

            if strategy == FanoutStrategy.VIP:
                # Via-in-Pad
                via = self._vip.generate(
                    pad_x=abs_x,
                    pad_y=abs_y,
                    layer=0,
                    target_layer=escape_layer,
                    net_name=pad.net,
                    pad_number=pad.number,
                )
                vias.append(via)
            else:
                # Dogbone
                pad_size = max(pad.width, pad.height)
                via, trace = self._dogbone.generate(
                    pad_x=abs_x,
                    pad_y=abs_y,
                    pad_size=pad_size,
                    pad_pitch=pitch,
                    component_center_x=comp.x,
                    component_center_y=comp.y,
                    layer=0,
                    net_name=pad.net,
                    pad_number=pad.number,
                )
                # Update via target layer based on ring
                via.end_layer = escape_layer
                vias.append(via)
                traces.append(trace)

        # Escape routing
        escape_results = {}
        if include_escape and vias:
            # Get courtyard bounds
            courtyard = self._get_component_courtyard(comp)

            # Build existing obstacle list
            existing_obstacles = self._get_pad_obstacles(comp)

            escape_results = self._escape_router.route_escapes(
                vias=vias,
                component_center=(comp.x, comp.y),
                courtyard_bounds=courtyard,
                existing_obstacles=existing_obstacles,
            )

            # Add escape traces
            for pad_num, escape_result in escape_results.items():
                if escape_result.success:
                    traces.extend(escape_result.traces)
                else:
                    warnings.append(
                        f"Escape failed for pad {pad_num}: {escape_result.failure_reason}"
                    )

        # Calculate statistics
        escape_success = sum(1 for r in escape_results.values() if r.success)
        escape_total = len(escape_results)
        total_via_count = len(vias)
        total_trace_length = sum(
            math.sqrt(
                (t.end[0] - t.start[0])**2 + (t.end[1] - t.start[1])**2
            )
            for t in traces
        )

        stats = {
            "pin_count": len(comp.pads),
            "ring_count": len(rings),
            "via_count": total_via_count,
            "trace_count": len(traces),
            "total_trace_length_mm": round(total_trace_length, 2),
            "escape_success_rate": escape_success / escape_total if escape_total > 0 else 1.0,
            "layers_used": sorted(layer_mapping.layer_to_pads.keys()) if layer_mapping else [],
        }

        return FanoutResult(
            success=True,
            component_ref=ref,
            strategy_used=strategy,
            vias=vias,
            traces=traces,
            escape_results=escape_results,
            layer_mapping=layer_mapping,
            pitch_detected=pitch,
            ring_count=len(rings),
            stats=stats,
            warnings=warnings,
        )

    def fanout_all_bgas(
        self,
        strategy: FanoutStrategy = FanoutStrategy.AUTO,
        include_escape: bool = True,
    ) -> Dict[str, FanoutResult]:
        """Generate fanout for all BGA components on the board.

        Args:
            strategy: Fanout strategy to use (AUTO detects from pitch)
            include_escape: Whether to include escape routing

        Returns:
            Dict mapping component ref to FanoutResult
        """
        bga_refs = self.detect_bgas()
        results = {}

        for ref in bga_refs:
            result = self.fanout_component(ref, strategy, include_escape)
            results[ref] = result

            if result.success:
                logger.info(
                    f"Fanout {ref}: {result.stats['via_count']} vias, "
                    f"{result.stats['trace_count']} traces"
                )
            else:
                logger.warning(f"Fanout failed for {ref}: {result.failure_reason}")

        return results

    def _get_pad_ring(self, pad_number: str, rings: List[PinRing]) -> int:
        """Get the ring index for a pad."""
        for ring in rings:
            if pad_number in ring.pad_numbers:
                return ring.index
        return -1

    def _get_component_courtyard(
        self, comp: Component
    ) -> Tuple[float, float, float, float]:
        """Get component courtyard bounds."""
        bbox = comp.get_bounding_box_with_pads()
        # Add some margin
        margin = 0.5
        return (
            bbox[0] - margin,
            bbox[1] - margin,
            bbox[2] + margin,
            bbox[3] + margin,
        )

    def _get_pad_obstacles(
        self, comp: Component
    ) -> List[Tuple[float, float, float, int]]:
        """Get pad positions as obstacles for escape routing."""
        obstacles = []
        for pad in comp.pads:
            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            radius = max(pad.width, pad.height) / 2
            layer = 0 if pad.layer.value.startswith("F.") else 1
            obstacles.append((abs_x, abs_y, radius, layer))
        return obstacles


def auto_fanout_component(
    board: Board,
    ref: str,
    dfm_profile=None,
) -> FanoutResult:
    """Convenience function to fanout a single component.

    Args:
        board: Board instance
        ref: Component reference
        dfm_profile: Optional DFM profile

    Returns:
        FanoutResult
    """
    generator = FanoutGenerator(board, dfm_profile)
    return generator.fanout_component(ref)


def auto_fanout_all(
    board: Board,
    dfm_profile=None,
) -> Dict[str, FanoutResult]:
    """Convenience function to fanout all BGAs on a board.

    Args:
        board: Board instance
        dfm_profile: Optional DFM profile

    Returns:
        Dict of ref -> FanoutResult
    """
    generator = FanoutGenerator(board, dfm_profile)
    return generator.fanout_all_bgas()
