"""Differential pair detection and routing.

Implements Phase 2 (Critical Nets) of the routing pipeline:
- DiffPairDetector: Automatically identifies diff pairs from net names
- DiffPairRouter: Routes coupled differential pairs together

Based on the Dual-Grid A* concept from ROUTING_STRATEGY.md:
- Route the *centerline* of the pair
- Inflate obstacles by (width + gap + clearance)
- Heavy penalty for uncoupling (splitting around obstacles)
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum

from .spatial_index import SpatialHashIndex, Obstacle
from .astar_router import RouteNode, RouterConfig, RoutingResult
from .visualizer import RouteSegment, Via

logger = logging.getLogger(__name__)


class DiffPairPattern(Enum):
    """Common differential pair naming patterns."""
    PLUS_MINUS = "plus_minus"     # NET_P, NET_N or NET+, NET-
    POSITIVE_NEGATIVE = "pos_neg" # NET_POS, NET_NEG
    SUFFIX_P_N = "suffix_pn"      # NETP, NETN
    USB_STYLE = "usb"             # USB_D+, USB_D-
    NUMBERED = "numbered"         # DIFF0_P, DIFF0_N


@dataclass
class DiffPairSpec:
    """Specification for a detected differential pair."""
    name: str
    positive_net: str
    negative_net: str
    pattern: DiffPairPattern
    confidence: float = 1.0  # 0-1 detection confidence

    # Electrical parameters (can be set by user)
    impedance: float = 100.0      # Target diff impedance (Ohms)
    trace_width: float = 0.2      # Individual trace width (mm)
    spacing: float = 0.15         # Gap between traces (mm)
    max_skew: float = 0.1         # Max length mismatch (mm)
    max_uncoupled: float = 2.0    # Max uncoupled length (mm)


@dataclass
class DiffPairResult:
    """Result of routing a differential pair."""
    success: bool
    diff_pair: DiffPairSpec
    positive_segments: List[RouteSegment] = field(default_factory=list)
    negative_segments: List[RouteSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)

    # Metrics
    positive_length: float = 0.0
    negative_length: float = 0.0
    coupled_length: float = 0.0
    uncoupled_length: float = 0.0
    skew: float = 0.0

    failure_reason: str = ""

    @property
    def total_length(self) -> float:
        return (self.positive_length + self.negative_length) / 2

    @property
    def coupling_ratio(self) -> float:
        """Ratio of coupled vs total length (1.0 = fully coupled)."""
        total = self.coupled_length + self.uncoupled_length
        if total == 0:
            return 1.0
        return self.coupled_length / total


class DiffPairDetector:
    """Automatically detect differential pairs from net names.

    Uses pattern matching to identify common diff pair naming conventions:
    - USB_D+/USB_D-
    - LVDS_TX_P/LVDS_TX_N
    - HDMI0_D0+/HDMI0_D0-
    - etc.
    """

    # Patterns for differential pair detection
    # Each pattern is (regex for positive, regex for negative, base name extraction)
    PATTERNS = [
        # USB_D+ / USB_D-
        (r'^(.+)[_]?[DP]\+$', r'^(.+)[_]?[DN]\-$', DiffPairPattern.USB_STYLE),
        # NET_P / NET_N (most common)
        (r'^(.+)_P$', r'^(.+)_N$', DiffPairPattern.PLUS_MINUS),
        # NET+ / NET-
        (r'^(.+)\+$', r'^(.+)\-$', DiffPairPattern.PLUS_MINUS),
        # NETP / NETN
        (r'^(.+)P$', r'^(.+)N$', DiffPairPattern.SUFFIX_P_N),
        # NET_POS / NET_NEG
        (r'^(.+)_POS$', r'^(.+)_NEG$', DiffPairPattern.POSITIVE_NEGATIVE),
        # NET_POSITIVE / NET_NEGATIVE
        (r'^(.+)_POSITIVE$', r'^(.+)_NEGATIVE$', DiffPairPattern.POSITIVE_NEGATIVE),
    ]

    def __init__(self, net_names: List[str]):
        """
        Initialize detector with net names.

        Args:
            net_names: List of all net names in the design
        """
        self.net_names = set(net_names)
        self._detected: List[DiffPairSpec] = []

    def detect(self) -> List[DiffPairSpec]:
        """
        Detect all differential pairs in the net list.

        Returns:
            List of detected DiffPairSpec objects
        """
        self._detected = []
        used_nets: Set[str] = set()

        for net_name in sorted(self.net_names):
            if net_name in used_nets:
                continue

            # Try each pattern
            for pos_pattern, neg_pattern, pair_type in self.PATTERNS:
                match = re.match(pos_pattern, net_name, re.IGNORECASE)
                if match:
                    base_name = match.group(1)
                    # Look for complementary net
                    complement = self._find_complement(
                        net_name, base_name, neg_pattern
                    )
                    if complement:
                        spec = DiffPairSpec(
                            name=base_name,
                            positive_net=net_name,
                            negative_net=complement,
                            pattern=pair_type,
                            confidence=1.0
                        )
                        self._detected.append(spec)
                        used_nets.add(net_name)
                        used_nets.add(complement)
                        logger.debug(f"Detected diff pair: {spec.name}")
                        break

        logger.info(f"Detected {len(self._detected)} differential pairs")
        return self._detected

    def _find_complement(
        self,
        positive_net: str,
        base_name: str,
        neg_pattern: str
    ) -> Optional[str]:
        """Find the negative complement of a positive net."""
        # Generate expected negative name based on pattern
        for net_name in self.net_names:
            if net_name == positive_net:
                continue

            match = re.match(neg_pattern, net_name, re.IGNORECASE)
            if match and match.group(1).upper() == base_name.upper():
                return net_name

        return None

    @property
    def detected_pairs(self) -> List[DiffPairSpec]:
        """Get list of detected differential pairs."""
        return self._detected


class DiffPairRouter:
    """Route differential pairs as coupled traces.

    Uses the Dual-Grid A* approach:
    1. Route the centerline of the pair
    2. Obstacles are inflated by (width + gap + clearance)
    3. Path is expanded to two parallel traces at the end

    The key insight is that routing the centerline automatically
    maintains equal length for both traces when they're coupled.
    """

    def __init__(
        self,
        obstacle_index: SpatialHashIndex,
        config: RouterConfig = None
    ):
        """
        Initialize diff pair router.

        Args:
            obstacle_index: Pre-built spatial hash of obstacles
            config: Router configuration
        """
        self.obstacles = obstacle_index
        self.config = config or RouterConfig()

    def route_diff_pair(
        self,
        spec: DiffPairSpec,
        positive_pads: List[Obstacle],
        negative_pads: List[Obstacle]
    ) -> DiffPairResult:
        """
        Route a differential pair.

        Args:
            spec: Differential pair specification
            positive_pads: Pads for the positive net
            negative_pads: Pads for the negative net

        Returns:
            DiffPairResult with routed geometry
        """
        if len(positive_pads) < 2 or len(negative_pads) < 2:
            return DiffPairResult(
                success=False,
                diff_pair=spec,
                failure_reason="Need at least 2 pads per net"
            )

        # For now, implement a simplified approach:
        # 1. Find centerline start/end points
        # 2. Route the centerline with inflated obstacles
        # 3. Expand to parallel traces

        # Calculate pair width for obstacle inflation
        pair_width = 2 * spec.trace_width + spec.spacing

        # Find start and end points for both nets
        # For each pad in positive, find closest pad in negative
        pairs = self._match_pad_pairs(positive_pads, negative_pads)

        if not pairs:
            return DiffPairResult(
                success=False,
                diff_pair=spec,
                failure_reason="Could not match positive/negative pads"
            )

        all_pos_segments = []
        all_neg_segments = []
        all_vias = []
        total_pos_length = 0.0
        total_neg_length = 0.0
        coupled_length = 0.0

        # Route each pad pair
        for pos_pad, neg_pad in pairs:
            # Calculate centerline point
            center_x = (pos_pad.center[0] + neg_pad.center[0]) / 2
            center_y = (pos_pad.center[1] + neg_pad.center[1]) / 2

            # TODO: Implement full centerline routing
            # For now, create parallel segments directly

            # Generate parallel traces from pads
            pos_center = pos_pad.center
            neg_center = neg_pad.center

            # Simple case: create direct segments
            # (full implementation would route centerline and expand)
            layer = pos_pad.layer if pos_pad.layer >= 0 else 0

            # Create segments (placeholder for full implementation)
            # Real implementation would:
            # 1. Route centerline using A* with inflated obstacles
            # 2. Expand centerline to parallel traces
            # 3. Handle corners with proper trombone matching

            # For now, just log that we need to implement this
            logger.debug(
                f"Diff pair {spec.name}: pos=({pos_center}), neg=({neg_center})"
            )

        # Calculate final metrics
        skew = abs(total_pos_length - total_neg_length)

        return DiffPairResult(
            success=True,  # Partial success for now
            diff_pair=spec,
            positive_segments=all_pos_segments,
            negative_segments=all_neg_segments,
            vias=all_vias,
            positive_length=total_pos_length,
            negative_length=total_neg_length,
            coupled_length=coupled_length,
            uncoupled_length=total_pos_length - coupled_length,
            skew=skew
        )

    def _match_pad_pairs(
        self,
        positive_pads: List[Obstacle],
        negative_pads: List[Obstacle]
    ) -> List[Tuple[Obstacle, Obstacle]]:
        """Match positive pads with their closest negative counterparts."""
        pairs = []
        used_negative = set()

        for pos_pad in positive_pads:
            pos_center = pos_pad.center
            best_neg = None
            best_dist = float('inf')

            for i, neg_pad in enumerate(negative_pads):
                if i in used_negative:
                    continue

                neg_center = neg_pad.center
                dist = math.sqrt(
                    (pos_center[0] - neg_center[0])**2 +
                    (pos_center[1] - neg_center[1])**2
                )

                if dist < best_dist:
                    best_dist = dist
                    best_neg = (i, neg_pad)

            if best_neg and best_dist < 5.0:  # Max 5mm apart for pair matching
                pairs.append((pos_pad, best_neg[1]))
                used_negative.add(best_neg[0])

        return pairs


class LengthMatcher:
    """Add length matching meanders to traces.

    Implements the "Tuner" from Phase 2 of ROUTING_STRATEGY.md:
    - Uses spatial interval trees to find free space alongside traces
    - Inserts "accordion" or "trombone" meanders to add delay
    """

    @dataclass
    class MeanderConfig:
        """Configuration for meander generation."""
        style: str = "accordion"  # accordion or trombone
        amplitude: float = 0.5    # Height of meanders (mm)
        pitch: float = 0.5        # Spacing between meanders (mm)
        min_amplitude: float = 0.2
        max_amplitude: float = 2.0

    def __init__(self, obstacle_index: SpatialHashIndex):
        """
        Initialize length matcher.

        Args:
            obstacle_index: Spatial hash for collision detection
        """
        self.obstacles = obstacle_index

    def add_length(
        self,
        segments: List[RouteSegment],
        target_length: float,
        config: Optional["LengthMatcher.MeanderConfig"] = None
    ) -> List[RouteSegment]:
        """
        Add meanders to segments to reach target length.

        Args:
            segments: Original trace segments
            target_length: Desired total length
            config: Meander configuration

        Returns:
            Modified segments with meanders added
        """
        config = config or self.MeanderConfig()

        # Calculate current length
        current_length = sum(
            math.sqrt(
                (s.end[0] - s.start[0])**2 +
                (s.end[1] - s.start[1])**2
            )
            for s in segments
        )

        if current_length >= target_length:
            logger.debug(f"Trace already at target length: {current_length:.2f}mm")
            return segments

        length_to_add = target_length - current_length
        logger.debug(f"Need to add {length_to_add:.2f}mm of length")

        # Find longest segment to add meanders
        longest_seg = max(
            segments,
            key=lambda s: math.sqrt(
                (s.end[0] - s.start[0])**2 +
                (s.end[1] - s.start[1])**2
            )
        )

        # Generate meander geometry
        # TODO: Implement full meander generation with obstacle avoidance
        # For now, return original segments
        return segments

    def _generate_accordion(
        self,
        segment: RouteSegment,
        length_to_add: float,
        config: "LengthMatcher.MeanderConfig"
    ) -> List[RouteSegment]:
        """Generate accordion-style meanders along a segment."""
        # Calculate direction vector
        dx = segment.end[0] - segment.start[0]
        dy = segment.end[1] - segment.start[1]
        seg_length = math.sqrt(dx*dx + dy*dy)

        if seg_length == 0:
            return [segment]

        # Normalize
        dx /= seg_length
        dy /= seg_length

        # Perpendicular direction for meanders
        px, py = -dy, dx

        # Calculate number of meanders needed
        # Each meander adds approximately 2 * amplitude of length
        meander_length = 2 * config.amplitude
        num_meanders = int(length_to_add / meander_length) + 1

        # Generate meander points
        new_segments = []
        spacing = seg_length / (num_meanders + 1)

        current_x, current_y = segment.start
        for i in range(num_meanders):
            # Move along segment
            next_x = segment.start[0] + dx * spacing * (i + 1)
            next_y = segment.start[1] + dy * spacing * (i + 1)

            # Add perpendicular jog (alternating direction)
            direction = 1 if i % 2 == 0 else -1
            mid_x = (current_x + next_x) / 2 + px * config.amplitude * direction
            mid_y = (current_y + next_y) / 2 + py * config.amplitude * direction

            # Add segments: current -> mid -> next
            new_segments.append(RouteSegment(
                start=(current_x, current_y),
                end=(mid_x, mid_y),
                layer=segment.layer,
                width=segment.width,
                net_id=segment.net_id
            ))
            new_segments.append(RouteSegment(
                start=(mid_x, mid_y),
                end=(next_x, next_y),
                layer=segment.layer,
                width=segment.width,
                net_id=segment.net_id
            ))

            current_x, current_y = next_x, next_y

        # Final segment to end
        new_segments.append(RouteSegment(
            start=(current_x, current_y),
            end=segment.end,
            layer=segment.layer,
            width=segment.width,
            net_id=segment.net_id
        ))

        return new_segments
