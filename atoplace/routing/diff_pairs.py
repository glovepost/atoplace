"""
Differential Pair Routing

Implements "Dual-Grid" A* routing for differential pairs.
Routes the virtual centerline with anisotropic costs to maintain coupling.
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set, Union

from ..board.abstraction import Board
from ..dfm.profiles import DFMProfile
from .spatial_index import SpatialHashIndex, Obstacle
from .visualizer import RouteVisualizer, RouteSegment, Via
from .astar_router import RouteNode

logger = logging.getLogger(__name__)


class DiffPairPattern(Enum):
    """Pattern type for detected differential pairs."""
    USB_STYLE = "usb_style"            # D+/D- or +/- style
    PLUS_MINUS = "plus_minus"          # _P/_N suffix style
    SUFFIX_P_N = "suffix_p_n"          # Direct P/N suffix (DATAP/DATAN)
    POSITIVE_NEGATIVE = "pos_neg"      # _POS/_NEG suffix style
    CUSTOM = "custom"                  # User-defined pair


@dataclass
class DiffPairSpec:
    """Specification for a differential pair."""
    net_p: str
    net_n: str
    gap: float = 0.15        # Gap between traces
    width: float = 0.2       # Trace width
    uncoupled_length_max: float = 5.0  # Max length allowed to be uncoupled
    phase_tolerance: float = 0.5       # Max phase skew (length difference)
    pattern: Optional[DiffPairPattern] = None  # How the pair was detected

    @property
    def positive_net(self) -> str:
        """Alias for net_p (positive net)."""
        return self.net_p

    @property
    def negative_net(self) -> str:
        """Alias for net_n (negative net)."""
        return self.net_n

    @property
    def name(self) -> str:
        """Auto-generated name from net names (for identification)."""
        # Extract common base name from the nets
        net_p = self.net_p
        net_n = self.net_n
        # Find common prefix
        common = ""
        for c1, c2 in zip(net_p, net_n):
            if c1 == c2:
                common += c1
            else:
                break
        # Trim trailing delimiters
        return common.rstrip("_+-")


@dataclass
class DiffPairGeometry:
    """Generated geometry for a diff pair segment."""
    centerline: List[RouteNode]
    segments_p: List[RouteSegment] = field(default_factory=list)
    segments_n: List[RouteSegment] = field(default_factory=list)
    vias_p: List[Via] = field(default_factory=list)
    vias_n: List[Via] = field(default_factory=list)


@dataclass
class DiffPairResult:
    """Result of routing a diff pair."""
    success: bool
    spec: DiffPairSpec
    geometry: Optional[DiffPairGeometry] = None  # Generated routing geometry
    uncoupled_length: float = 0.0
    phase_skew: float = 0.0
    failure_reason: str = ""


class DiffPairDetector:
    """Detects differential pairs from net names.

    Can be initialized with either a Board object or a list of net names.
    """

    def __init__(self, board_or_nets):
        """Initialize with a Board object or a list of net names.

        Args:
            board_or_nets: Either a Board object (uses board.nets.keys())
                          or a list of net name strings.
        """
        if isinstance(board_or_nets, list):
            self._net_names = board_or_nets
            self.board = None
        else:
            self.board = board_or_nets
            self._net_names = None

    def detect(self) -> List[DiffPairSpec]:
        """
        Auto-detect differential pairs based on naming conventions.

        Looks for suffixes: _P/_N, +/-, _DP/_DN, _POS/_NEG
        """
        pairs = []
        # Support both Board objects and direct net name lists
        if self._net_names is not None:
            nets = self._net_names
        else:
            nets = list(self.board.nets.keys())
        processed = set()

        # Suffix patterns with their corresponding DiffPairPattern enum
        suffix_patterns = [
            (('_P', '_N'), DiffPairPattern.PLUS_MINUS),
            (('+', '-'), DiffPairPattern.USB_STYLE),
            (('_DP', '_DN'), DiffPairPattern.PLUS_MINUS),
            (('_POS', '_NEG'), DiffPairPattern.POSITIVE_NEGATIVE),
            (('P', 'N'), DiffPairPattern.SUFFIX_P_N),  # Direct P/N suffix
        ]

        for net in nets:
            if net in processed:
                continue

            net_upper = net.upper()

            for (p_suf, n_suf), pattern_type in suffix_patterns:
                if net_upper.endswith(p_suf):
                    base = net[:len(net)-len(p_suf)]
                    mate = base + n_suf

                    # Check if mate exists (case-insensitive search)
                    mate_actual = None
                    for candidate in nets:
                        if candidate.upper() == mate.upper():
                            mate_actual = candidate
                            break

                    if mate_actual:
                        # Found a pair
                        pairs.append(DiffPairSpec(
                            net_p=net,
                            net_n=mate_actual,
                            pattern=pattern_type
                        ))
                        processed.add(net)
                        processed.add(mate_actual)
                        break

        logger.info(f"Detected {len(pairs)} differential pairs")
        return pairs


class DiffPairRouter:
    """
    Routes differential pairs using coupled pathfinding.
    """

    def __init__(
        self,
        obstacle_index: SpatialHashIndex,
        dfm_profile: DFMProfile,
        visualizer: Optional[RouteVisualizer] = None
    ):
        self.obstacles = obstacle_index
        self.dfm = dfm_profile
        self.viz = visualizer

    def route_pair(self, spec: DiffPairSpec, start_pads: Tuple, end_pads: Tuple) -> DiffPairResult:
        """
        Route a single differential pair.
        
        Args:
            spec: Diff pair specification
            start_pads: (pad_p, pad_n) tuple of start Obstacles
            end_pads: (pad_p, pad_n) tuple of end Obstacles
            
        Returns:
            DiffPairResult
        """
        # Placeholder for full implementation
        # 1. Calculate effective width (2*w + g)
        # 2. Inflate obstacles by effective width/2 + clearance
        # 3. Run A* on centerline
        # 4. Generate dual traces
        # 5. Length match
        
        logger.warning(f"DiffPairRouter.route_pair not fully implemented for {spec.net_p}/{spec.net_n}")
        return DiffPairResult(success=False, spec=spec, failure_reason="Not implemented")


class LengthMatcher:
    """
    Adds meanders (accordions) to traces to match target lengths.
    """
    
    def match_length(
        self, 
        segments: List[RouteSegment], 
        target_length: float, 
        tolerance: float = 0.1
    ) -> List[RouteSegment]:
        """
        Modify segments to increase total length to target.
        """
        # Placeholder
        return segments