"""Pin swapping optimization module (Phase 0 of routing pipeline).

Reduces ratsnest crossings by intelligently swapping functionally equivalent pins
before routing begins. This is critical for complex FPGA/MCU boards where poor
pin assignment creates unroutable congestion.

This module implements:
- Swap group detection for MCU GPIO banks, FPGA I/O banks, and connector pins
- Crossing count analysis to measure ratsnest complexity
- Bipartite matching optimization (Hungarian algorithm) to minimize crossings
- Constraint file generation (XDC for Xilinx, QSF for Intel, netlist updates)

Usage:
    from atoplace.routing.pinswapper import PinSwapper

    swapper = PinSwapper(board)
    result = swapper.optimize_component("U1")

    # Or auto-detect and optimize all swappable components
    results = swapper.optimize_all()

    # Generate constraint updates
    swapper.export_constraints("constraints.xdc", format="xdc")
"""

from .detector import (
    SwapGroupDetector,
    SwapGroup,
    SwapGroupType,
    SwappablePin,
)
from .crossing import (
    CrossingCounter,
    CrossingResult,
    RatsnestEdge,
)
from .optimizer import (
    BipartiteMatcher,
    MatchingResult,
    SwapAssignment,
)
from .constraints import (
    ConstraintGenerator,
    ConstraintFormat,
    ConstraintUpdate,
)
from .swapper import (
    PinSwapper,
    SwapResult,
    SwapConfig,
)

__all__ = [
    # Main entry point
    "PinSwapper",
    "SwapResult",
    "SwapConfig",
    # Detection
    "SwapGroupDetector",
    "SwapGroup",
    "SwapGroupType",
    "SwappablePin",
    # Crossing analysis
    "CrossingCounter",
    "CrossingResult",
    "RatsnestEdge",
    # Optimization
    "BipartiteMatcher",
    "MatchingResult",
    "SwapAssignment",
    # Constraint generation
    "ConstraintGenerator",
    "ConstraintFormat",
    "ConstraintUpdate",
]
