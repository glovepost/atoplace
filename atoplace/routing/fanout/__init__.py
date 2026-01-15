"""BGA/FPGA Fanout routing module.

Automated escape routing for high-density BGA packages (FPGAs, MCUs, SoCs).

This module implements:
- Dogbone fanout pattern (for pitch >= 0.65mm)
- Via-in-Pad (VIP) pattern (for pitch <= 0.5mm)
- Layer assignment using the "Onion" model (outer rings on outer layers)
- Escape routing from fanout vias to clear routing space

Usage:
    from atoplace.routing.fanout import FanoutGenerator

    generator = FanoutGenerator(board, dfm_profile)
    result = generator.fanout_component("U1")

    # Or auto-detect and fanout all BGAs
    results = generator.fanout_all_bgas()
"""

from .generator import FanoutGenerator, FanoutResult, FanoutStrategy
from .patterns import (
    DogbonePattern,
    VIPPattern,
    FanoutVia,
    FanoutTrace,
    calculate_optimal_dogbone_offset,
)
from .layer_assigner import LayerAssigner, PinRing, LayerMapping
from .escape_router import EscapeRouter, EscapeResult, EscapeDirection

__all__ = [
    # Main entry point
    "FanoutGenerator",
    "FanoutResult",
    "FanoutStrategy",
    # Patterns
    "DogbonePattern",
    "VIPPattern",
    "FanoutVia",
    "FanoutTrace",
    "calculate_optimal_dogbone_offset",
    # Layer assignment
    "LayerAssigner",
    "PinRing",
    "LayerMapping",
    # Escape routing
    "EscapeRouter",
    "EscapeResult",
    "EscapeDirection",
]
