"""Board abstraction layer for unified KiCad/atopile access."""

from .abstraction import Board, Component, Net, Pad, Layer

__all__ = ["Board", "Component", "Net", "Pad", "Layer"]
