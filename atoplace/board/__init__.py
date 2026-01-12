"""Board abstraction layer for unified KiCad/atopile access."""

from .abstraction import Board, Component, Net, Pad, Layer, BoardOutline
from .atopile_adapter import (
    AtopileProjectLoader,
    AtopileModuleParser,
    ComponentMetadata,
    ModuleHierarchy,
    detect_board_source,
    load_board_auto,
)

__all__ = [
    # Core abstractions
    "Board",
    "Component",
    "Net",
    "Pad",
    "Layer",
    "BoardOutline",
    # Atopile integration
    "AtopileProjectLoader",
    "AtopileModuleParser",
    "ComponentMetadata",
    "ModuleHierarchy",
    "detect_board_source",
    "load_board_auto",
]
