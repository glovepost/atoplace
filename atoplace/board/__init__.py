"""Board abstraction layer for unified KiCad/atopile access."""

from .abstraction import Board, Component, Net, Pad, Layer, BoardOutline, RefDesText
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
    "RefDesText",
    # Atopile integration
    "AtopileProjectLoader",
    "AtopileModuleParser",
    "ComponentMetadata",
    "ModuleHierarchy",
    "detect_board_source",
    "load_board_auto",
]
