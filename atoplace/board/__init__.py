"""Board abstraction layer for unified KiCad/atopile access."""

from .abstraction import Board, BoardOutline, Component, Layer, Net, Pad, RefDesText
from .atopile_adapter import (
    AtopileModuleParser,
    AtopileProjectLoader,
    ComponentMetadata,
    ModuleHierarchy,
    detect_board_source,
    load_board_auto,
)
from .lock_file import (
    AtoplaceLock,
    ComponentPosition,
    apply_lock_to_board,
    create_lock_from_board,
    get_lock_file_path,
    merge_lock_files,
    parse_lock_file,
    write_lock_file,
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
    # Lock file persistence
    "AtoplaceLock",
    "ComponentPosition",
    "get_lock_file_path",
    "parse_lock_file",
    "write_lock_file",
    "apply_lock_to_board",
    "create_lock_from_board",
    "merge_lock_files",
]
