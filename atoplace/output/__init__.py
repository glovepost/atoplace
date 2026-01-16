"""Manufacturing output generation.

⚠️  PLANNED FEATURES - NOT YET IMPLEMENTED ⚠️

This module contains stubs for planned manufacturing output features
including Gerber generation, JLCPCB export, and assembly file creation.

These features are currently under development. For now, use KiCad's
native export tools after running AtoPlace placement/routing.

See README.md in this directory for:
- Planned features and capabilities
- Current workarounds using KiCad
- Implementation status and roadmap
- How to contribute

Example current workflow:
    >>> from atoplace.board.abstraction import Board
    >>> board = Board.from_kicad("input.kicad_pcb")
    >>> # ... perform placement operations ...
    >>> board.to_kicad("output.kicad_pcb")
    >>> # Then use KiCad GUI/CLI for Gerber export
"""

# Lazy imports to avoid ModuleNotFoundError until implementation exists
__all__ = ["ManufacturingOutputGenerator", "JLCPCBExporter", "GerberGenerator"]


def __getattr__(name):
    """Lazy import output components.

    Raises:
        ImportError: These features are not yet implemented.
            See atoplace/output/README.md for details and workarounds.
    """
    error_msg = (
        f"{name} is not yet implemented. "
        "This feature is planned for a future release.\n\n"
        "Current workaround: Use KiCad's native export after running AtoPlace.\n"
        "See atoplace/output/README.md for details."
    )

    if name == "ManufacturingOutputGenerator":
        try:
            from .manufacturing import ManufacturingOutputGenerator
            return ManufacturingOutputGenerator
        except ImportError:
            raise ImportError(error_msg)
    elif name == "JLCPCBExporter":
        try:
            from .jlcpcb import JLCPCBExporter
            return JLCPCBExporter
        except ImportError:
            raise ImportError(error_msg)
    elif name == "GerberGenerator":
        try:
            from .gerber import GerberGenerator
            return GerberGenerator
        except ImportError:
            raise ImportError(error_msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
