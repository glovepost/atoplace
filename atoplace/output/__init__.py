"""Manufacturing output generation."""

# Lazy imports to avoid ModuleNotFoundError until implementation exists
__all__ = ["ManufacturingOutputGenerator", "JLCPCBExporter", "GerberGenerator"]


def __getattr__(name):
    """Lazy import output components."""
    if name == "ManufacturingOutputGenerator":
        try:
            from .manufacturing import ManufacturingOutputGenerator
            return ManufacturingOutputGenerator
        except ImportError:
            raise ImportError(
                "ManufacturingOutputGenerator is not yet implemented. "
                "This feature is planned for a future release."
            )
    elif name == "JLCPCBExporter":
        try:
            from .jlcpcb import JLCPCBExporter
            return JLCPCBExporter
        except ImportError:
            raise ImportError(
                "JLCPCBExporter is not yet implemented. "
                "This feature is planned for a future release."
            )
    elif name == "GerberGenerator":
        try:
            from .gerber import GerberGenerator
            return GerberGenerator
        except ImportError:
            raise ImportError(
                "GerberGenerator is not yet implemented. "
                "This feature is planned for a future release."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
