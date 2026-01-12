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
                "ManufacturingOutputGenerator not yet implemented. "
                "See docs/PRODUCT_PLAN.md Phase 4 for implementation plan."
            )
    elif name == "JLCPCBExporter":
        try:
            from .jlcpcb import JLCPCBExporter
            return JLCPCBExporter
        except ImportError:
            raise ImportError(
                "JLCPCBExporter not yet implemented. "
                "See docs/PRODUCT_PLAN.md Phase 4 for implementation plan."
            )
    elif name == "GerberGenerator":
        try:
            from .gerber import GerberGenerator
            return GerberGenerator
        except ImportError:
            raise ImportError(
                "GerberGenerator not yet implemented. "
                "See docs/PRODUCT_PLAN.md Phase 4 for implementation plan."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
