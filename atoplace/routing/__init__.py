"""Routing engine with Freerouting integration."""

# Lazy imports to avoid ModuleNotFoundError until implementation exists
__all__ = ["FreeroutingRunner", "NetClassAssigner", "DiffPairDetector"]


def __getattr__(name):
    """Lazy import routing components."""
    if name == "FreeroutingRunner":
        try:
            from .freerouting import FreeroutingRunner
            return FreeroutingRunner
        except ImportError:
            raise ImportError(
                "FreeroutingRunner not yet implemented. "
                "See docs/PRODUCT_PLAN.md Phase 3 for implementation plan."
            )
    elif name == "NetClassAssigner":
        try:
            from .net_classes import NetClassAssigner
            return NetClassAssigner
        except ImportError:
            raise ImportError(
                "NetClassAssigner not yet implemented. "
                "See docs/PRODUCT_PLAN.md Phase 3 for implementation plan."
            )
    elif name == "DiffPairDetector":
        try:
            from .diff_pairs import DiffPairDetector
            return DiffPairDetector
        except ImportError:
            raise ImportError(
                "DiffPairDetector not yet implemented. "
                "See docs/PRODUCT_PLAN.md Phase 3 for implementation plan."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
