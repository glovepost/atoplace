"""MCP server for Claude integration."""

# Lazy imports to avoid ModuleNotFoundError until implementation exists
__all__ = ["AtoPlaceMCPServer", "create_server"]


def __getattr__(name):
    """Lazy import MCP components."""
    if name == "AtoPlaceMCPServer":
        try:
            from .server import AtoPlaceMCPServer
            return AtoPlaceMCPServer
        except ImportError:
            raise ImportError(
                "AtoPlaceMCPServer not yet implemented. "
                "See docs/PRODUCT_PLAN.md MCP Server Integration section."
            )
    elif name == "create_server":
        try:
            from .server import create_server
            return create_server
        except ImportError:
            raise ImportError(
                "create_server not yet implemented. "
                "See docs/PRODUCT_PLAN.md MCP Server Integration section."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
