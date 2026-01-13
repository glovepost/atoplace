"""
AtoPlace MCP (Model Context Protocol) Server

Exposes the Layout DSL and Context Generators to LLM agents.

Modules:
- server: FastMCP server with tool registrations
- context: Multi-level context generators (macro, micro, vision)
"""

from .server import mcp, main, session

# Context generators
from .context import (
    Microscope,
    MicroscopeData,
    MacroContext,
    BoardSummary,
    SemanticGrid,
    ModuleMap,
    VisionContext,
)

__all__ = [
    # Server
    "mcp",
    "main",
    "session",
    # Context - Micro
    "Microscope",
    "MicroscopeData",
    # Context - Macro
    "MacroContext",
    "BoardSummary",
    "SemanticGrid",
    "ModuleMap",
    # Context - Vision
    "VisionContext",
]
