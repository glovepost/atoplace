"""
AtoPlace MCP (Model Context Protocol) Server

AI-powered PCB layout tools exposed via MCP for LLM agents.

Quick Start:
    python -m atoplace.mcp.launcher

Architecture:
    - launcher: Single entry point, auto-starts bridge + server
    - bridge: Runs in KiCad Python (3.9) with pcbnew access
    - server: FastMCP server (3.10+) with 26 PCB layout tools
    - ipc: Unix socket communication between processes

The launcher auto-detects KiCad Python on macOS/Linux/Windows.
Override with KICAD_PYTHON environment variable if needed.
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

# IPC components (lazy import to avoid Python 3.9 issues)
def get_ipc_client():
    """Get IPC client class."""
    from .ipc import IPCClient
    return IPCClient

def get_ipc_session():
    """Get IPC session class."""
    from .ipc_session import IPCSession
    return IPCSession

def get_kipy_session():
    """Get KiPy session class for live KiCad IPC (KiCad 9+)."""
    from .kipy_session import KiPySession
    return KiPySession

# Backend selection
from .backends import BackendMode, get_backend_mode, create_session as create_backend_session

# DRC (lazy import)
def get_drc_runner():
    """Get DRC runner instance."""
    from .drc import get_drc_runner as _get_runner
    return _get_runner()

def get_drc_fixer():
    """Get DRC fixer class."""
    from .drc import DRCFixer
    return DRCFixer

__all__ = [
    # Server
    "mcp",
    "main",
    "session",
    # Backend selection
    "BackendMode",
    "get_backend_mode",
    "create_backend_session",
    # Session classes (lazy)
    "get_ipc_client",
    "get_ipc_session",
    "get_kipy_session",
    # DRC
    "get_drc_runner",
    "get_drc_fixer",
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
