"""
AtoPlace MCP Server

Exposes the Layout DSL and Context Generators to LLM agents via
the Model Context Protocol (MCP).

Tools are organized into categories:
- Board Management: load, save
- Placement Actions: move, place_next_to, align, etc.
- Discovery: find components, get bounds, list unplaced
- Topology: get connections, find critical nets
- Context: inspect region, get summary, render view
- Validation: check overlaps, validate placement
"""

from pathlib import Path
from typing import List, Optional
import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Stub for when MCP is not installed
    class FastMCP:
        def __init__(self, name): self.name = name
        def tool(self): return lambda f: f
        def resource(self, uri): return lambda f: f
        def run(self): print("MCP not installed")

from ..rpc.client import RpcClient
from .prompts import SYSTEM_PROMPT, FIX_OVERLAPS_PROMPT
from .kicad import find_kicad_python


# Initialize FastMCP server
mcp = FastMCP("atoplace")

# Initialize RPC Client with proper KiCad Python detection
kicad_py = find_kicad_python()
if not kicad_py:
    # Fall back to regular python3 if KiCad Python not found
    logger.warning("KiCad Python not found, falling back to python3 - pcbnew may not be available")
    kicad_py = "python3"
else:
    logger.info("Using KiCad Python: %s", kicad_py)
client = RpcClient(kicad_py)


# =============================================================================
# Response Helpers
# =============================================================================

def _error_response(message: str, code: str = "error") -> str:
    """Standard error response format."""
    logger.warning("Error response: code=%s message=%s", code, message)
    return json.dumps({
        "status": "error",
        "code": code,
        "message": message
    })


def _success_response(data: dict) -> str:
    """Standard success response format."""
    return json.dumps({"status": "success", **data})


# =============================================================================
# Board Management Tools
# =============================================================================

@mcp.tool()
def load_board(path: str) -> str:
    """Load a PCB file for editing."""
    try:
        result = client.call("load_board", path=path)
        return _success_response(result)
    except Exception as e:
        return _error_response(str(e), "load_failed")


@mcp.tool()
def save_board(path: Optional[str] = None) -> str:
    """Save the modified board."""
    try:
        path = client.call("save_board", path=path)
        return _success_response({"path": path})
    except Exception as e:
        return _error_response(str(e), "save_failed")


# =============================================================================
# Placement Action Tools
# =============================================================================

@mcp.tool()
def move_absolute(ref: str, x: float, y: float, rotation: Optional[float] = None) -> str:
    """Move component to absolute coordinates."""
    try:
        result = client.call("move_absolute", ref=ref, x=x, y=y, rotation=rotation)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def move_relative(ref: str, dx: float, dy: float) -> str:
    """Move component by a relative delta."""
    try:
        result = client.call("move_relative", ref=ref, dx=dx, dy=dy)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def rotate(ref: str, angle: float) -> str:
    """Set absolute rotation of component."""
    try:
        result = client.call("rotate", ref=ref, angle=angle)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def place_next_to(
    ref: str, target: str, side: str = "right", clearance: float = 0.5, align: str = "center"
) -> str:
    """Place a component next to another."""
    try:
        result = client.call("place_next_to", ref=ref, target=target, side=side, clearance=clearance, align=align)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def align_components(refs: List[str], axis: str = "x", anchor: str = "first") -> str:
    """Align components along an axis."""
    try:
        result = client.call("align_components", refs=refs, axis=axis, anchor=anchor)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def distribute_evenly(
    refs: List[str], start_ref: Optional[str] = None, end_ref: Optional[str] = None, axis: str = "auto"
) -> str:
    """Distribute components evenly between two points."""
    try:
        result = client.call("distribute_evenly", refs=refs, start_ref=start_ref, end_ref=end_ref, axis=axis)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def stack_components(
    refs: List[str], direction: str = "down", spacing: float = 0.5, alignment: str = "center"
) -> str:
    """Stack components sequentially in a direction."""
    try:
        result = client.call("stack_components", refs=refs, direction=direction, spacing=spacing, alignment=alignment)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def lock_components(refs: List[str], locked: bool = True) -> str:
    """Lock or unlock components."""
    try:
        result = client.call("lock_components", refs=refs, locked=locked)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def arrange_pattern(
    refs: List[str],
    pattern: str = "grid",
    spacing: float = 2,
    cols: Optional[int] = None,
    radius: Optional[float] = None,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None
) -> str:
    """Arrange components in a pattern (grid, row, column, or circular)."""
    try:
        result = client.call("arrange_pattern",refs=refs, pattern=pattern, spacing=spacing,
                           cols=cols, radius=radius, center_x=center_x, center_y=center_y)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")

@mcp.tool()
def cluster_around(
    anchor_ref: str, target_refs: List[str], side: str = "nearest", clearance: float = 0.5
) -> str:
    """Cluster components around an anchor."""
    try:
        result = client.call("cluster_around", anchor_ref=anchor_ref, target_refs=target_refs, side=side, clearance=clearance)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def inspect_region(refs: List[str], padding: float = 5.0, include_image: bool = False) -> str:
    """Get detailed geometric data for a set of components."""
    try:
        result = client.call("inspect_region", refs=refs, padding=padding, include_image=include_image)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "inspect_failed")


@mcp.tool()
def get_board_summary() -> str:
    """Get executive summary of board state."""
    try:
        result = client.call("get_board_summary")
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "summary_failed")


@mcp.tool()
def check_overlaps(refs: Optional[List[str]] = None) -> str:
    """Check for component overlaps."""
    try:
        result = client.call("check_overlaps", refs=refs)
        return _success_response(result)
    except Exception as e:
        return _error_response(str(e), "check_failed")


# =============================================================================
# Discovery Tools
# =============================================================================

@mcp.tool()
def get_unplaced_components() -> str:
    """Get list of components that are outside the board area."""
    try:
        result = client.call("get_unplaced_components")
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "discovery_failed")

@mcp.tool()
def find_components(query: str, filter_by: str = "ref") -> str:
    """Find components matching a query (ref, value, or footprint)."""
    try:
        result = client.call("find_components", query=query, filter_by=filter_by)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "search_failed")


# =============================================================================
# Validation Tools
# =============================================================================

@mcp.tool()
def run_drc(
    use_kicad: bool = True,
    dfm_profile: str = "jlcpcb_standard",
    severity_filter: str = "all"
) -> str:
    """Run Design Rule Check on the loaded board."""
    try:
        result = client.call("run_drc", use_kicad=use_kicad, dfm_profile=dfm_profile, severity_filter=severity_filter)
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "drc_failed")

@mcp.tool()
def validate_placement() -> str:
    """Run placement validation and get confidence report."""
    try:
        result = client.call("validate_placement")
        return json.dumps(result)
    except Exception as e:
        return _error_response(str(e), "validation_failed")


# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("prompts://system")
def system_prompt_resource() -> str:
    return SYSTEM_PROMPT

@mcp.resource("prompts://fix_overlaps")
def fix_overlaps_prompt_resource() -> str:
    return FIX_OVERLAPS_PROMPT


def main():
    if not MCP_AVAILABLE:
        print("Error: MCP package not installed.")
        return
    mcp.run()

if __name__ == "__main__":
    main()
