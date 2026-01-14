"""
AtoPlace MCP Server

Exposes the Layout DSL and Context Generators to LLM agents via
the Model Context Protocol (MCP).

Tools are organized into categories:
- Board Management: load, save, undo/redo
- Placement Actions: move, place_next_to, align, etc.
- Discovery: find components, get bounds, list unplaced
- Topology: get connections, find critical nets
- Context: inspect region, get summary, render view
- Validation: check overlaps, validate placement
"""

from pathlib import Path
from typing import List, Optional, Union
import json
import logging
import os
import sys

# Configure logging - use file to keep STDIO clean for MCP protocol
LOG_FILE = os.environ.get("ATOPLACE_LOG", "/tmp/atoplace.log")
_log_configured = False

def _configure_logging():
    """Configure logging once."""
    global _log_configured
    if _log_configured:
        return

    # Only add file handler if not already configured
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.FileHandler(LOG_FILE, mode="a")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    _log_configured = True

_configure_logging()
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
        def run(self): pass

from ..board.abstraction import Board
from ..api.actions import LayoutActions
from ..api.session import Session
from .context.micro import Microscope
from .context.macro import MacroContext
from .context.vision import VisionContext
from .prompts import SYSTEM_PROMPT, FIX_OVERLAPS_PROMPT


# Initialize FastMCP server
mcp = FastMCP("atoplace")

# Session state - supports multiple backends:
# - direct: Direct pcbnew access (KiCad Python environment)
# - ipc: Bridge-based IPC to pcbnew process
# - kipy: Live KiCad IPC via official kicad-python API (KiCad 9+)
#
# Set via ATOPLACE_BACKEND env var: direct, ipc, or kipy
# Or legacy: ATOPLACE_USE_IPC=1 or ATOPLACE_USE_KIPY=1
from .backends import create_session, get_backend_mode, BackendNotAvailableError

try:
    session = create_session()
    logger.info("MCP server using %s mode", get_backend_mode().value)
except BackendNotAvailableError as e:
    logger.warning("Preferred backend not available: %s. Falling back to direct.", e)
    session = Session()
    logger.info("MCP server using direct mode (fallback)")


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
# Validation Helpers
# =============================================================================

def _validate_ref(ref: str) -> Optional[str]:
    """Validate component reference exists. Returns error message or None."""
    if not ref:
        return "Component reference cannot be empty"
    if not session.board or ref not in session.board.components:
        valid_refs = list(session.board.components.keys())[:5] if session.board else []
        hint = f" Valid refs include: {valid_refs}" if valid_refs else ""
        return f"Component '{ref}' not found.{hint}"
    return None


def _validate_refs(refs: List[str]) -> Optional[str]:
    """Validate multiple component references. Returns error message or None."""
    if not refs:
        return "Component reference list cannot be empty"
    missing = [r for r in refs if r not in session.board.components]
    if missing:
        return f"Components not found: {missing[:5]}"
    return None


def _validate_side(side: str) -> Optional[str]:
    """Validate side parameter. Returns error message or None."""
    valid = ["top", "bottom", "left", "right"]
    if side.lower() not in valid:
        return f"Side must be one of: {valid}. Got: '{side}'"
    return None


def _validate_axis(axis: str) -> Optional[str]:
    """Validate axis parameter. Returns error message or None."""
    valid = ["x", "y"]
    if axis.lower() not in valid:
        return f"Axis must be one of: {valid}. Got: '{axis}'"
    return None


def _validate_anchor(anchor: str) -> Optional[str]:
    """Validate anchor parameter. Returns error message or None."""
    valid = ["first", "last", "center"]
    if anchor.lower() not in valid:
        return f"Anchor must be one of: {valid}. Got: '{anchor}'"
    return None


def _require_board():
    """Ensure a board is loaded."""
    if not session.is_loaded:
        raise ValueError("No board loaded. Call 'load_board' first.")


# =============================================================================
# Board Management Tools
# =============================================================================

@mcp.tool()
def load_board(path: str) -> str:
    """
    Load a PCB file for editing.

    Args:
        path: Path to KiCad .kicad_pcb file

    Returns:
        Summary of loaded board
    """
    logger.info("load_board called", extra={"path": path})
    try:
        session.load(Path(path))
        board = session.board
        logger.info("Board loaded successfully", extra={
            "components": len(board.components),
            "nets": len(board.nets)
        })
        return _success_response({
            "path": path,
            "components": len(board.components),
            "nets": len(board.nets),
        })
    except Exception as e:
        logger.exception("Failed to load board")
        return _error_response(str(e), "load_failed")


@mcp.tool()
def save_board(path: Optional[str] = None) -> str:
    """
    Save the modified board.

    Args:
        path: Output path. If None, saves to <original>.placed.kicad_pcb

    Returns:
        Path where board was saved
    """
    logger.info("save_board called", extra={"path": path})
    _require_board()
    try:
        output_path = session.save(Path(path) if path else None)
        logger.info("Board saved", extra={"output_path": str(output_path)})
        return _success_response({"path": str(output_path)})
    except Exception as e:
        logger.exception("Failed to save board")
        return _error_response(str(e), "save_failed")


@mcp.tool()
def undo() -> str:
    """Undo the last placement action."""
    logger.debug("undo called")
    _require_board()
    if session.undo():
        logger.info("Undo successful")
        return _success_response({"action": "undone"})
    return _success_response({"action": "nothing_to_undo"})


@mcp.tool()
def redo() -> str:
    """Redo the last undone action."""
    logger.debug("redo called")
    _require_board()
    if session.redo():
        logger.info("Redo successful")
        return _success_response({"action": "redone"})
    return _success_response({"action": "nothing_to_redo"})


# =============================================================================
# Placement Action Tools
# =============================================================================

@mcp.tool()
def move_component(ref: str, x: float, y: float, rotation: Optional[float] = None) -> str:
    """
    Move a component to absolute coordinates.

    Args:
        ref: Component reference (e.g., "U1", "C1")
        x: X coordinate in mm
        y: Y coordinate in mm
        rotation: Optional rotation in degrees (0-360)
    """
    logger.info("move_component called", extra={"ref": ref, "x": x, "y": y, "rotation": rotation})
    _require_board()

    # Validate input
    if err := _validate_ref(ref):
        return _error_response(err, "invalid_ref")

    session.checkpoint(f"Move {ref}")
    actions = LayoutActions(session.board)
    result = actions.move_absolute(ref, x, y, rotation)
    if result.success:
        session.mark_modified(result.modified_refs)
        logger.debug("Move successful", extra={"modified": result.modified_refs})
    else:
        logger.warning("Move failed", extra={"reason": result.message})
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def place_next_to(
    ref: str,
    target: str,
    side: str = "right",
    clearance: float = 0.5,
    align: str = "center"
) -> str:
    """
    Place a component next to another with specified clearance.

    Args:
        ref: Component to move
        target: Reference component to place next to
        side: Which side ("top", "bottom", "left", "right")
        clearance: Gap between components in mm
        align: Alignment ("center", "top", "bottom", "left", "right")
    """
    logger.info("place_next_to called", extra={
        "ref": ref, "target": target, "side": side, "clearance": clearance, "align": align
    })
    _require_board()

    # Validate inputs
    if err := _validate_ref(ref):
        return _error_response(err, "invalid_ref")
    if err := _validate_ref(target):
        return _error_response(err, "invalid_target")
    if err := _validate_side(side):
        return _error_response(err, "invalid_side")

    session.checkpoint(f"Place {ref} next to {target}")
    actions = LayoutActions(session.board)
    result = actions.place_next_to(ref, target, side, clearance, align)
    if result.success:
        session.mark_modified(result.modified_refs)
        logger.debug("Place next to successful")
    else:
        logger.warning("Place next to failed", extra={"reason": result.message})
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def align_components(refs: List[str], axis: str = "x", anchor: str = "first") -> str:
    """
    Align multiple components along an axis.

    Args:
        refs: List of component references to align
        axis: Alignment axis ("x" for row, "y" for column)
        anchor: Reference point ("first", "last", "center")
    """
    logger.info("align_components called", extra={"refs": refs, "axis": axis, "anchor": anchor})
    _require_board()

    # Validate inputs
    if err := _validate_refs(refs):
        return _error_response(err, "invalid_refs")
    if err := _validate_axis(axis):
        return _error_response(err, "invalid_axis")
    if err := _validate_anchor(anchor):
        return _error_response(err, "invalid_anchor")

    session.checkpoint(f"Align {len(refs)} components")
    actions = LayoutActions(session.board)
    result = actions.align_components(refs, axis, anchor)
    if result.success:
        session.mark_modified(result.modified_refs)
        logger.debug("Align successful")
    else:
        logger.warning("Align failed", extra={"reason": result.message})
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def distribute_evenly(
    refs: List[str],
    start_ref: Optional[str] = None,
    end_ref: Optional[str] = None,
    axis: str = "auto"
) -> str:
    """
    Distribute components evenly between two points or outer extremes.

    Args:
        refs: List of components to distribute
        start_ref: Component to anchor start (optional)
        end_ref: Component to anchor end (optional)
        axis: "x", "y", or "auto"
    """
    logger.info("distribute_evenly called", extra={"refs": refs})
    _require_board()

    if err := _validate_refs(refs):
        return _error_response(err, "invalid_refs")

    session.checkpoint(f"Distribute {len(refs)} components")
    actions = LayoutActions(session.board)
    result = actions.distribute_evenly(refs, start_ref, end_ref, axis)
    
    if result.success:
        session.mark_modified(result.modified_refs)
    else:
        logger.warning("Distribute failed", extra={"reason": result.message})
        
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def stack_components(
    refs: List[str],
    direction: str = "down",
    spacing: float = 0.5,
    alignment: str = "center"
) -> str:
    """
    Stack components sequentially in a direction.

    Args:
        refs: List of components to stack
        direction: "up", "down", "left", "right"
        spacing: Gap between components
        alignment: "center", "left", "right", "top", "bottom"
    """
    logger.info("stack_components called", extra={"refs": refs, "dir": direction})
    _require_board()

    if err := _validate_refs(refs):
        return _error_response(err, "invalid_refs")

    session.checkpoint(f"Stack {len(refs)} components")
    actions = LayoutActions(session.board)
    result = actions.stack_components(refs, direction, spacing, alignment)
    
    if result.success:
        session.mark_modified(result.modified_refs)
    else:
        logger.warning("Stack failed", extra={"reason": result.message})
        
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def swap_positions(ref1: str, ref2: str) -> str:
    """
    Swap the positions of two components.

    Args:
        ref1: First component reference
        ref2: Second component reference
    """
    logger.info("swap_positions called", extra={"ref1": ref1, "ref2": ref2})
    _require_board()

    # Validate inputs
    if err := _validate_ref(ref1):
        return _error_response(err, "invalid_ref1")
    if err := _validate_ref(ref2):
        return _error_response(err, "invalid_ref2")

    c1 = session.board.components.get(ref1)
    c2 = session.board.components.get(ref2)

    if c1.locked or c2.locked:
        locked = [r for r, c in [(ref1, c1), (ref2, c2)] if c.locked]
        return _error_response(f"Cannot swap locked components: {locked}", "locked_component")

    session.checkpoint(f"Swap {ref1} and {ref2}")

    # Swap positions
    c1.x, c2.x = c2.x, c1.x
    c1.y, c2.y = c2.y, c1.y

    session.mark_modified([ref1, ref2])
    logger.debug("Swap successful")
    return _success_response({"success": True, "message": f"Swapped {ref1} and {ref2}"})


@mcp.tool()
def group_components(refs: List[str], group_name: str) -> str:
    """
    Group components logically.

    Args:
        refs: List of components
        group_name: Name of the group
    """
    logger.info("group_components called", extra={"group": group_name})
    _require_board()
    
    if err := _validate_refs(refs):
        return _error_response(err, "invalid_refs")
        
    actions = LayoutActions(session.board)
    result = actions.group_components(refs, group_name)
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def lock_components(refs: List[str], locked: bool = True) -> str:
    """
    Lock or unlock components to prevent accidental movement.

    Args:
        refs: List of components
        locked: True to lock, False to unlock
    """
    logger.info("lock_components called", extra={"locked": locked})
    _require_board()
    
    if err := _validate_refs(refs):
        return _error_response(err, "invalid_refs")
        
    actions = LayoutActions(session.board)
    result = actions.lock_components(refs, locked)
    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def arrange_pattern(
    refs: List[str],
    pattern: str = "grid",
    spacing: float = 2.0,
    cols: Optional[int] = None,
    radius: Optional[float] = None,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
) -> str:
    """
    Arrange components in a pattern (grid, row, column, or circular).

    Args:
        refs: List of component references to arrange
        pattern: Pattern type ("grid", "row", "column", "circular")
        spacing: Spacing between components in mm
        cols: Number of columns for grid (auto-calculated if not specified)
        radius: Radius for circular pattern (auto-calculated if not specified)
        center_x: X center for arrangement (uses centroid if not specified)
        center_y: Y center for arrangement (uses centroid if not specified)
    """
    logger.info("arrange_pattern called", extra={
        "refs": refs, "pattern": pattern, "spacing": spacing
    })
    _require_board()

    # Validate inputs
    if err := _validate_refs(refs):
        return _error_response(err, "invalid_refs")

    valid_patterns = ["grid", "row", "column", "circular"]
    if pattern not in valid_patterns:
        return _error_response(f"Pattern must be one of: {valid_patterns}", "invalid_pattern")

    center = None
    if center_x is not None and center_y is not None:
        center = (center_x, center_y)

    session.checkpoint(f"Arrange {len(refs)} in {pattern}")
    actions = LayoutActions(session.board)
    result = actions.arrange_pattern(refs, pattern, spacing, cols, radius, center)

    if result.success:
        session.mark_modified(result.modified_refs)
        logger.debug("Arrange pattern successful", extra={"modified": len(result.modified_refs)})
    else:
        logger.warning("Arrange pattern failed", extra={"reason": result.message})

    return _success_response({"success": result.success, "message": result.message})


@mcp.tool()
def cluster_around(
    anchor_ref: str,
    target_refs: List[str],
    side: str = "nearest",
    clearance: float = 0.5,
) -> str:
    """
    Cluster components around an anchor on a specified side.

    Useful for grouping decoupling capacitors near an IC or
    organizing passives around a central component.

    Args:
        anchor_ref: Reference component to cluster around
        target_refs: List of components to cluster
        side: Side to cluster on ("top", "bottom", "left", "right", "nearest")
        clearance: Gap between anchor and clustered components in mm
    """
    logger.info("cluster_around called", extra={
        "anchor": anchor_ref, "targets": target_refs, "side": side
    })
    _require_board()

    # Validate inputs
    if err := _validate_ref(anchor_ref):
        return _error_response(err, "invalid_anchor")
    if err := _validate_refs(target_refs):
        return _error_response(err, "invalid_targets")

    valid_sides = ["top", "bottom", "left", "right", "nearest"]
    if side not in valid_sides:
        return _error_response(f"Side must be one of: {valid_sides}", "invalid_side")

    session.checkpoint(f"Cluster around {anchor_ref}")
    actions = LayoutActions(session.board)
    result = actions.cluster_around(anchor_ref, target_refs, side, clearance)

    if result.success:
        session.mark_modified(result.modified_refs)
        logger.debug("Cluster around successful", extra={"modified": len(result.modified_refs)})
    else:
        logger.warning("Cluster around failed", extra={"reason": result.message})

    return _success_response({"success": result.success, "message": result.message})


# =============================================================================
# Discovery Tools
# =============================================================================

@mcp.tool()
def find_components(query: str, filter_by: str = "ref") -> str:
    """
    Find components matching a query.

    Args:
        query: Search string
        filter_by: Field to search ("ref", "value", "footprint")

    Returns:
        List of matching components with their locations
    """
    logger.info("find_components called", extra={"query": query, "filter_by": filter_by})
    _require_board()

    valid_filters = ["ref", "value", "footprint"]
    if filter_by not in valid_filters:
        return _error_response(f"filter_by must be one of: {valid_filters}", "invalid_filter")

    matches = []
    query_lower = query.lower()

    for ref, comp in session.board.components.items():
        match = False
        if filter_by == "ref" and query_lower in ref.lower():
            match = True
        elif filter_by == "value" and query_lower in (comp.value or "").lower():
            match = True
        elif filter_by == "footprint" and query_lower in (comp.footprint or "").lower():
            match = True

        if match:
            matches.append({
                "ref": ref,
                "value": comp.value,
                "footprint": comp.footprint,
                "x": round(comp.x, 2),
                "y": round(comp.y, 2),
                "rotation": comp.rotation,
                "locked": comp.locked,
            })

    logger.debug("find_components result", extra={"count": len(matches)})
    return _success_response({"count": len(matches), "components": matches[:50]})


@mcp.tool()
def get_board_bounds() -> str:
    """
    Get the board boundary dimensions.

    Returns:
        Board bounding box and dimensions
    """
    logger.debug("get_board_bounds called")
    _require_board()
    macro = MacroContext(session.board)
    summary = macro.get_summary()

    return _success_response({
        "width": round(summary.board_width, 2),
        "height": round(summary.board_height, 2),
        "component_count": summary.component_count,
        "net_count": summary.net_count,
    })


@mcp.tool()
def get_unplaced_components() -> str:
    """
    Get list of components that appear to be outside the board area.

    Returns:
        List of unplaced component references
    """
    logger.debug("get_unplaced_components called")
    _require_board()
    macro = MacroContext(session.board)
    summary = macro.get_summary()

    # Find components outside normal bounds
    unplaced = []
    for ref, comp in session.board.components.items():
        # Components at origin or very far from center are likely unplaced
        if abs(comp.x) > 500 or abs(comp.y) > 500:
            unplaced.append(ref)

    return _success_response({
        "count": len(unplaced),
        "refs": unplaced[:100],
        "estimated_unplaced": summary.unplaced_count,
    })


# =============================================================================
# Topology Tools
# =============================================================================

@mcp.tool()
def get_connected_components(ref: str) -> str:
    """
    Get all components connected to a reference via nets.

    Args:
        ref: Component reference to analyze

    Returns:
        Dictionary of net names to connected component lists
    """
    logger.info("get_connected_components called", extra={"ref": ref})
    _require_board()

    # Validate input
    if err := _validate_ref(ref):
        return _error_response(err, "invalid_ref")

    comp = session.board.components.get(ref)
    connections = {}

    for pad in comp.pads:
        if not pad.net:
            continue

        net = session.board.nets.get(pad.net)
        if not net:
            continue

        connected_refs = []
        for comp_ref, _ in net.connections:
            if comp_ref and comp_ref != ref:
                if comp_ref not in connected_refs:
                    connected_refs.append(comp_ref)

        if connected_refs:
            connections[pad.net] = connected_refs

    logger.debug("get_connected_components result", extra={"net_count": len(connections)})
    return _success_response({"ref": ref, "connections": connections})


@mcp.tool()
def get_critical_nets() -> str:
    """
    Get nets that require special routing attention (power, high-speed).

    Returns:
        List of critical nets with their properties
    """
    logger.debug("get_critical_nets called")
    _require_board()
    macro = MacroContext(session.board)
    summary = macro.get_summary()

    critical = []

    # Power nets
    for net_name in summary.power_nets:
        critical.append({"name": net_name, "type": "power", "priority": "high"})

    # Ground nets
    for net_name in summary.ground_nets:
        critical.append({"name": net_name, "type": "ground", "priority": "high"})

    # High fanout nets
    for net_stat in summary.high_fanout_nets:
        if net_stat.name not in summary.power_nets + summary.ground_nets:
            critical.append({
                "name": net_stat.name,
                "type": "high_fanout",
                "pad_count": net_stat.pad_count,
                "priority": "medium"
            })

    return _success_response({"critical_nets": critical})


# =============================================================================
# Context Tools
# =============================================================================

@mcp.tool()
def inspect_region(refs: List[str], padding: float = 5.0, include_image: bool = False) -> str:
    """
    Get detailed geometric data for a set of components.

    Args:
        refs: List of component references to inspect
        padding: Padding around the region in mm
        include_image: If True, includes SVG visualization in response

    Returns:
        Detailed JSON with positions, bounding boxes, and gap analysis
    """
    logger.info("inspect_region called", extra={"refs": refs, "padding": padding})
    _require_board()

    # Validate refs (allow partial matches for inspection)
    if refs:
        missing = [r for r in refs if r not in session.board.components]
        if missing:
            logger.warning("Some refs not found", extra={"missing": missing})

    microscope = Microscope(session.board)
    data = microscope.inspect_region(refs, padding)
    result = json.loads(data.to_json())
    
    if include_image:
        vision = VisionContext(session.board)
        # Use existing render logic but keep it minimal
        image = vision.render_region(refs, show_dimensions=True)
        result["image_svg"] = image.svg_content
        
    return json.dumps(result)


@mcp.tool()
def get_board_summary() -> str:
    """
    Get executive summary of board state.

    Returns:
        High-level board statistics and critical issues
    """
    logger.debug("get_board_summary called")
    _require_board()
    macro = MacroContext(session.board)
    return macro.get_summary().to_json()


@mcp.tool()
def get_semantic_grid() -> str:
    """
    Get 3x3 semantic grid mapping of components.

    Returns:
        Mapping of zones to component lists
    """
    logger.debug("get_semantic_grid called")
    _require_board()
    macro = MacroContext(session.board)
    return macro.get_semantic_grid().to_json()


@mcp.tool()
def get_module_map() -> str:
    """
    Get hierarchical module structure of the board.

    Returns:
        Tree structure of functional modules
    """
    logger.debug("get_module_map called")
    _require_board()
    macro = MacroContext(session.board)
    return macro.get_module_map().to_json()


@mcp.tool()
def render_region(refs: List[str], show_dimensions: bool = True) -> str:
    """
    Generate an SVG visualization of a board region.

    Args:
        refs: Components to include in the view
        show_dimensions: Whether to show gap dimensions

    Returns:
        SVG content as string
    """
    logger.info("render_region called", extra={"refs": refs, "show_dimensions": show_dimensions})
    _require_board()
    vision = VisionContext(session.board)
    image = vision.render_region(refs, show_dimensions=show_dimensions)
    return _success_response({
        "svg": image.svg_content,
        "component_count": image.component_count,
        "annotations": image.annotations,
    })


# =============================================================================
# Validation Tools
# =============================================================================

@mcp.tool()
def check_overlaps(refs: Optional[List[str]] = None) -> str:
    """
    Check for component overlaps.

    Args:
        refs: Optional list of refs to check. If None, checks all.

    Returns:
        List of overlapping pairs with overlap amount
    """
    logger.info("check_overlaps called", extra={"refs": refs})
    _require_board()

    components = []
    if refs:
        components = [session.board.components[r] for r in refs if r in session.board.components]
    else:
        components = list(session.board.components.values())

    overlaps = []

    for i, c1 in enumerate(components):
        for c2 in components[i+1:]:
            # Simple AABB overlap check
            w1, h1 = c1.width, c1.height
            w2, h2 = c2.width, c2.height

            dx = abs(c1.x - c2.x)
            dy = abs(c1.y - c2.y)

            overlap_x = (w1/2 + w2/2) - dx
            overlap_y = (h1/2 + h2/2) - dy

            if overlap_x > 0 and overlap_y > 0:
                overlaps.append({
                    "refs": [c1.reference, c2.reference],
                    "overlap_x": round(overlap_x, 3),
                    "overlap_y": round(overlap_y, 3),
                })

    logger.debug("check_overlaps result", extra={"overlap_count": len(overlaps)})
    return _success_response({
        "overlap_count": len(overlaps),
        "overlaps": overlaps[:20],  # Limit output
    })


@mcp.tool()
def validate_placement() -> str:
    """
    Run full placement validation.

    Returns:
        Validation report with issues and recommendations
    """
    logger.info("validate_placement called")
    _require_board()

    issues = []

    # Check overlaps
    overlap_result = json.loads(check_overlaps())
    if overlap_result.get("overlap_count", 0) > 0:
        issues.append({
            "severity": "error",
            "type": "overlap",
            "message": f"{overlap_result['overlap_count']} overlapping component pairs",
        })

    # Check bounds
    macro = MacroContext(session.board)
    summary = macro.get_summary()

    if summary.unplaced_count > 0:
        issues.append({
            "severity": "warning",
            "type": "unplaced",
            "message": f"{summary.unplaced_count} components may be outside board bounds",
        })

    for issue in summary.critical_issues:
        issues.append({
            "severity": "warning",
            "type": "board",
            "message": issue,
        })

    error_count = len([i for i in issues if i["severity"] == "error"])
    is_valid = error_count == 0
    logger.info("validate_placement result", extra={"valid": is_valid, "issue_count": len(issues)})

    return _success_response({
        "valid": is_valid,
        "issue_count": len(issues),
        "issues": issues,
    })


# =============================================================================
# DRC Tools
# =============================================================================

@mcp.tool()
def run_drc(
    use_kicad: bool = True,
    dfm_profile: str = "jlcpcb_standard",
    severity_filter: str = "all"
) -> str:
    """
    Run Design Rule Check on the loaded board.

    This runs DRC using two backends:
    - atoplace: Python-based checks (placement-focused, always available)
    - kicad-cli: Native KiCad DRC (comprehensive, if kicad-cli is available)

    Args:
        use_kicad: Whether to use kicad-cli for native DRC (default: True)
        dfm_profile: DFM profile name (default: "jlcpcb_standard")
            Options: jlcpcb_standard, jlcpcb_advanced, oshpark_2layer, pcbway_standard
        severity_filter: Filter results by severity (default: "all")
            Options: "all", "errors", "warnings"

    Returns:
        DRC report with violations that can be actioned.
        Each violation has an "actionable" flag indicating if it can be auto-fixed.
    """
    logger.info("run_drc called", extra={
        "use_kicad": use_kicad,
        "dfm_profile": dfm_profile,
        "severity_filter": severity_filter
    })
    _require_board()

    from .drc import get_drc_runner, DRCFixer

    runner = get_drc_runner()

    # Get board path for kicad-cli
    board_path = session.source_path if hasattr(session, 'source_path') else None

    # Run combined DRC
    result = runner.run_combined_drc(
        board=session.board,
        board_path=board_path,
        dfm_profile=dfm_profile,
        use_kicad=use_kicad
    )

    # Filter by severity if requested
    violations = result.violations
    if severity_filter == "errors":
        violations = [v for v in violations if v.severity == "error"]
    elif severity_filter == "warnings":
        violations = [v for v in violations if v.severity == "warning"]

    # Count actionable violations
    actionable_count = sum(1 for v in violations if v.actionable)

    logger.info("run_drc result", extra={
        "passed": result.passed,
        "error_count": result.error_count,
        "warning_count": result.warning_count
    })

    return _success_response({
        "passed": result.passed,
        "summary": result.summary,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "violation_count": len(violations),
        "actionable_count": actionable_count,
        "violations": [v.to_dict() for v in violations],
    })


@mcp.tool()
def get_drc_details(violation_id: str) -> str:
    """
    Get detailed information about a specific DRC violation.

    Args:
        violation_id: The violation ID from run_drc results (e.g., "v001")

    Returns:
        Detailed violation information including suggested fix actions.
    """
    logger.info("get_drc_details called", extra={"violation_id": violation_id})

    from .drc import get_drc_runner

    runner = get_drc_runner()
    violation = runner.get_violation(violation_id)

    if not violation:
        return _error_response(
            f"Violation '{violation_id}' not found. Run run_drc first.",
            "violation_not_found"
        )

    # Add extra context
    details = violation.to_dict()

    # If items reference components, add their positions
    if violation.items and session.is_loaded:
        component_details = []
        for ref in violation.items:
            comp = session.board.components.get(ref)
            if comp:
                component_details.append({
                    "ref": ref,
                    "x": round(comp.x, 3),
                    "y": round(comp.y, 3),
                    "rotation": comp.rotation,
                    "locked": comp.locked,
                })
        details["component_details"] = component_details

    return _success_response(details)


@mcp.tool()
def fix_drc_violation(
    violation_id: str,
    strategy: str = "auto"
) -> str:
    """
    Attempt to automatically fix a DRC violation.

    This works by moving components to resolve spacing issues. Not all
    violations can be auto-fixed (e.g., routing issues, footprint problems).

    Args:
        violation_id: The violation ID to fix (e.g., "v001")
        strategy: Fix strategy to use:
            - "auto": Automatically choose best strategy (default)
            - "move_first": Move the first component in the violation
            - "move_second": Move the second component
            - "spread": Move both components apart equally

    Returns:
        Result of the fix attempt, including any component moves made.
    """
    logger.info("fix_drc_violation called", extra={
        "violation_id": violation_id,
        "strategy": strategy
    })
    _require_board()

    from .drc import get_drc_runner, DRCFixer

    runner = get_drc_runner()
    violation = runner.get_violation(violation_id)

    if not violation:
        return _error_response(
            f"Violation '{violation_id}' not found. Run run_drc first.",
            "violation_not_found"
        )

    if not violation.actionable:
        return _error_response(
            f"Violation '{violation_id}' ({violation.rule}) cannot be auto-fixed. "
            f"Reason: {violation.suggested_action or 'Not a placement issue'}",
            "not_actionable"
        )

    # Validate strategy
    valid_strategies = ["auto", "move_first", "move_second", "spread"]
    if strategy not in valid_strategies:
        return _error_response(
            f"Invalid strategy '{strategy}'. Use one of: {valid_strategies}",
            "invalid_strategy"
        )

    # Attempt fix
    fixer = DRCFixer(session.board)
    success, message, updates = fixer.fix_violation(violation, strategy)

    if not success:
        return _error_response(message, "fix_failed")

    # Apply the updates
    applied_updates = []
    for update in updates:
        ref = update["ref"]
        new_x = update.get("x")
        new_y = update.get("y")

        # Use session to update (supports kipy real-time mode)
        if hasattr(session, 'update_component'):
            session.update_component(ref, x=new_x, y=new_y)
        else:
            # Direct mode - update board
            comp = session.board.components.get(ref)
            if comp:
                if new_x is not None:
                    comp.x = new_x
                if new_y is not None:
                    comp.y = new_y
                session.mark_modified([ref])

        applied_updates.append({
            "ref": ref,
            "new_x": round(new_x, 3) if new_x else None,
            "new_y": round(new_y, 3) if new_y else None,
        })

    logger.info("fix_drc_violation result", extra={
        "success": True,
        "updates_count": len(applied_updates)
    })

    return _success_response({
        "fixed": True,
        "message": message,
        "violation_id": violation_id,
        "updates": applied_updates,
        "hint": "Run run_drc() again to verify the fix"
    })


# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("board://summary")
def board_summary_resource() -> str:
    """Board summary as a resource."""
    if not session.is_loaded:
        return _error_response("No board loaded", "no_board")
    return get_board_summary()


@mcp.resource("board://modules")
def board_modules_resource() -> str:
    """Module map as a resource."""
    if not session.is_loaded:
        return _error_response("No board loaded", "no_board")
    return get_module_map()


@mcp.resource("board://components")
def board_components_resource() -> str:
    """List of all components as a resource."""
    if not session.is_loaded:
        return _error_response("No board loaded", "no_board")

    components = []
    for ref, comp in session.board.components.items():
        components.append({
            "ref": ref,
            "value": comp.value,
            "footprint": comp.footprint,
            "x": round(comp.x, 2),
            "y": round(comp.y, 2),
            "rotation": comp.rotation,
            "locked": comp.locked,
        })

    return _success_response({"component_count": len(components), "components": components})


@mcp.resource("board://nets")
def board_nets_resource() -> str:
    """Net connectivity as a resource."""
    if not session.is_loaded:
        return _error_response("No board loaded", "no_board")

    nets = []
    for name, net in session.board.nets.items():
        nets.append({
            "name": name,
            "code": net.code,
            "connection_count": len(net.connections),
            "is_power": net.is_power,
            "is_ground": net.is_ground,
            "components": list(net.get_component_refs())[:20],  # Limit for readability
        })

    return _success_response({"net_count": len(nets), "nets": nets})


@mcp.resource("board://session")
def board_session_resource() -> str:
    """Current session state as a resource."""
    stats = session.get_stats()
    return _success_response(stats)


@mcp.resource("prompts://system")
def system_prompt_resource() -> str:
    """The system prompt for LLM context."""
    return SYSTEM_PROMPT


@mcp.resource("prompts://fix_overlaps")
def fix_overlaps_prompt_resource() -> str:
    """Prompt for overlap resolution."""
    return FIX_OVERLAPS_PROMPT


# =============================================================================
# Entry Point
# =============================================================================

def configure_session(use_ipc: bool = False, socket_path: str = None):
    """
    Configure the session mode.

    Args:
        use_ipc: Use IPC mode to communicate with KiCad bridge
        socket_path: Override default socket path for IPC
    """
    global session

    if use_ipc:
        from .ipc_session import IPCSession
        session = IPCSession(socket_path) if socket_path else IPCSession()
        logger.info("Configured MCP server for IPC mode")
    else:
        session = Session()
        logger.info("Configured MCP server for direct mode")


def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AtoPlace MCP Server - AI-powered PCB placement"
    )
    parser.add_argument(
        "--ipc", "-i",
        action="store_true",
        help="Use IPC mode to communicate with KiCad bridge"
    )
    parser.add_argument(
        "--socket", "-s",
        default=None,
        help="Unix socket path for IPC communication"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not MCP_AVAILABLE:
        logger.error("MCP package not installed. Install with: pip install mcp")
        sys.exit(1)

    # Configure session based on args or environment
    use_ipc = args.ipc or _USE_IPC
    socket_path = args.socket or _IPC_SOCKET

    if use_ipc:
        configure_session(use_ipc=True, socket_path=socket_path)
        logger.info("Starting MCP server in IPC mode")
    else:
        logger.info("Starting MCP server in direct mode")

    mcp.run()


if __name__ == "__main__":
    main()