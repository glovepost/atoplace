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
from typing import List, Optional
import json

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

from ..board.abstraction import Board
from ..api.actions import LayoutActions
from ..api.session import Session
from .context.micro import Microscope
from .context.macro import MacroContext
from .context.vision import VisionContext


# Initialize FastMCP server
mcp = FastMCP("atoplace")

# Session state (manages board, undo/redo, dirty tracking)
session = Session()


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
    try:
        session.load(Path(path))
        board = session.board
        return json.dumps({
            "status": "loaded",
            "path": path,
            "components": len(board.components),
            "nets": len(board.nets),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def save_board(path: Optional[str] = None) -> str:
    """
    Save the modified board.

    Args:
        path: Output path. If None, saves to <original>.placed.kicad_pcb

    Returns:
        Path where board was saved
    """
    _require_board()
    try:
        output_path = session.save(Path(path) if path else None)
        return json.dumps({"status": "saved", "path": str(output_path)})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def undo() -> str:
    """Undo the last placement action."""
    _require_board()
    if session.undo():
        return json.dumps({"status": "undone"})
    return json.dumps({"status": "nothing_to_undo"})


@mcp.tool()
def redo() -> str:
    """Redo the last undone action."""
    _require_board()
    if session.redo():
        return json.dumps({"status": "redone"})
    return json.dumps({"status": "nothing_to_redo"})


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
    _require_board()
    session.checkpoint(f"Move {ref}")
    actions = LayoutActions(session.board)
    result = actions.move_absolute(ref, x, y, rotation)
    if result.success:
        session.mark_modified(result.modified_refs)
    return json.dumps({"success": result.success, "message": result.message})


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
    _require_board()
    session.checkpoint(f"Place {ref} next to {target}")
    actions = LayoutActions(session.board)
    result = actions.place_next_to(ref, target, side, clearance, align)
    if result.success:
        session.mark_modified(result.modified_refs)
    return json.dumps({"success": result.success, "message": result.message})


@mcp.tool()
def align_components(refs: List[str], axis: str = "x", anchor: str = "first") -> str:
    """
    Align multiple components along an axis.

    Args:
        refs: List of component references to align
        axis: Alignment axis ("x" for row, "y" for column)
        anchor: Reference point ("first", "last", "center")
    """
    _require_board()
    session.checkpoint(f"Align {len(refs)} components")
    actions = LayoutActions(session.board)
    result = actions.align_components(refs, axis, anchor)
    if result.success:
        session.mark_modified(result.modified_refs)
    return json.dumps({"success": result.success, "message": result.message})


@mcp.tool()
def swap_positions(ref1: str, ref2: str) -> str:
    """
    Swap the positions of two components.

    Args:
        ref1: First component reference
        ref2: Second component reference
    """
    _require_board()
    c1 = session.board.components.get(ref1)
    c2 = session.board.components.get(ref2)

    if not c1 or not c2:
        return json.dumps({"success": False, "message": "Component not found"})

    if c1.locked or c2.locked:
        return json.dumps({"success": False, "message": "Cannot swap locked components"})

    session.checkpoint(f"Swap {ref1} and {ref2}")

    # Swap positions
    c1.x, c2.x = c2.x, c1.x
    c1.y, c2.y = c2.y, c1.y

    session.mark_modified([ref1, ref2])
    return json.dumps({"success": True, "message": f"Swapped {ref1} and {ref2}"})


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
    _require_board()
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

    return json.dumps({"count": len(matches), "components": matches[:50]})


@mcp.tool()
def get_board_bounds() -> str:
    """
    Get the board boundary dimensions.

    Returns:
        Board bounding box and dimensions
    """
    _require_board()
    macro = MacroContext(session.board)
    summary = macro.get_summary()

    return json.dumps({
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
    _require_board()
    macro = MacroContext(session.board)
    summary = macro.get_summary()

    # Find components outside normal bounds
    unplaced = []
    for ref, comp in session.board.components.items():
        # Components at origin or very far from center are likely unplaced
        if abs(comp.x) > 500 or abs(comp.y) > 500:
            unplaced.append(ref)

    return json.dumps({
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
    _require_board()
    comp = session.board.components.get(ref)
    if not comp:
        return json.dumps({"error": f"Component {ref} not found"})

    connections = {}

    for pad in comp.pads:
        if not pad.net:
            continue

        net = session.board.nets.get(pad.net)
        if not net:
            continue

        connected_refs = []
        for net_pad in net.pads:
            if net_pad.component_ref and net_pad.component_ref != ref:
                if net_pad.component_ref not in connected_refs:
                    connected_refs.append(net_pad.component_ref)

        if connected_refs:
            connections[pad.net] = connected_refs

    return json.dumps({"ref": ref, "connections": connections})


@mcp.tool()
def get_critical_nets() -> str:
    """
    Get nets that require special routing attention (power, high-speed).

    Returns:
        List of critical nets with their properties
    """
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

    return json.dumps({"critical_nets": critical})


# =============================================================================
# Context Tools
# =============================================================================

@mcp.tool()
def inspect_region(refs: List[str], padding: float = 5.0) -> str:
    """
    Get detailed geometric data for a set of components.

    Args:
        refs: List of component references to inspect
        padding: Padding around the region in mm

    Returns:
        Detailed JSON with positions, bounding boxes, and gap analysis
    """
    _require_board()
    microscope = Microscope(session.board)
    data = microscope.inspect_region(refs, padding)
    return data.to_json()


@mcp.tool()
def get_board_summary() -> str:
    """
    Get executive summary of board state.

    Returns:
        High-level board statistics and critical issues
    """
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
    _require_board()
    vision = VisionContext(session.board)
    image = vision.render_region(refs, show_dimensions=show_dimensions)
    return json.dumps({
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

    return json.dumps({
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
    _require_board()

    issues = []

    # Check overlaps
    overlap_result = json.loads(check_overlaps())
    if overlap_result["overlap_count"] > 0:
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

    return json.dumps({
        "valid": len([i for i in issues if i["severity"] == "error"]) == 0,
        "issue_count": len(issues),
        "issues": issues,
    })


# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("board://summary")
def board_summary_resource() -> str:
    """Board summary as a resource."""
    if not session.is_loaded:
        return json.dumps({"error": "No board loaded"})
    return get_board_summary()


@mcp.resource("board://modules")
def board_modules_resource() -> str:
    """Module map as a resource."""
    if not session.is_loaded:
        return json.dumps({"error": "No board loaded"})
    return get_module_map()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP package not installed.")
        print("Install with: pip install mcp")
        return
    mcp.run()


if __name__ == "__main__":
    main()
