"""
AtoPlace MCP Server

Exposes the Layout DSL and Context Generators to LLM agents via
the Model Context Protocol (MCP).

Supports multiple backends:
- kipy: Live KiCad 9+ IPC for real-time component manipulation
- ipc: Bridge-based IPC to pcbnew process
- rpc: Subprocess RPC via stdin/stdout
- direct: Direct pcbnew access (KiCad Python environment)

Set via ATOPLACE_BACKEND env var. Default: kipy (with fallback chain)

Tools are organized into categories:
- Board Management: load, save, undo/redo
- Placement Actions: move, place_next_to, align, etc.
- Discovery: find components, get bounds, list unplaced
- Topology: get connections, find critical nets
- Context: inspect region, get summary, render view
- Validation: check overlaps, validate placement, DRC
"""

from pathlib import Path
from typing import List, Optional
import json
import logging
import os

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
        def run(self): print("MCP not installed")

from ..api.actions import LayoutActions
from ..api.session import Session
from .context.micro import Microscope
from .context.macro import MacroContext
from .context.vision import VisionContext
from .prompts import SYSTEM_PROMPT, FIX_OVERLAPS_PROMPT


# Initialize FastMCP server
mcp = FastMCP("atoplace")

# Session state - supports multiple backends
# Prefer KIPY (live KiCad 9+), fall back to RPC, then IPC, then direct
from .backends import create_session_with_fallback, get_backend_mode, BackendMode, BackendNotAvailableError

# Determine preferred backend from environment
preferred = BackendMode.KIPY  # Default to KIPY for bleeding-edge
env_backend = os.environ.get("ATOPLACE_BACKEND", "").lower()
if env_backend == "rpc":
    preferred = BackendMode.DIRECT  # RPC uses direct mode internally
elif env_backend == "ipc":
    preferred = BackendMode.IPC
elif env_backend == "direct":
    preferred = BackendMode.DIRECT

try:
    session, actual_mode = create_session_with_fallback(preferred)
    logger.info("MCP server using %s mode", actual_mode.value)
except BackendNotAvailableError as e:
    logger.error("No backends available: %s. Starting with empty session.", e)
    session = Session()
    actual_mode = BackendMode.DIRECT


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


def _require_board():
    """Raise error if no board is loaded."""
    if not session.is_loaded:
        raise ValueError("No board loaded. Call load_board first.")


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
    if not session.board:
        return "No board loaded"
    missing = [r for r in refs if r not in session.board.components]
    if missing:
        return f"Components not found: {missing[:5]}"
    return None


# =============================================================================
# Board Management Tools
# =============================================================================

@mcp.tool()
def load_board(path: str) -> str:
    """
    Load a PCB file for editing.

    In kipy mode, this connects to the currently open board in KiCad.
    In other modes, loads the specified file.
    """
    try:
        session.load(Path(path))
        logger.info("Loaded board: %d components, %d nets",
                   len(session.board.components), len(session.board.nets))
        return _success_response({
            "component_count": len(session.board.components),
            "net_count": len(session.board.nets),
            "backend": actual_mode.value
        })
    except Exception as e:
        logger.error("Load failed: %s", e)
        return _error_response(str(e), "load_failed")


@mcp.tool()
def save_board(path: Optional[str] = None) -> str:
    """
    Save the modified board.

    Args:
        path: Optional output path. If not provided, uses default behavior
              for the current backend.

    Backend-specific behavior:
    - KIPY mode: Changes synced to KiCad. Default save works (user saves with
                 Ctrl+S/Cmd+S). Explicit path raises error - use KiCad's
                 "File > Save As" menu instead.
    - IPC/Direct modes: Explicit path supported for Save As functionality.

    In kipy mode without path, changes are synced and user saves via Ctrl+S.
    In other modes, writes to the specified path or default .placed path.
    """
    try:
        _require_board()
        output_path = session.save(Path(path) if path else None)
        logger.info("Saved board to: %s", output_path)
        return _success_response({"path": str(output_path)})
    except Exception as e:
        logger.error("Save failed: %s", e)
        return _error_response(str(e), "save_failed")


@mcp.tool()
def undo() -> str:
    """Undo last operation (uses KiCad's undo in kipy mode)."""
    try:
        _require_board()
        if session.undo():
            return _success_response({"message": "Undo successful"})
        else:
            return _error_response("No operations to undo", "undo_failed")
    except Exception as e:
        return _error_response(str(e), "undo_failed")


@mcp.tool()
def redo() -> str:
    """Redo last undone operation (uses KiCad's redo in kipy mode)."""
    try:
        _require_board()
        if session.redo():
            return _success_response({"message": "Redo successful"})
        else:
            return _error_response("No operations to redo", "redo_failed")
    except Exception as e:
        return _error_response(str(e), "redo_failed")


# =============================================================================
# Placement Action Tools
# =============================================================================

@mcp.tool()
def move_absolute(ref: str, x: float, y: float, rotation: Optional[float] = None) -> str:
    """
    Move component to absolute coordinates.

    In kipy mode, changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_ref(ref):
            return _error_response(err, "invalid_ref")

        session.checkpoint(f"Move {ref} to ({x:.2f}, {y:.2f})")
        actions = LayoutActions(session.board)
        result = actions.move_absolute(ref, x, y, rotation)

        if result.success:
            session.mark_modified(result.modified_refs)
            logger.debug("Move absolute successful: %s", result.message)
        else:
            logger.warning("Move absolute failed: %s", result.message)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        logger.error("Move absolute error: %s", e)
        return _error_response(str(e), "action_failed")


@mcp.tool()
def move_relative(ref: str, dx: float, dy: float) -> str:
    """
    Move component by a relative delta.

    In kipy mode, changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_ref(ref):
            return _error_response(err, "invalid_ref")

        session.checkpoint(f"Move {ref} by ({dx:.2f}, {dy:.2f})")
        actions = LayoutActions(session.board)
        result = actions.move_relative(ref, dx, dy)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def rotate(ref: str, angle: float) -> str:
    """
    Set absolute rotation of component.

    In kipy mode, changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_ref(ref):
            return _error_response(err, "invalid_ref")

        session.checkpoint(f"Rotate {ref} to {angle}°")
        actions = LayoutActions(session.board)
        result = actions.rotate(ref, angle)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def place_next_to(
    ref: str, target: str, side: str = "right", clearance: float = 0.5, align: str = "center"
) -> str:
    """
    Place a component next to another with specified clearance.

    In kipy mode, changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_ref(ref):
            return _error_response(err, "invalid_ref")
        if err := _validate_ref(target):
            return _error_response(err, "invalid_target")

        session.checkpoint(f"Place {ref} next to {target}")
        actions = LayoutActions(session.board)
        result = actions.place_next_to(ref, target, side, clearance, align)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def align_components(refs: List[str], axis: str = "x", anchor: str = "first") -> str:
    """
    Align components along an axis.

    In kipy mode, all changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_refs(refs):
            return _error_response(err, "invalid_refs")

        session.checkpoint(f"Align {len(refs)} components")
        actions = LayoutActions(session.board)
        result = actions.align_components(refs, axis, anchor)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def distribute_evenly(
    refs: List[str], start_ref: Optional[str] = None, end_ref: Optional[str] = None, axis: str = "auto"
) -> str:
    """
    Distribute components evenly between two points.

    In kipy mode, all changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_refs(refs):
            return _error_response(err, "invalid_refs")

        session.checkpoint(f"Distribute {len(refs)} components")
        actions = LayoutActions(session.board)
        result = actions.distribute_evenly(refs, start_ref, end_ref, axis)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def stack_components(
    refs: List[str], direction: str = "down", spacing: float = 0.5, alignment: str = "center"
) -> str:
    """
    Stack components sequentially in a direction.

    In kipy mode, all changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_refs(refs):
            return _error_response(err, "invalid_refs")

        session.checkpoint(f"Stack {len(refs)} components")
        actions = LayoutActions(session.board)
        result = actions.stack_components(refs, direction, spacing, alignment)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def lock_components(refs: List[str], locked: bool = True) -> str:
    """
    Lock or unlock components to prevent accidental movement.

    In kipy mode, lock state syncs to KiCad.
    """
    try:
        _require_board()
        if err := _validate_refs(refs):
            return _error_response(err, "invalid_refs")

        session.checkpoint(f"Lock {len(refs)} components")
        actions = LayoutActions(session.board)
        result = actions.lock_components(refs, locked)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
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
    """
    Arrange components in a pattern (grid, row, column, or circular).

    In kipy mode, all changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_refs(refs):
            return _error_response(err, "invalid_refs")

        session.checkpoint(f"Arrange {len(refs)} components in {pattern}")
        actions = LayoutActions(session.board)

        # Combine center_x and center_y into tuple if both provided
        center = None
        if center_x is not None and center_y is not None:
            center = (center_x, center_y)

        result = actions.arrange_pattern(refs, pattern, spacing, cols, radius, center)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


@mcp.tool()
def cluster_around(
    anchor_ref: str, target_refs: List[str], side: str = "nearest", clearance: float = 0.5
) -> str:
    """
    Cluster components around an anchor on a specified side.

    Useful for grouping decoupling capacitors near an IC.
    In kipy mode, all changes appear instantly in KiCad viewport.
    """
    try:
        _require_board()
        if err := _validate_ref(anchor_ref):
            return _error_response(err, "invalid_anchor")
        if err := _validate_refs(target_refs):
            return _error_response(err, "invalid_targets")

        session.checkpoint(f"Cluster {len(target_refs)} around {anchor_ref}")
        actions = LayoutActions(session.board)
        result = actions.cluster_around(anchor_ref, target_refs, side, clearance)

        if result.success:
            session.mark_modified(result.modified_refs)

        return _success_response({"success": result.success, "message": result.message})
    except Exception as e:
        return _error_response(str(e), "action_failed")


# =============================================================================
# Context/Inspection Tools
# =============================================================================

@mcp.tool()
def inspect_region(refs: List[str], padding: float = 5.0, include_image: bool = False) -> str:
    """
    Get detailed geometric data for a set of components.

    Returns pad positions, bounding boxes, gap analysis, and optionally SVG visualization.
    """
    try:
        _require_board()
        if err := _validate_refs(refs):
            return _error_response(err, "invalid_refs")

        micro = Microscope(session.board)
        data = micro.inspect_region(refs, padding)
        result = json.loads(data.to_json())

        if include_image:
            vision = VisionContext(session.board)
            image = vision.render_region(refs, show_dimensions=True)
            result["image_svg"] = image.svg_content

        return json.dumps(result)
    except Exception as e:
        logger.error("Inspect region failed: %s", e)
        return _error_response(str(e), "inspect_failed")


@mcp.tool()
def get_board_summary() -> str:
    """Get executive summary of board state."""
    try:
        _require_board()
        macro = MacroContext(session.board)
        return json.dumps(json.loads(macro.get_summary().to_json()))
    except Exception as e:
        return _error_response(str(e), "summary_failed")


@mcp.tool()
def check_overlaps(refs: Optional[List[str]] = None) -> str:
    """Check for component overlaps."""
    try:
        _require_board()

        components = []
        if refs:
            if err := _validate_refs(refs):
                return _error_response(err, "invalid_refs")
            components = [session.board.components[r] for r in refs]
        else:
            components = list(session.board.components.values())

        overlaps = []
        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
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

        return _success_response({
            "overlap_count": len(overlaps),
            "overlaps": overlaps[:20]
        })
    except Exception as e:
        return _error_response(str(e), "check_failed")


# =============================================================================
# Discovery Tools
# =============================================================================

@mcp.tool()
def get_unplaced_components() -> str:
    """Get list of components that are outside the board area."""
    try:
        _require_board()
        macro = MacroContext(session.board)
        unplaced = macro.get_unplaced_components()
        return json.dumps({"count": len(unplaced), "refs": unplaced[:50]})
    except Exception as e:
        return _error_response(str(e), "discovery_failed")


@mcp.tool()
def find_components(query: str, filter_by: str = "ref") -> str:
    """
    Find components matching a query.

    Args:
        query: Search string
        filter_by: Field to search ("ref", "value", "footprint")
    """
    try:
        _require_board()
        matches = []
        for ref, comp in session.board.components.items():
            search_value = ""
            if filter_by == "ref":
                search_value = ref
            elif filter_by == "value":
                search_value = comp.value
            elif filter_by == "footprint":
                search_value = comp.footprint

            if query.lower() in search_value.lower():
                matches.append({
                    "ref": ref,
                    "value": comp.value,
                    "footprint": comp.footprint,
                    "x": comp.x,
                    "y": comp.y
                })

        return json.dumps({"count": len(matches), "matches": matches[:50]})
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
    """
    Run Design Rule Check on the loaded board.

    Args:
        use_kicad: Whether to use KiCad's native DRC (if available)
        dfm_profile: DFM profile ("jlcpcb_standard", "jlcpcb_advanced", etc.)
        severity_filter: Filter by severity ("all", "errors", "warnings")
    """
    try:
        _require_board()
        from ..validation.drc import DRCChecker
        from ..dfm.profiles import get_profile

        dfm = get_profile(dfm_profile)
        checker = DRCChecker(session.board, dfm)
        passed, violations = checker.run_checks()

        # Filter by severity
        if severity_filter != "all":
            violations = [v for v in violations if v.severity == severity_filter]

        return json.dumps({
            "violation_count": len(violations),
            "violations": [{
                "rule": v.rule,
                "severity": v.severity,
                "message": v.message,
                "items": v.items,
                "location": {"x": v.location[0], "y": v.location[1]}
            } for v in violations[:50]]
        })
    except Exception as e:
        logger.error("DRC failed: %s", e)
        return _error_response(str(e), "drc_failed")


@mcp.tool()
def validate_placement() -> str:
    """Run placement validation and get confidence report."""
    try:
        _require_board()
        from ..validation.confidence import ConfidenceScorer

        scorer = ConfidenceScorer()  # Uses default DFM profile
        report = scorer.assess(session.board)

        return json.dumps({
            "overall_score": report.overall_score,
            "placement_score": report.placement_score,
            "routing_score": report.routing_score,
            "dfm_score": report.dfm_score,
            "electrical_score": report.electrical_score,
            "flags": [{"category": f.category.value, "severity": f.severity.value, "message": f.message} for f in report.flags[:20]],
            "component_count": report.component_count,
            "net_count": report.net_count
        })
    except Exception as e:
        return _error_response(str(e), "validation_failed")


# =============================================================================
# Advanced Placement Tools
# =============================================================================

@mcp.tool()
def optimize_placement(
    constraints: Optional[List[str]] = None,
    iterations: int = 100,
    enable_modules: bool = True
) -> str:
    """
    Run force-directed placement optimization with optional constraints.

    This uses atoplace's sophisticated physics-based placement algorithm
    that considers:
    - Component connectivity (minimizes wire length)
    - Component repulsion (prevents overlaps)
    - Board boundaries (keeps components in bounds)
    - User constraints (proximity, edge placement, grouping, etc.)
    - Module detection (groups related components)

    Args:
        constraints: List of natural language constraints, e.g.:
                    ["Keep C1 close to U1", "USB connector on left edge"]
        iterations: Number of optimization iterations (default: 100)
        enable_modules: Auto-detect and group functional modules (default: True)

    Returns:
        JSON with optimization results and metrics
    """
    try:
        _require_board()
        from ..placement.force_directed import ForceDirectedRefiner, RefinementConfig
        from ..placement.module_detector import ModuleDetector
        from ..nlp.constraint_parser import ConstraintParser

        # Detect modules if enabled
        modules = None
        if enable_modules:
            detector = ModuleDetector(session.board)
            detected_modules = detector.detect()
            modules = {ref: mod.module_type.value for mod in detected_modules for ref in mod.components}
            logger.info("Detected %d modules with %d components", len(detected_modules), len(modules))

        # Parse constraints
        parsed_constraints = []
        if constraints:
            parser = ConstraintParser(session.board)
            for constraint_text in constraints:
                result = parser.parse(constraint_text)
                if result.constraints:
                    parsed_constraints.extend(result.constraints)
                    logger.info("Parsed constraint: %s", constraint_text)

        # Create checkpoint before optimization
        session.checkpoint("Placement optimization")

        # Run force-directed refinement
        config = RefinementConfig(max_iterations=iterations)
        refiner = ForceDirectedRefiner(session.board, config=config, modules=modules)

        # Add parsed constraints
        for parsed_constraint in parsed_constraints:
            refiner.add_constraint(parsed_constraint.constraint)

        # Run optimization
        refiner.refine()

        # Mark all components as modified
        modified_refs = list(session.board.components.keys())
        session.mark_modified(modified_refs)

        return _success_response({
            "success": True,
            "iterations_run": iterations,
            "modules_detected": len(set(modules.values())) if modules else 0,
            "constraints_applied": len(parsed_constraints),
            "components_optimized": len(modified_refs),
            "message": f"Optimized placement for {len(modified_refs)} components"
        })
    except Exception as e:
        logger.error("Placement optimization failed: %s", e, exc_info=True)
        return _error_response(str(e), "optimization_failed")


@mcp.tool()
def detect_modules() -> str:
    """
    Detect functional modules in the board using connectivity and heuristics.

    Identifies module types like:
    - Microcontrollers and their support circuits
    - Power regulators and decoupling
    - RF frontends and matching networks
    - Sensors and signal conditioning
    - ESD protection circuits
    - Crystal oscillators

    Returns:
        JSON with detected modules and their components
    """
    try:
        _require_board()
        from ..placement.module_detector import ModuleDetector

        detector = ModuleDetector(session.board)
        modules = detector.detect()

        result = []
        for module in modules:
            result.append({
                "name": module.name,
                "type": module.module_type.value,
                "components": list(module.components),
                "primary_component": module.primary_component,
                "priority": module.priority,
                "placement_hints": module.placement_hints
            })

        return json.dumps({
            "module_count": len(modules),
            "modules": result
        }, indent=2)
    except Exception as e:
        logger.error("Module detection failed: %s", e, exc_info=True)
        return _error_response(str(e), "detection_failed")


@mcp.tool()
def parse_constraint(text: str) -> str:
    """
    Parse natural language placement constraint into structured form.

    Supported constraint types:
    - Proximity: "Keep C1 close to U1", "Place C1 within 5mm of U1"
    - Edge: "USB connector on left edge", "J1 at top edge"
    - Zone: "Keep RF components in top-left", "Analog in zone (10,10,50,50)"
    - Grouping: "Group all decoupling capacitors", "Keep power components together"
    - Separation: "Keep analog and digital separate", "Separate RF from MCU"
    - Fixed: "Lock U1 in place", "Fix connector positions"

    Args:
        text: Natural language constraint description

    Returns:
        JSON with parsed constraint details and confidence
    """
    try:
        _require_board()
        from ..nlp.constraint_parser import ConstraintParser

        parser = ConstraintParser(session.board)
        result = parser.parse(text)

        constraints_data = []
        for parsed_constraint in result.constraints:
            # Get the actual constraint object
            constraint = parsed_constraint.constraint

            constraint_dict = {
                "type": constraint.constraint_type.value,
                "confidence": parsed_constraint.confidence.value,
                "source_text": parsed_constraint.source_text,
                "description": str(constraint)
            }

            # Add type-specific fields
            if hasattr(constraint, "components"):
                constraint_dict["components"] = list(constraint.components)
            if hasattr(constraint, "edge"):
                constraint_dict["edge"] = constraint.edge.value if hasattr(constraint.edge, 'value') else str(constraint.edge)
            if hasattr(constraint, "distance"):
                constraint_dict["distance"] = constraint.distance
            if hasattr(constraint, "zone"):
                constraint_dict["zone"] = constraint.zone
            if hasattr(constraint, "target"):
                constraint_dict["target"] = constraint.target
            if hasattr(constraint, "reference"):
                constraint_dict["reference"] = constraint.reference

            constraints_data.append(constraint_dict)

        return json.dumps({
            "original_text": text,
            "parsed_count": len(result.constraints),
            "constraints": constraints_data,
            "unrecognized_text": result.unrecognized_text,
            "warnings": result.warnings
        }, indent=2)
    except Exception as e:
        logger.error("Constraint parsing failed: %s", e, exc_info=True)
        return _error_response(str(e), "parse_failed")


@mcp.tool()
def get_atopile_context(ato_path: Optional[str] = None) -> str:
    """
    Load atopile project context to understand design intent.

    Extracts:
    - Module hierarchy and relationships
    - Component metadata (tolerances, ratings, part numbers)
    - Interface definitions (power, signals, connectors)
    - Design intent and constraints from ato files

    Args:
        ato_path: Path to .ato file or project directory
                 If None, tries to auto-detect from board path

    Returns:
        JSON with atopile project context
    """
    try:
        _require_board()
        from ..board.atopile_adapter import (
            AtopileProjectLoader,
            AtopileModuleParser,
            detect_board_source
        )

        # Auto-detect atopile source if not provided
        if ato_path is None and session.source_path:
            try:
                # Try to find project root from the loaded board path
                project_root = AtopileProjectLoader.find_project_root(session.source_path)
                if project_root:
                    ato_path = str(project_root)
                    logger.info("Auto-detected atopile project root: %s", ato_path)
                else:
                    return _error_response(
                        "No atopile project found. Provide ato_path parameter.",
                        "no_atopile_source"
                    )
            except Exception as e:
                logger.warning("Could not auto-detect atopile source: %s", e)
                return _error_response(
                    "No atopile source found. Provide ato_path parameter.",
                    "no_atopile_source"
                )

        if not ato_path:
            return _error_response(
                "No atopile path provided and auto-detection failed",
                "missing_path"
            )

        # Load atopile project
        project_path = Path(ato_path)
        if not project_path.exists():
            return _error_response(
                f"Path does not exist: {ato_path}",
                "path_not_found"
            )

        # Initialize loader
        loader = AtopileProjectLoader(project_path)

        # Get project configuration
        project = loader.project

        # Parse module hierarchy from .ato files
        modules_data = []
        ato_source = loader.get_ato_source_path()
        if ato_source and ato_source.exists():
            parser = AtopileModuleParser()
            module_hierarchy = parser.parse_file(ato_source)

            for module_path, module_info in module_hierarchy.items():
                modules_data.append({
                    "path": module_path,
                    "name": module_info.name,
                    "type": module_info.module_type or "unknown",
                    "components": module_info.components,
                    "submodules": len(module_info.submodules),
                    "parent": module_info.parent
                })

        # Extract component metadata from lock file
        components_metadata = {}
        lock_data = loader.lock_data
        if lock_data:
            components_section = lock_data.get("components", {})
            for ref, comp_data in components_section.items():
                if isinstance(comp_data, dict):
                    components_metadata[ref] = {
                        "value": comp_data.get("value", ""),
                        "mpn": comp_data.get("mpn", ""),
                        "package": comp_data.get("package", ""),
                        "manufacturer": comp_data.get("manufacturer", ""),
                        "description": comp_data.get("description", "")
                    }

        # Build response
        result = {
            "project_name": project_path.name,
            "project_root": str(loader.root),
            "ato_version": project.ato_version,
            "builds": list(project.builds.keys()),
            "module_count": len(modules_data),
            "modules": modules_data,
            "components_with_metadata": len(components_metadata),
            "component_metadata": components_metadata
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("Atopile context loading failed: %s", e, exc_info=True)
        return _error_response(str(e), "context_load_failed")


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

    logger.info("Starting AtoPlace MCP server with backend: %s", actual_mode.value)
    mcp.run()

if __name__ == "__main__":
    main()
