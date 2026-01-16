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
def _get_default_log_path() -> str:
    """Get secure default log file path.

    Uses user-specific directory to avoid world-readable /tmp.
    Falls back to /tmp if user directory is not available.
    """
    # Try user's cache directory first (not world-readable)
    home = Path.home()
    cache_dir = home / ".cache" / "atoplace"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Set restrictive permissions (owner only)
        cache_dir.chmod(0o700)
        return str(cache_dir / "mcp-server.log")
    except (OSError, PermissionError):
        # Fall back to /tmp with unique filename
        import getpass
        try:
            username = getpass.getuser()
        except Exception:
            username = "unknown"
        return f"/tmp/atoplace-{username}.log"


LOG_FILE = os.environ.get("ATOPLACE_LOG", _get_default_log_path())
_log_configured = False

def _configure_logging():
    """Configure logging once."""
    global _log_configured
    if _log_configured:
        return

    # Only add file handler if not already configured
    root = logging.getLogger()
    if not root.handlers:
        try:
            handler = logging.FileHandler(LOG_FILE, mode="a")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%H:%M:%S"
            ))
            root.addHandler(handler)
            root.setLevel(logging.INFO)

            # Try to set restrictive permissions on log file
            try:
                os.chmod(LOG_FILE, 0o600)
            except (OSError, PermissionError):
                pass  # Best effort - may fail on some systems
        except (OSError, PermissionError) as e:
            # If file logging fails, just skip it (MCP needs clean STDIO)
            pass
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
from ..api.inspection import BoardInspector
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


class SessionProvider:
    """
    Session provider for dependency injection.

    This class manages the MCP server session and allows tests to inject
    mock sessions without modifying the module's global state.

    Usage:
        # In tests, replace the session:
        from atoplace.mcp.server import session_provider
        session_provider.set_session(mock_session)

        # Reset after test:
        session_provider.reset()
    """

    def __init__(self):
        self._session: Optional[Session] = None
        self._actual_mode: Optional[BackendMode] = None
        self._initialized = False

    def _initialize(self):
        """Initialize session with backend fallback chain."""
        if self._initialized:
            return

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
            self._session, self._actual_mode = create_session_with_fallback(preferred)
            logger.info("MCP server using %s mode", self._actual_mode.value)
        except BackendNotAvailableError as e:
            logger.error("No backends available: %s. Starting with empty session.", e)
            self._session = Session()
            self._actual_mode = BackendMode.DIRECT

        self._initialized = True

    @property
    def session(self) -> Session:
        """Get the current session, initializing if needed."""
        self._initialize()
        return self._session

    @property
    def actual_mode(self) -> BackendMode:
        """Get the actual backend mode."""
        self._initialize()
        return self._actual_mode

    def set_session(self, new_session: Session, mode: BackendMode = BackendMode.DIRECT):
        """
        Replace the session (for testing).

        Args:
            new_session: The session to use
            mode: The backend mode to report
        """
        self._session = new_session
        self._actual_mode = mode
        self._initialized = True

    def reset(self):
        """Reset to uninitialized state (for testing)."""
        self._session = None
        self._actual_mode = None
        self._initialized = False


# Global session provider instance
session_provider = SessionProvider()

# For backward compatibility, expose session and actual_mode as module-level attributes
# These are properties that delegate to the session provider
class _SessionProxy:
    """Proxy that delegates to session_provider for backward compatibility."""
    def __getattr__(self, name):
        return getattr(session_provider.session, name)

    def __setattr__(self, name, value):
        setattr(session_provider.session, name, value)


session = _SessionProxy()


def _get_actual_mode() -> BackendMode:
    """Get actual backend mode from session provider."""
    return session_provider.actual_mode

# Initialize immediately (preserves original behavior)
# Can be deferred by setting ATOPLACE_LAZY_INIT=1
if not os.environ.get("ATOPLACE_LAZY_INIT"):
    session_provider._initialize()


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


# Valid values for side and axis parameters
VALID_SIDES = {"top", "bottom", "left", "right"}
VALID_AXES = {"x", "y"}


def _validate_side(side: str) -> Optional[str]:
    """Validate side parameter. Returns error message or None."""
    if side.lower() not in VALID_SIDES:
        return f"Invalid side '{side}'. Must be one of: top, bottom, left, right"
    return None


def _validate_axis(axis: str) -> Optional[str]:
    """Validate axis parameter. Returns error message or None."""
    if axis.lower() not in VALID_AXES:
        return f"Invalid axis '{axis}'. Must be one of: x, y"
    return None


# Path validation constants
# Allowed paths can be configured via environment variable
ALLOWED_PATH_ROOTS = os.environ.get("ATOPLACE_ALLOWED_PATHS", "").split(":") if os.environ.get("ATOPLACE_ALLOWED_PATHS") else []
BLOCKED_PATH_PATTERNS = [
    "/etc/", "/var/", "/usr/", "/bin/", "/sbin/",  # System directories
    "/.ssh/", "/.gnupg/", "/.aws/", "/.config/",   # Sensitive user directories
    "/credentials", "/secrets", "/private",         # Common sensitive paths
]

# Pagination constants for discovery tools
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def _validate_path(path: str, operation: str = "access") -> Optional[str]:
    """
    Validate file path for security.

    Checks:
    1. Path must exist or be in a valid parent directory
    2. Path must not contain traversal sequences
    3. Path must not access blocked directories
    4. If ATOPLACE_ALLOWED_PATHS is set, path must be under an allowed root

    Args:
        path: File path to validate
        operation: Operation being performed (for error messages)

    Returns:
        Error message if path is invalid, None if valid
    """
    try:
        # Resolve to absolute path, following symlinks
        resolved = Path(path).resolve()
        path_str = str(resolved).lower()

        # Check for path traversal attempts
        if ".." in path:
            return f"Path traversal not allowed: {path}"

        # Check for blocked patterns
        for pattern in BLOCKED_PATH_PATTERNS:
            if pattern.lower() in path_str:
                return f"Access to {pattern} is not allowed"

        # If allowed paths are configured, enforce them
        if ALLOWED_PATH_ROOTS:
            is_allowed = False
            for allowed in ALLOWED_PATH_ROOTS:
                if allowed and str(resolved).startswith(str(Path(allowed).resolve())):
                    is_allowed = True
                    break
            if not is_allowed:
                return f"Path not in allowed directories. Set ATOPLACE_ALLOWED_PATHS to configure."

        # For load operations, check file extension
        if operation == "load":
            suffix = resolved.suffix.lower()
            allowed_extensions = {".kicad_pcb", ".kicad_pcb-bak"}
            if suffix not in allowed_extensions and not resolved.is_dir():
                return f"Invalid file type '{suffix}'. Only .kicad_pcb files are supported."

        return None

    except Exception as e:
        return f"Invalid path: {e}"


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
        # Validate path for security
        if err := _validate_path(path, operation="load"):
            logger.warning("Path validation failed: %s", err)
            return _error_response(err, "invalid_path")

        session.load(Path(path))
        logger.info("Loaded board: %d components, %d nets",
                   len(session.board.components), len(session.board.nets))
        return _success_response({
            "component_count": len(session.board.components),
            "net_count": len(session.board.nets),
            "backend": _get_actual_mode().value
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

        # Validate path for security if provided
        if path:
            if err := _validate_path(path, operation="save"):
                logger.warning("Path validation failed: %s", err)
                return _error_response(err, "invalid_path")

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
            return _success_response({"action": "undone", "message": "Undo successful"})
        else:
            return _success_response({"action": "nothing_to_undo", "message": "No operations to undo"})
    except Exception as e:
        return _error_response(str(e), "undo_failed")


@mcp.tool()
def redo() -> str:
    """Redo last undone operation (uses KiCad's redo in kipy mode)."""
    try:
        _require_board()
        if session.redo():
            return _success_response({"action": "redone", "message": "Redo successful"})
        else:
            return _success_response({"action": "nothing_to_redo", "message": "No operations to redo"})
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


# Alias for backward compatibility with tests
def move_component(ref: str, x: float, y: float, rotation: Optional[float] = None) -> str:
    """Alias for move_absolute for backward compatibility."""
    return move_absolute(ref, x, y, rotation)


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
        if err := _validate_side(side):
            return _error_response(err, "invalid_side")

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
        if err := _validate_axis(axis):
            return _error_response(err, "invalid_axis")

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
def check_overlaps(
    refs: Optional[List[str]] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0
) -> str:
    """
    Check for component overlaps.

    Supports pagination for boards with many overlapping components.

    Args:
        refs: Optional list of component refs to check (default: all components)
        limit: Maximum number of overlaps to return (default: 50, max: 500)
        offset: Number of overlaps to skip for pagination (default: 0)

    Returns:
        JSON with overlaps, total count, and pagination info
    """
    try:
        _require_board()

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        # Validate refs if provided
        if refs:
            if err := _validate_refs(refs):
                return _error_response(err, "invalid_refs")

        # Use shared inspection logic
        inspector = BoardInspector(session.board)
        overlaps = inspector.check_overlaps(refs)
        total_count = len(overlaps)

        # Apply pagination
        paginated = overlaps[offset:offset + limit]
        has_more = (offset + len(paginated)) < total_count

        return _success_response({
            "total_count": total_count,
            "count": len(paginated),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "overlaps": paginated
        })
    except Exception as e:
        return _error_response(str(e), "check_failed")


# =============================================================================
# Discovery Tools
# =============================================================================


@mcp.tool()
def get_unplaced_components(limit: int = DEFAULT_LIMIT, offset: int = 0) -> str:
    """Get list of components that are outside the board area.

    Supports pagination for large boards with many unplaced components.

    Args:
        limit: Maximum number of results to return (default: 50, max: 500)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        JSON with component refs, total count, and pagination info
    """
    try:
        _require_board()

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        macro = MacroContext(session.board)
        unplaced = macro.get_unplaced_components()
        total_count = len(unplaced)

        # Apply pagination
        paginated = unplaced[offset:offset + limit]
        has_more = (offset + len(paginated)) < total_count

        return json.dumps({
            "total_count": total_count,
            "count": len(paginated),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "refs": paginated
        })
    except Exception as e:
        return _error_response(str(e), "discovery_failed")


# Valid filter_by values for find_components
VALID_FILTERS = {"ref", "value", "footprint"}


@mcp.tool()
def find_components(
    query: str,
    filter_by: str = "ref",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0
) -> str:
    """
    Find components matching a query.

    Supports pagination for large result sets.

    Args:
        query: Search string
        filter_by: Field to search ("ref", "value", "footprint")
        limit: Maximum number of results to return (default: 50, max: 500)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        JSON with matched components, total count, and pagination info
    """
    try:
        _require_board()
        if filter_by.lower() not in VALID_FILTERS:
            return _error_response(
                f"Invalid filter '{filter_by}'. Must be one of: ref, value, footprint",
                "invalid_filter"
            )

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        # Use shared inspection logic
        inspector = BoardInspector(session.board)
        try:
            matches = inspector.find_components(query, filter_by)
            total_count = len(matches)
        except ValueError as e:
            return _error_response(str(e), "invalid_filter")

        # Apply pagination
        paginated = matches[offset:offset + limit]
        has_more = (offset + len(paginated)) < total_count

        return json.dumps({
            "total_count": total_count,
            "count": len(paginated),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "matches": paginated
        })
    except Exception as e:
        return _error_response(str(e), "search_failed")


# =============================================================================
# Validation Tools
# =============================================================================

@mcp.tool()
def run_drc(
    use_kicad: bool = True,
    dfm_profile: str = "jlcpcb_standard",
    severity_filter: str = "all",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0
) -> str:
    """
    Run Design Rule Check on the loaded board.

    Supports pagination for boards with many DRC violations.

    Args:
        use_kicad: Whether to use KiCad's native DRC (if available)
        dfm_profile: DFM profile ("jlcpcb_standard", "jlcpcb_advanced", etc.)
        severity_filter: Filter by severity ("all", "errors", "warnings")
        limit: Maximum number of violations to return (default: 50, max: 500)
        offset: Number of violations to skip for pagination (default: 0)

    Returns:
        JSON with violations, total count, and pagination info
    """
    try:
        _require_board()
        from ..validation.drc import DRCChecker
        from ..dfm.profiles import get_profile

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        dfm = get_profile(dfm_profile)
        checker = DRCChecker(session.board, dfm)
        passed, violations = checker.run_checks()

        # Filter by severity
        if severity_filter != "all":
            violations = [v for v in violations if v.severity == severity_filter]

        total_count = len(violations)

        # Apply pagination
        paginated = violations[offset:offset + limit]
        has_more = (offset + len(paginated)) < total_count

        return json.dumps({
            "passed": passed,
            "total_count": total_count,
            "count": len(paginated),
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
            "violations": [{
                "rule": v.rule,
                "severity": v.severity,
                "message": v.message,
                "items": v.items,
                "location": {"x": v.location[0], "y": v.location[1]}
            } for v in paginated]
        })
    except Exception as e:
        logger.error("DRC failed: %s", e)
        return _error_response(str(e), "drc_failed")


@mcp.tool()
def validate_placement(limit: int = DEFAULT_LIMIT, offset: int = 0) -> str:
    """
    Run placement validation and get confidence report.

    Supports pagination for the flags list.

    Args:
        limit: Maximum number of flags to return (default: 50, max: 500)
        offset: Number of flags to skip for pagination (default: 0)

    Returns:
        JSON with validation scores and paginated flags
    """
    try:
        _require_board()
        from ..validation.confidence import ConfidenceScorer

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        scorer = ConfidenceScorer()  # Uses default DFM profile
        report = scorer.assess(session.board)

        all_flags = report.flags
        total_flags = len(all_flags)

        # Apply pagination to flags
        paginated_flags = all_flags[offset:offset + limit]
        has_more = (offset + len(paginated_flags)) < total_flags

        return json.dumps({
            "overall_score": report.overall_score,
            "placement_score": report.placement_score,
            "routing_score": report.routing_score,
            "dfm_score": report.dfm_score,
            "electrical_score": report.electrical_score,
            "flags_total_count": total_flags,
            "flags_count": len(paginated_flags),
            "flags_offset": offset,
            "flags_limit": limit,
            "flags_has_more": has_more,
            "flags": [{"category": f.category.value, "severity": f.severity.value, "message": f.message} for f in paginated_flags],
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
# BGA Fanout Tools
# =============================================================================

@mcp.tool()
def detect_bga_components() -> str:
    """
    Detect BGA-like components on the board.

    Identifies components that likely need fanout routing based on:
    - Footprint name patterns (BGA, LGA, QFN, FPGA, etc.)
    - Number of pads (grid-like arrangements)
    - Pad arrangement (regular grid pattern)

    Returns:
        JSON with list of detected BGA component references and basic info
    """
    try:
        _require_board()
        from ..routing.fanout import FanoutGenerator

        generator = FanoutGenerator(session.board)
        bga_refs = generator.detect_bgas()

        # Get basic info for each detected BGA
        bga_info = []
        for ref in bga_refs:
            comp = session.board.components.get(ref)
            if comp:
                pitch = generator.measure_pitch(comp)
                bga_info.append({
                    "ref": ref,
                    "footprint": comp.footprint,
                    "pad_count": len(comp.pads),
                    "pitch_mm": round(pitch, 3) if pitch > 0 else None,
                    "position": {"x": round(comp.x, 3), "y": round(comp.y, 3)},
                })

        return json.dumps({
            "detected_count": len(bga_refs),
            "components": bga_info,
        }, indent=2)

    except Exception as e:
        logger.error("BGA detection failed: %s", e, exc_info=True)
        return _error_response(str(e), "detection_failed")


@mcp.tool()
def fanout_component(
    ref: str,
    strategy: str = "auto",
    include_escape: bool = True
) -> str:
    """
    Generate fanout pattern for a specific BGA component.

    Creates escape routing vias and traces for high-density BGA packages:
    - Dogbone pattern for pitch >= 0.65mm
    - Via-in-Pad (VIP) for pitch <= 0.5mm
    - Layer assignment using onion model
    - Escape routing to clear space

    Args:
        ref: Component reference (e.g., "U1")
        strategy: "auto" (detect from pitch), "dogbone", or "vip"
        include_escape: Include escape routing from vias to board edge

    Returns:
        JSON with generated vias, traces, and statistics
    """
    try:
        _require_board()
        from ..routing.fanout import FanoutGenerator, FanoutStrategy

        # Map strategy string to enum
        strategy_map = {
            "auto": FanoutStrategy.AUTO,
            "dogbone": FanoutStrategy.DOGBONE,
            "vip": FanoutStrategy.VIP,
        }
        if strategy.lower() not in strategy_map:
            return _error_response(
                f"Invalid strategy: {strategy}. Use: auto, dogbone, vip",
                "invalid_strategy"
            )

        generator = FanoutGenerator(session.board)
        result = generator.fanout_component(
            ref,
            strategy=strategy_map[strategy.lower()],
            include_escape=include_escape,
        )

        if not result.success:
            return _error_response(result.failure_reason, "fanout_failed")

        # Build response
        vias_data = [
            {
                "x": round(v.x, 3),
                "y": round(v.y, 3),
                "drill": v.drill_diameter,
                "pad": v.pad_diameter,
                "start_layer": v.start_layer,
                "end_layer": v.end_layer,
                "net": v.net_name,
                "pad_number": v.pad_number,
            }
            for v in result.vias
        ]

        traces_data = [
            {
                "start": {"x": round(t.start[0], 3), "y": round(t.start[1], 3)},
                "end": {"x": round(t.end[0], 3), "y": round(t.end[1], 3)},
                "width": t.width,
                "layer": t.layer,
                "net": t.net_name,
            }
            for t in result.traces
        ]

        return json.dumps({
            "success": True,
            "component": ref,
            "strategy_used": result.strategy_used.value,
            "pitch_mm": round(result.pitch_detected, 3),
            "ring_count": result.ring_count,
            "via_count": len(result.vias),
            "trace_count": len(result.traces),
            "vias": vias_data,
            "traces": traces_data,
            "stats": result.stats,
            "warnings": result.warnings,
        }, indent=2)

    except Exception as e:
        logger.error("Fanout generation failed: %s", e, exc_info=True)
        return _error_response(str(e), "fanout_failed")


@mcp.tool()
def fanout_all_bgas(
    strategy: str = "auto",
    include_escape: bool = True
) -> str:
    """
    Generate fanout for all detected BGA components on the board.

    Automatically detects BGA components and generates escape routing for each.

    Args:
        strategy: "auto" (detect from pitch), "dogbone", or "vip"
        include_escape: Include escape routing from vias to board edge

    Returns:
        JSON with summary and per-component results
    """
    try:
        _require_board()
        from ..routing.fanout import FanoutGenerator, FanoutStrategy

        # Map strategy string to enum
        strategy_map = {
            "auto": FanoutStrategy.AUTO,
            "dogbone": FanoutStrategy.DOGBONE,
            "vip": FanoutStrategy.VIP,
        }
        if strategy.lower() not in strategy_map:
            return _error_response(
                f"Invalid strategy: {strategy}. Use: auto, dogbone, vip",
                "invalid_strategy"
            )

        generator = FanoutGenerator(session.board)
        results = generator.fanout_all_bgas(
            strategy=strategy_map[strategy.lower()],
            include_escape=include_escape,
        )

        # Build summary
        success_count = sum(1 for r in results.values() if r.success)
        total_vias = sum(len(r.vias) for r in results.values() if r.success)
        total_traces = sum(len(r.traces) for r in results.values() if r.success)

        # Per-component summary
        component_results = []
        for ref, result in results.items():
            comp_result = {
                "ref": ref,
                "success": result.success,
            }
            if result.success:
                comp_result.update({
                    "strategy": result.strategy_used.value,
                    "pitch_mm": round(result.pitch_detected, 3),
                    "rings": result.ring_count,
                    "vias": len(result.vias),
                    "traces": len(result.traces),
                })
            else:
                comp_result["failure_reason"] = result.failure_reason
            component_results.append(comp_result)

        return json.dumps({
            "success": success_count > 0,
            "total_components": len(results),
            "successful": success_count,
            "failed": len(results) - success_count,
            "total_vias": total_vias,
            "total_traces": total_traces,
            "components": component_results,
        }, indent=2)

    except Exception as e:
        logger.error("Fanout all BGAs failed: %s", e, exc_info=True)
        return _error_response(str(e), "fanout_failed")


@mcp.tool()
def get_fanout_preview(ref: str) -> str:
    """
    Get a preview of fanout parameters for a component without generating.

    Analyzes the component to determine:
    - Detected pitch
    - Recommended strategy
    - Number of rings
    - Layer assignments

    Args:
        ref: Component reference (e.g., "U1")

    Returns:
        JSON with fanout analysis and recommendations
    """
    try:
        _require_board()
        from ..routing.fanout import FanoutGenerator, LayerAssigner

        comp = session.board.components.get(ref)
        if not comp:
            return _error_response(f"Component not found: {ref}", "not_found")

        generator = FanoutGenerator(session.board)

        # Check if it's a BGA candidate
        is_bga = generator._is_bga_candidate(comp)

        # Measure pitch
        pitch = generator.measure_pitch(comp)

        # Determine strategy
        recommended_strategy = "dogbone" if pitch > 0.5 else "vip"

        # Analyze rings
        pads_data = []
        for pad in comp.pads:
            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            pads_data.append((pad.number, abs_x, abs_y))

        layer_assigner = LayerAssigner(layer_count=session.board.layer_count)
        rings = layer_assigner.analyze_rings(pads_data, pitch)
        layer_mapping = layer_assigner.assign_layers(rings, strategy="balanced")

        # Build ring info
        ring_info = [
            {
                "index": ring.index,
                "pin_count": ring.pin_count,
                "assigned_layer": layer_mapping.ring_to_layer.get(ring.index, 0),
            }
            for ring in rings
        ]

        return json.dumps({
            "ref": ref,
            "is_bga_candidate": is_bga,
            "footprint": comp.footprint,
            "pad_count": len(comp.pads),
            "pitch_mm": round(pitch, 3) if pitch > 0 else None,
            "recommended_strategy": recommended_strategy if is_bga else None,
            "ring_count": len(rings),
            "rings": ring_info,
            "layers_needed": sorted(layer_mapping.layer_to_pads.keys()) if layer_mapping else [],
            "board_layers": session.board.layer_count,
        }, indent=2)

    except Exception as e:
        logger.error("Fanout preview failed: %s", e, exc_info=True)
        return _error_response(str(e), "preview_failed")


# =============================================================================
# Routing Tools
# =============================================================================

@mcp.tool()
def route_board(
    visualize: bool = False,
    diff_pairs: Optional[List[str]] = None,
    critical_nets: Optional[List[str]] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0
) -> str:
    """
    Route all nets on the board using the multi-phase routing pipeline.

    The routing pipeline executes in phases:
    1. Fanout & Escape: BGA/QFN escape routing
    2. Critical Nets: Differential pairs (if specified)
    3. General Routing: A* with greedy multiplier for all remaining nets

    Supports pagination for the per-net results list.

    Args:
        visualize: Generate HTML visualization of routing process
        diff_pairs: List of diff pair definitions in format "NAME:POS_NET:NEG_NET"
                   e.g., ["USB:USB_D+:USB_D-", "HDMI0:HDMI_TX0_P:HDMI_TX0_N"]
        critical_nets: List of net names to route with high priority
        limit: Maximum number of net results to return (default: 50, max: 500)
        offset: Number of net results to skip for pagination (default: 0)

    Returns:
        JSON with routing statistics and per-net results (paginated)
    """
    try:
        _require_board()
        from ..routing import RoutingManager, RoutingManagerConfig

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        config = RoutingManagerConfig(visualize=visualize)
        manager = RoutingManager(session.board, config=config)

        # Parse and add diff pairs
        if diff_pairs:
            for dp_str in diff_pairs:
                parts = dp_str.split(":")
                if len(parts) >= 3:
                    name, pos_net, neg_net = parts[0], parts[1], parts[2]
                    manager.add_diff_pair(name, pos_net, neg_net)
                    logger.info(f"Added diff pair: {name} ({pos_net}/{neg_net})")

        # Add critical nets
        if critical_nets:
            manager.set_critical_nets(critical_nets)

        # Run routing
        result = manager.route_all()

        # Build response
        net_summaries = []
        for net_name, net_result in result.net_results.items():
            net_summaries.append({
                "net": net_name,
                "success": net_result.success,
                "length_mm": round(net_result.total_length, 2),
                "vias": net_result.via_count,
                "segments": len(net_result.segments),
                "failure_reason": net_result.failure_reason if not net_result.success else None
            })

        total_nets_in_results = len(net_summaries)

        # Apply pagination to net summaries
        paginated = net_summaries[offset:offset + limit]
        has_more = (offset + len(paginated)) < total_nets_in_results

        return json.dumps({
            "success": result.success,
            "completion_rate": round(result.completion_rate, 1),
            "total_nets": result.total_nets,
            "routed_nets": result.routed_nets,
            "failed_nets": result.failed_nets,
            "total_length_mm": round(result.total_length, 1),
            "total_vias": result.total_vias,
            "phases_completed": [p.value for p in result.phases_completed],
            "nets_total_count": total_nets_in_results,
            "nets_count": len(paginated),
            "nets_offset": offset,
            "nets_limit": limit,
            "nets_has_more": has_more,
            "nets": paginated,
            "errors": result.errors,
            "warnings": result.warnings
        }, indent=2)

    except Exception as e:
        logger.error("Board routing failed: %s", e, exc_info=True)
        return _error_response(str(e), "routing_failed")


@mcp.tool()
def route_net(net_name: str, visualize: bool = False) -> str:
    """
    Route a single net by name.

    Uses A* with greedy multiplier for fast pathfinding.

    Args:
        net_name: Name of the net to route
        visualize: Generate visualization of routing process

    Returns:
        JSON with routing result
    """
    try:
        _require_board()
        from ..routing import RoutingManager, RoutingManagerConfig

        config = RoutingManagerConfig(visualize=visualize)
        manager = RoutingManager(session.board, config=config)

        result = manager.route_net(net_name)

        segments_data = [
            {
                "start": {"x": round(s.start[0], 3), "y": round(s.start[1], 3)},
                "end": {"x": round(s.end[0], 3), "y": round(s.end[1], 3)},
                "layer": s.layer,
                "width": s.width
            }
            for s in result.segments
        ]

        vias_data = [
            {
                "x": round(v.x, 3),
                "y": round(v.y, 3),
                "drill": v.drill_diameter,
                "pad": v.pad_diameter
            }
            for v in result.vias
        ]

        return json.dumps({
            "success": result.success,
            "net": net_name,
            "length_mm": round(result.total_length, 2),
            "via_count": result.via_count,
            "segment_count": len(result.segments),
            "iterations": result.iterations,
            "explored_nodes": result.explored_count,
            "segments": segments_data,
            "vias": vias_data,
            "failure_reason": result.failure_reason if not result.success else None
        }, indent=2)

    except Exception as e:
        logger.error("Net routing failed: %s", e, exc_info=True)
        return _error_response(str(e), "routing_failed")


@mcp.tool()
def detect_diff_pairs() -> str:
    """
    Auto-detect differential pairs from net names.

    Identifies common diff pair patterns:
    - USB_D+/USB_D-
    - NET_P/NET_N
    - LVDS_TX_POS/LVDS_TX_NEG
    - etc.

    Returns:
        JSON with detected differential pairs
    """
    try:
        _require_board()
        from ..routing import DiffPairDetector

        # Get all net names
        net_names = list(session.board.nets.keys())

        detector = DiffPairDetector(net_names)
        pairs = detector.detect()

        pairs_data = [
            {
                "name": p.name,
                "positive_net": p.positive_net,
                "negative_net": p.negative_net,
                "pattern": p.pattern.value,
                "confidence": p.confidence
            }
            for p in pairs
        ]

        return json.dumps({
            "detected_count": len(pairs),
            "total_nets": len(net_names),
            "diff_pairs": pairs_data
        }, indent=2)

    except Exception as e:
        logger.error("Diff pair detection failed: %s", e, exc_info=True)
        return _error_response(str(e), "detection_failed")


@mcp.tool()
def get_routing_preview() -> str:
    """
    Get a preview of routing parameters and estimated complexity.

    Analyzes the board to provide:
    - Net count and classification
    - Estimated routing difficulty
    - Detected BGA components requiring fanout
    - Identified differential pairs

    Returns:
        JSON with routing analysis and recommendations
    """
    try:
        _require_board()
        from ..routing import DiffPairDetector
        from ..routing.fanout import FanoutGenerator

        # Count nets by pad count
        net_stats = {"single_pad": 0, "two_pad": 0, "multi_pad": 0}
        total_pads = 0
        for net_name, net in session.board.nets.items():
            pad_count = len(net.pads) if hasattr(net, 'pads') else 0
            total_pads += pad_count
            if pad_count <= 1:
                net_stats["single_pad"] += 1
            elif pad_count == 2:
                net_stats["two_pad"] += 1
            else:
                net_stats["multi_pad"] += 1

        # Detect diff pairs
        net_names = list(session.board.nets.keys())
        detector = DiffPairDetector(net_names)
        diff_pairs = detector.detect()

        # Detect BGAs
        generator = FanoutGenerator(session.board)
        bga_refs = generator.detect_bgas()

        # Estimate complexity
        complexity = "simple"
        if len(bga_refs) > 0:
            complexity = "complex"
        elif len(diff_pairs) > 2:
            complexity = "moderate"
        elif len(session.board.nets) > 100:
            complexity = "moderate"

        return json.dumps({
            "net_count": len(session.board.nets),
            "component_count": len(session.board.components),
            "layer_count": session.board.layer_count,
            "net_stats": net_stats,
            "diff_pairs_detected": len(diff_pairs),
            "diff_pair_names": [p.name for p in diff_pairs],
            "bga_components": bga_refs,
            "estimated_complexity": complexity,
            "recommendations": {
                "enable_fanout": len(bga_refs) > 0,
                "route_diff_pairs_first": len(diff_pairs) > 0,
                "estimated_routing_time": "quick" if complexity == "simple" else "moderate"
            }
        }, indent=2)

    except Exception as e:
        logger.error("Routing preview failed: %s", e, exc_info=True)
        return _error_response(str(e), "preview_failed")


# =============================================================================
# Pin Swap Optimization Tools
# =============================================================================

@mcp.tool()
def analyze_pin_swaps(
    ref: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0
) -> str:
    """
    Analyze potential pin swaps for routing optimization.

    Detects swappable pin groups on FPGAs, MCUs, and connectors. Reports
    potential improvement from optimizing pin assignments.

    Supports pagination when analyzing all components.

    Args:
        ref: Specific component reference (e.g., "U1"). If None, analyzes all.
        limit: Maximum number of component analyses to return (default: 50, max: 500)
        offset: Number of component analyses to skip for pagination (default: 0)

    Returns:
        JSON with detected swap groups and improvement potential
    """
    try:
        _require_board()
        from ..routing.pinswapper import PinSwapper, SwapConfig

        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_LIMIT))
        offset = max(0, offset)

        config = SwapConfig()
        swapper = PinSwapper(session.board, config)

        if ref:
            # Analyze single component - no pagination needed
            analysis = swapper.analyze_component(ref)
            if "error" in analysis:
                return _error_response(analysis["error"], "analysis_failed")

            return json.dumps({
                "component": ref,
                "analysis": analysis
            }, indent=2)
        else:
            # Analyze all swappable components
            groups = swapper._detector.detect_all()
            analyses = []

            for comp_ref in groups.keys():
                analysis = swapper.analyze_component(comp_ref)
                if "error" not in analysis:
                    analyses.append({
                        "component": comp_ref,
                        "swap_groups": analysis.get("swap_groups", 0),
                        "current_crossings": analysis.get("current_crossings", 0),
                        "groups": analysis.get("groups", [])
                    })

            total_count = len(analyses)

            # Apply pagination
            paginated = analyses[offset:offset + limit]
            has_more = (offset + len(paginated)) < total_count

            # Get global crossing stats
            crossing_result = swapper.get_crossing_analysis()

            return json.dumps({
                "components_with_swaps": total_count,
                "total_crossings": crossing_result.total_crossings,
                "crossing_density": round(crossing_result.crossing_density, 2),
                "analyses_total_count": total_count,
                "analyses_count": len(paginated),
                "analyses_offset": offset,
                "analyses_limit": limit,
                "analyses_has_more": has_more,
                "analyses": paginated
            }, indent=2)

    except Exception as e:
        logger.error("Pin swap analysis failed: %s", e, exc_info=True)
        return _error_response(str(e), "analysis_failed")


@mcp.tool()
def optimize_pin_swaps(
    ref: Optional[str] = None,
    min_improvement: float = 5.0,
    apply: bool = True
) -> str:
    """
    Optimize pin assignments to reduce routing complexity.

    Uses bipartite matching (Hungarian algorithm) to find optimal pin-to-net
    assignments that minimize wire length and crossings.

    Args:
        ref: Specific component to optimize. If None, optimizes all.
        min_improvement: Minimum improvement % to apply swaps (default: 5.0)
        apply: Whether to apply swaps to the board (default: True)

    Returns:
        JSON with optimization results and swaps performed
    """
    try:
        _require_board()
        from ..routing.pinswapper import PinSwapper, SwapConfig

        config = SwapConfig(min_improvement=min_improvement)
        swapper = PinSwapper(session.board, config)

        if ref:
            result = swapper.optimize_component(ref, apply=apply)
            results = {ref: result}
        else:
            results = swapper.optimize_all(apply=apply)

        # Build response
        result_data = []
        total_swaps = 0
        for comp_ref, result in results.items():
            total_swaps += result.total_swaps
            result_data.append({
                "component": comp_ref,
                "success": result.success,
                "groups_detected": result.groups_detected,
                "groups_optimized": result.groups_optimized,
                "swaps_performed": result.total_swaps,
                "crossing_improvement": f"{result.crossing_improvement:.1f}%",
                "wire_improvement": f"{result.wire_improvement:.1f}%",
                "failure_reason": result.failure_reason if not result.success else None
            })

        # Get crossing stats
        crossing_result = swapper.get_crossing_analysis()

        return json.dumps({
            "success": total_swaps > 0,
            "total_swaps": total_swaps,
            "components_optimized": len([r for r in results.values() if r.total_swaps > 0]),
            "total_crossings": crossing_result.total_crossings,
            "results": result_data,
            "constraint_updates_pending": swapper.pending_constraint_count
        }, indent=2)

    except Exception as e:
        logger.error("Pin swap optimization failed: %s", e, exc_info=True)
        return _error_response(str(e), "optimization_failed")


@mcp.tool()
def get_crossing_analysis() -> str:
    """
    Analyze ratsnest crossings on the board.

    Counts wire crossings to measure routing complexity. Lower crossing
    count = easier routing. Use this before and after pin swaps to
    measure improvement.

    Returns:
        JSON with crossing statistics and worst nets
    """
    try:
        _require_board()
        from ..routing.pinswapper import CrossingCounter

        counter = CrossingCounter(session.board)
        result = counter.count_all()

        return json.dumps({
            "total_crossings": result.total_crossings,
            "edges_analyzed": result.edges_analyzed,
            "crossing_density": round(result.crossing_density, 3),
            "crossings_by_component": dict(sorted(
                result.crossings_by_component.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]),
            "worst_nets": result.worst_nets[:10],
            "crossing_pairs_sample": result.crossing_pairs[:10]
        }, indent=2)

    except Exception as e:
        logger.error("Crossing analysis failed: %s", e, exc_info=True)
        return _error_response(str(e), "analysis_failed")


@mcp.tool()
def export_pin_constraints(
    format: str = "xdc",
    output_path: Optional[str] = None
) -> str:
    """
    Export pin swap constraints to a file.

    Generates constraint file updates for FPGA tools reflecting the
    optimized pin assignments.

    Args:
        format: Output format ("xdc", "qsf", "tcl", "csv", "json")
        output_path: Optional file path. If None, returns content as string.

    Returns:
        JSON with constraint file content or path
    """
    try:
        _require_board()
        from ..routing.pinswapper import PinSwapper, SwapConfig, ConstraintFormat

        format_map = {
            "xdc": ConstraintFormat.XDC,
            "qsf": ConstraintFormat.QSF,
            "tcl": ConstraintFormat.TCL,
            "csv": ConstraintFormat.NETLIST,
            "json": ConstraintFormat.JSON,
        }
        if format.lower() not in format_map:
            return _error_response(
                f"Invalid format: {format}. Use: xdc, qsf, tcl, csv, json",
                "invalid_format"
            )

        constraint_format = format_map[format.lower()]

        # Create swapper and run optimization
        config = SwapConfig()
        swapper = PinSwapper(session.board, config)
        swapper.optimize_all(apply=False)  # Don't apply, just generate constraints

        if swapper.pending_constraint_count == 0:
            return json.dumps({
                "success": False,
                "message": "No pin swaps to export. Run optimize_pin_swaps first."
            })

        if output_path:
            swapper.export_constraints(Path(output_path), constraint_format)
            return json.dumps({
                "success": True,
                "path": output_path,
                "constraint_count": swapper.pending_constraint_count
            })
        else:
            content = swapper.get_constraint_preview(constraint_format)
            return json.dumps({
                "success": True,
                "format": format,
                "constraint_count": swapper.pending_constraint_count,
                "content": content
            }, indent=2)

    except Exception as e:
        logger.error("Constraint export failed: %s", e, exc_info=True)
        return _error_response(str(e), "export_failed")


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

    logger.info("Starting AtoPlace MCP server with backend: %s", _get_actual_mode().value)
    mcp.run()

if __name__ == "__main__":
    main()
