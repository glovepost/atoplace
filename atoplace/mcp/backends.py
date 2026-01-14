"""
Backend Selection Factory for MCP Sessions.

Supports three backend modes:
- direct: Direct pcbnew access (requires KiCad Python environment)
- ipc: Bridge-based IPC to pcbnew process (Python 3.10+ compatible)
- kipy: Live KiCad IPC via official kicad-python API (KiCad 9+)

The kipy backend enables real-time component manipulation with instant
visual updates in KiCad, without save/reload cycles.
"""

import os
import logging
from enum import Enum
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)


class BackendMode(Enum):
    """Available backend modes for MCP session."""
    DIRECT = "direct"  # Direct pcbnew access
    IPC = "ipc"        # Bridge-based IPC
    KIPY = "kipy"      # KiCad IPC API (kicad-python)


class BackendNotAvailableError(Exception):
    """Raised when a requested backend is not available."""
    pass


def get_backend_mode() -> BackendMode:
    """
    Determine backend mode from environment variables.

    Checks in priority order:
    1. ATOPLACE_BACKEND: Explicit mode selection (direct/ipc/kipy)
    2. ATOPLACE_USE_KIPY: Set to 1/true for kipy mode
    3. ATOPLACE_USE_IPC: Set to 1/true for IPC mode
    4. Default: direct mode

    Returns:
        BackendMode enum value
    """
    # Explicit backend selection
    explicit = os.environ.get("ATOPLACE_BACKEND", "").lower().strip()
    if explicit:
        if explicit in ("kipy", "live", "kicad-ipc"):
            return BackendMode.KIPY
        if explicit in ("ipc", "bridge"):
            return BackendMode.IPC
        if explicit in ("direct", "pcbnew", "swig"):
            return BackendMode.DIRECT
        logger.warning("Unknown ATOPLACE_BACKEND value: %s, using direct", explicit)

    # Legacy environment variables
    if os.environ.get("ATOPLACE_USE_KIPY", "").lower() in ("1", "true", "yes"):
        return BackendMode.KIPY
    if os.environ.get("ATOPLACE_USE_IPC", "").lower() in ("1", "true", "yes"):
        return BackendMode.IPC

    return BackendMode.DIRECT


def check_kipy_available() -> Tuple[bool, str]:
    """
    Check if kipy is available and can connect to KiCad.

    Returns:
        Tuple of (available: bool, message: str)
    """
    # Check if kipy package is installed
    try:
        from kipy import KiCad
    except ImportError:
        return False, "kipy not installed (pip install kicad-python)"

    # Try to connect to KiCad
    try:
        # Try multiple socket paths (cross-platform)
        socket_paths = _get_socket_paths()

        for path in socket_paths:
            try:
                kicad = KiCad(socket_path=path) if path else KiCad()
                kicad.ping()
                socket_desc = path or "auto-detected"
                return True, f"Connected to KiCad via {socket_desc}"
            except Exception:
                continue

        return False, "KiCad not running or API not enabled (Preferences > Plugins)"

    except Exception as e:
        return False, f"kipy connection error: {e}"


def check_ipc_available() -> Tuple[bool, str]:
    """
    Check if IPC bridge is available.

    Returns:
        Tuple of (available: bool, message: str)
    """
    from .ipc import DEFAULT_SOCKET_PATH

    socket_path = os.environ.get("ATOPLACE_IPC_SOCKET", DEFAULT_SOCKET_PATH)

    if os.path.exists(socket_path):
        return True, f"Bridge socket found: {socket_path}"

    return False, f"Bridge socket not found: {socket_path}"


def check_direct_available() -> Tuple[bool, str]:
    """
    Check if direct pcbnew access is available.

    Returns:
        Tuple of (available: bool, message: str)
    """
    try:
        import pcbnew
        version = pcbnew.Version() if hasattr(pcbnew, 'Version') else "unknown"
        return True, f"pcbnew available (KiCad {version})"
    except ImportError:
        return False, "pcbnew not available (not in KiCad Python environment)"


def get_available_backends() -> dict:
    """
    Get information about all available backends.

    Returns:
        Dict with backend info: {mode: {'available': bool, 'message': str}}
    """
    return {
        'kipy': dict(zip(['available', 'message'], check_kipy_available())),
        'ipc': dict(zip(['available', 'message'], check_ipc_available())),
        'direct': dict(zip(['available', 'message'], check_direct_available())),
    }


def create_session(mode: Optional[BackendMode] = None):
    """
    Create appropriate session based on backend mode.

    Args:
        mode: Backend mode (auto-detected from environment if None)

    Returns:
        Session instance (Session, IPCSession, or KiPySession)

    Raises:
        BackendNotAvailableError: If requested backend is not available
    """
    if mode is None:
        mode = get_backend_mode()

    logger.info("Creating session with backend: %s", mode.value)

    if mode == BackendMode.KIPY:
        return _create_kipy_session()

    if mode == BackendMode.IPC:
        return _create_ipc_session()

    # Direct mode
    return _create_direct_session()


def create_session_with_fallback(preferred: BackendMode = BackendMode.KIPY):
    """
    Create session with automatic fallback to available backends.

    Tries backends in order: preferred -> kipy -> ipc -> direct

    Args:
        preferred: Preferred backend mode

    Returns:
        Tuple of (session, actual_mode)
    """
    # Build priority order
    order = [preferred]
    for mode in [BackendMode.KIPY, BackendMode.IPC, BackendMode.DIRECT]:
        if mode not in order:
            order.append(mode)

    last_error = None
    for mode in order:
        try:
            session = create_session(mode)
            if mode != preferred:
                logger.warning("Fell back to %s mode (preferred: %s)",
                              mode.value, preferred.value)
            return session, mode
        except BackendNotAvailableError as e:
            last_error = e
            logger.debug("Backend %s not available: %s", mode.value, e)
            continue

    raise BackendNotAvailableError(
        f"No backends available. Last error: {last_error}"
    )


def _create_kipy_session():
    """Create a KiPySession instance."""
    try:
        from .kipy_session import KiPySession
    except ImportError as e:
        raise BackendNotAvailableError(
            f"kipy_session module not available: {e}. "
            "Ensure kicad-python is installed: pip install kicad-python"
        )

    available, msg = check_kipy_available()
    if not available:
        raise BackendNotAvailableError(f"kipy backend: {msg}")

    return KiPySession()


def _create_ipc_session():
    """Create an IPCSession instance."""
    try:
        from .ipc_session import IPCSession
    except ImportError as e:
        raise BackendNotAvailableError(f"ipc_session module not available: {e}")

    socket_path = os.environ.get("ATOPLACE_IPC_SOCKET")
    return IPCSession(socket_path) if socket_path else IPCSession()


def _create_direct_session():
    """Create a direct Session instance."""
    try:
        from ..api.session import Session
    except ImportError as e:
        raise BackendNotAvailableError(f"Session module not available: {e}")

    return Session()


def _get_socket_paths() -> list:
    """
    Get list of socket paths to try for kipy connection.

    Returns paths in priority order for cross-platform support.
    """
    paths = []

    # Check environment variable first
    env_socket = os.environ.get("KICAD_IPC_SOCKET")
    if env_socket:
        paths.append(env_socket)

    # Linux default paths
    paths.append("ipc:///tmp/kicad/api.sock")

    # XDG runtime directory (Linux)
    try:
        uid = os.getuid()
        paths.append(f"ipc:///run/user/{uid}/kicad/api.sock")
    except AttributeError:
        pass  # Windows doesn't have getuid()

    # Auto-detect (works on macOS, Windows)
    paths.append(None)

    return paths
