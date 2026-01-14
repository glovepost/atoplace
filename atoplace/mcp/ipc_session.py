"""
IPC-based Session for MCP Server

Provides a Session-compatible interface that communicates with the
KiCad bridge via IPC. This allows the MCP server to run in Python 3.10+
while pcbnew operations run in Python 3.9.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..board.abstraction import Board
from .ipc import IPCClient, IPCError, deserialize_board, serialize_board, DEFAULT_SOCKET_PATH

logger = logging.getLogger(__name__)


class IPCSession:
    """
    Session implementation that uses IPC to communicate with KiCad bridge.

    Maintains a local copy of the board for read operations and context
    generation, while delegating mutations to the bridge process.
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.board: Optional[Board] = None
        self.source_path: Optional[Path] = None
        self._client = IPCClient(socket_path)
        self._connected = False
        self._dirty = False
        self._dirty_refs: set = set()

    @property
    def is_loaded(self) -> bool:
        """Check if a board is loaded."""
        return self.board is not None

    @property
    def is_dirty(self) -> bool:
        """Check if board has unsaved changes."""
        return self._dirty

    @property
    def dirty_components(self) -> List[str]:
        """Get list of modified components."""
        return list(self._dirty_refs)

    def connect(self) -> bool:
        """Connect to the KiCad bridge."""
        if self._connected:
            return True

        if self._client.connect():
            self._connected = True
            # Verify connection with ping
            try:
                result = self._client.call("ping")
                logger.info("Connected to KiCad bridge")
                return True
            except IPCError as e:
                logger.error("Bridge ping failed: %s", e)
                self._connected = False
                return False
        return False

    def disconnect(self):
        """Disconnect from the KiCad bridge."""
        self._client.disconnect()
        self._connected = False

    def load(self, path: Path) -> "IPCSession":
        """
        Load a board from file via the bridge.

        Args:
            path: Path to KiCad PCB file

        Returns:
            Self for chaining
        """
        if not self.connect():
            raise ConnectionError("Cannot connect to KiCad bridge")

        try:
            result = self._client.call("load_board", path=str(path))
            self.board = deserialize_board(result["board"])
            self.source_path = path
            self._dirty = False
            self._dirty_refs.clear()
            logger.info("Loaded board via IPC: %d components", len(self.board.components))
            return self
        except IPCError as e:
            raise RuntimeError(f"Failed to load board: {e.message}")

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save board to file via the bridge.

        Args:
            path: Output path. If None, uses source path with .placed suffix.

        Returns:
            Path where board was saved
        """
        if not self.board:
            raise ValueError("No board loaded")

        if not self.connect():
            raise ConnectionError("Cannot connect to KiCad bridge")

        # First sync any local changes to the bridge
        self._sync_to_bridge()

        try:
            result = self._client.call("save_board", path=str(path) if path else None)
            self._dirty = False
            return Path(result["path"])
        except IPCError as e:
            raise RuntimeError(f"Failed to save board: {e.message}")

    def checkpoint(self, description: str = ""):
        """Create a checkpoint for undo."""
        if not self.board or not self._connected:
            return

        # Sync local changes before checkpoint
        self._sync_to_bridge()

        try:
            self._client.call("checkpoint", description=description)
        except IPCError as e:
            logger.warning("Checkpoint failed: %s", e.message)

    def undo(self) -> bool:
        """Undo last change."""
        if not self._connected:
            return False

        try:
            result = self._client.call("undo")
            if result.get("board"):
                self.board = deserialize_board(result["board"])
                self._dirty = True
            return result.get("action") == "undone"
        except IPCError as e:
            logger.error("Undo failed: %s", e.message)
            return False

    def redo(self) -> bool:
        """Redo last undone change."""
        if not self._connected:
            return False

        try:
            result = self._client.call("redo")
            if result.get("board"):
                self.board = deserialize_board(result["board"])
                self._dirty = True
            return result.get("action") == "redone"
        except IPCError as e:
            logger.error("Redo failed: %s", e.message)
            return False

    def mark_modified(self, refs: List[str]):
        """Mark components as modified."""
        self._dirty = True
        self._dirty_refs.update(refs)

    def clear_dirty(self):
        """Clear dirty state."""
        self._dirty_refs.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        if self._connected:
            try:
                return self._client.call("get_session_stats")
            except IPCError:
                pass

        return {
            "loaded": self.is_loaded,
            "source": str(self.source_path) if self.source_path else None,
            "dirty": self.is_dirty,
            "dirty_count": len(self._dirty_refs),
            "undo_available": False,
            "redo_available": False,
            "components": len(self.board.components) if self.board else 0,
            "mode": "ipc",
        }

    def _sync_to_bridge(self):
        """Sync local component changes to the bridge."""
        if not self._dirty_refs or not self._connected:
            return

        updates = []
        for ref in self._dirty_refs:
            comp = self.board.components.get(ref)
            if comp:
                updates.append({
                    "ref": ref,
                    "x": comp.x,
                    "y": comp.y,
                    "rotation": comp.rotation,
                    "locked": comp.locked,
                })

        if updates:
            try:
                self._client.call("update_components", updates=updates)
                self._dirty_refs.clear()
            except IPCError as e:
                logger.error("Failed to sync to bridge: %s", e.message)

    def _refresh_from_bridge(self):
        """Refresh local board state from bridge."""
        if not self._connected:
            return

        try:
            result = self._client.call("get_board")
            if result.get("board"):
                self.board = deserialize_board(result["board"])
        except IPCError as e:
            logger.error("Failed to refresh from bridge: %s", e.message)


def create_session(use_ipc: bool = False, socket_path: str = None) -> Any:
    """
    Create a session instance.

    Args:
        use_ipc: If True, use IPC session for bridge communication
        socket_path: Override default socket path

    Returns:
        Session or IPCSession instance
    """
    if use_ipc:
        path = socket_path or DEFAULT_SOCKET_PATH
        return IPCSession(path)
    else:
        from ..api.session import Session
        return Session()
