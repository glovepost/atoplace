#!/usr/bin/env python3
"""
KiCad Bridge Process

This module runs in KiCad's Python 3.9 environment and provides
pcbnew operations to the MCP server via IPC.

Run with:
    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/\
    Versions/Current/bin/python3 -m atoplace.mcp.bridge
"""

import sys
import os
import signal
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRIDGE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Session State (mirror of api/session.py but Python 3.9 compatible)
# =============================================================================

@dataclass
class ComponentState:
    """Snapshot of a component's position state."""
    x: float
    y: float
    rotation: float
    locked: bool


@dataclass
class BoardSnapshot:
    """Complete snapshot of board state for undo/redo."""
    component_states: Dict[str, ComponentState]
    description: str = ""


class BridgeSession:
    """
    Manages board state in the bridge process.

    Python 3.9 compatible version of the Session class.
    """

    MAX_UNDO_STACK = 50

    def __init__(self):
        self.board = None
        self.source_path: Optional[Path] = None
        self._undo_stack: List[BoardSnapshot] = []
        self._redo_stack: List[BoardSnapshot] = []
        self._dirty: bool = False
        self._dirty_refs: set = set()

    @property
    def is_loaded(self) -> bool:
        return self.board is not None

    def load(self, path: Path):
        """Load a board from KiCad file."""
        from ..board.abstraction import Board
        self.board = Board.from_kicad(path)
        self.source_path = path
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._dirty_refs.clear()
        self._save_snapshot("Initial load")
        return self.board

    def save(self, path: Optional[Path] = None) -> Path:
        """Save board to KiCad file."""
        if not self.board:
            raise ValueError("No board loaded")

        if path is None:
            if self.source_path:
                stem = self.source_path.stem
                if not stem.endswith('.placed'):
                    stem = stem + ".placed"
                path = self.source_path.parent / (stem + ".kicad_pcb")
            else:
                raise ValueError("No path specified and no source path")

        self.board.to_kicad(path)
        self._dirty = False
        return path

    def checkpoint(self, description: str = ""):
        """Create undo checkpoint."""
        if self.board:
            self._save_snapshot(description)

    def undo(self) -> bool:
        """Undo last change."""
        if len(self._undo_stack) <= 1:
            return False

        self._redo_stack.append(self._take_snapshot("Redo point"))
        self._undo_stack.pop()
        if self._undo_stack:
            self._restore_snapshot(self._undo_stack[-1])
        self._dirty = True
        return True

    def redo(self) -> bool:
        """Redo last undone change."""
        if not self._redo_stack:
            return False

        snapshot = self._redo_stack.pop()
        self._undo_stack.append(snapshot)
        self._restore_snapshot(snapshot)
        self._dirty = True
        return True

    def mark_modified(self, refs: List[str]):
        """Mark components as modified."""
        self._dirty = True
        self._dirty_refs.update(refs)

    def _take_snapshot(self, description: str = "") -> BoardSnapshot:
        """Create snapshot of current board state."""
        states = {}
        for ref, comp in self.board.components.items():
            states[ref] = ComponentState(
                x=comp.x,
                y=comp.y,
                rotation=comp.rotation,
                locked=comp.locked
            )
        return BoardSnapshot(component_states=states, description=description)

    def _save_snapshot(self, description: str = ""):
        """Save current state to undo stack."""
        snapshot = self._take_snapshot(description)
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        while len(self._undo_stack) > self.MAX_UNDO_STACK:
            self._undo_stack.pop(0)

    def _restore_snapshot(self, snapshot: BoardSnapshot):
        """Restore board state from snapshot."""
        for ref, state in snapshot.component_states.items():
            if ref in self.board.components:
                comp = self.board.components[ref]
                comp.x = state.x
                comp.y = state.y
                comp.rotation = state.rotation
                comp.locked = state.locked

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "loaded": self.is_loaded,
            "source": str(self.source_path) if self.source_path else None,
            "dirty": self._dirty,
            "dirty_count": len(self._dirty_refs),
            "undo_available": len(self._undo_stack) > 1,
            "redo_available": len(self._redo_stack) > 0,
            "components": len(self.board.components) if self.board else 0,
        }


# =============================================================================
# Bridge Handler
# =============================================================================

class KiCadBridge:
    """
    Handles IPC requests from the MCP server.

    Provides access to pcbnew operations and manages board state.
    """

    def __init__(self):
        self.session = BridgeSession()

    def handle_load_board(self, path: str) -> Dict[str, Any]:
        """Load a KiCad board file."""
        from .ipc import serialize_board

        logger.info("Loading board: %s", path)
        board = self.session.load(Path(path))

        return {
            "success": True,
            "path": path,
            "components": len(board.components),
            "nets": len(board.nets),
            "board": serialize_board(board),
        }

    def handle_save_board(self, path: str = None) -> Dict[str, Any]:
        """Save the board to file."""
        output_path = self.session.save(Path(path) if path else None)
        logger.info("Saved board to: %s", output_path)
        return {
            "success": True,
            "path": str(output_path),
        }

    def handle_get_board(self) -> Dict[str, Any]:
        """Get the current board state."""
        from .ipc import serialize_board

        if not self.session.is_loaded:
            return {"loaded": False, "board": None}

        return {
            "loaded": True,
            "board": serialize_board(self.session.board),
        }

    def handle_update_component(
        self,
        ref: str,
        x: float = None,
        y: float = None,
        rotation: float = None,
        locked: bool = None
    ) -> Dict[str, Any]:
        """Update a component's position/rotation/lock state."""
        if not self.session.is_loaded:
            return {"success": False, "message": "No board loaded"}

        comp = self.session.board.components.get(ref)
        if not comp:
            return {"success": False, "message": "Component not found: " + ref}

        if comp.locked and not (locked is False):
            return {"success": False, "message": "Component is locked: " + ref}

        modified = False
        if x is not None:
            comp.x = x
            modified = True
        if y is not None:
            comp.y = y
            modified = True
        if rotation is not None:
            comp.rotation = rotation
            modified = True
        if locked is not None:
            comp.locked = locked
            modified = True

        if modified:
            self.session.mark_modified([ref])

        return {"success": True, "ref": ref}

    def handle_update_components(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch update multiple components."""
        if not self.session.is_loaded:
            return {"success": False, "message": "No board loaded"}

        results = []
        modified_refs = []

        for update in updates:
            ref = update.get("ref")
            comp = self.session.board.components.get(ref)
            if not comp:
                results.append({"ref": ref, "success": False, "message": "Not found"})
                continue

            if comp.locked:
                results.append({"ref": ref, "success": False, "message": "Locked"})
                continue

            if "x" in update:
                comp.x = update["x"]
            if "y" in update:
                comp.y = update["y"]
            if "rotation" in update:
                comp.rotation = update["rotation"]
            if "locked" in update:
                comp.locked = update["locked"]

            modified_refs.append(ref)
            results.append({"ref": ref, "success": True})

        if modified_refs:
            self.session.mark_modified(modified_refs)

        return {"success": True, "results": results, "modified_count": len(modified_refs)}

    def handle_checkpoint(self, description: str = "") -> Dict[str, Any]:
        """Create an undo checkpoint."""
        self.session.checkpoint(description)
        return {"success": True}

    def handle_undo(self) -> Dict[str, Any]:
        """Undo last change."""
        from .ipc import serialize_board

        success = self.session.undo()
        result = {"success": success, "action": "undone" if success else "nothing_to_undo"}

        if success and self.session.board:
            result["board"] = serialize_board(self.session.board)

        return result

    def handle_redo(self) -> Dict[str, Any]:
        """Redo last undone change."""
        from .ipc import serialize_board

        success = self.session.redo()
        result = {"success": success, "action": "redone" if success else "nothing_to_redo"}

        if success and self.session.board:
            result["board"] = serialize_board(self.session.board)

        return result

    def handle_get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self.session.get_stats()

    def handle_ping(self) -> Dict[str, Any]:
        """Health check."""
        return {"pong": True, "loaded": self.session.is_loaded}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_bridge(socket_path: str = None):
    """Run the KiCad bridge server."""
    from .ipc import IPCServer, DEFAULT_SOCKET_PATH

    socket_path = socket_path or DEFAULT_SOCKET_PATH
    bridge = KiCadBridge()
    server = IPCServer(socket_path)

    # Register handlers
    server.register_handler("ping", bridge.handle_ping)
    server.register_handler("load_board", bridge.handle_load_board)
    server.register_handler("save_board", bridge.handle_save_board)
    server.register_handler("get_board", bridge.handle_get_board)
    server.register_handler("update_component", bridge.handle_update_component)
    server.register_handler("update_components", bridge.handle_update_components)
    server.register_handler("checkpoint", bridge.handle_checkpoint)
    server.register_handler("undo", bridge.handle_undo)
    server.register_handler("redo", bridge.handle_redo)
    server.register_handler("get_session_stats", bridge.handle_get_session_stats)

    # Handle shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down bridge...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start server
    logger.info("Starting KiCad bridge server...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    """Entry point for bridge process."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AtoPlace KiCad Bridge - Provides pcbnew access to MCP server"
    )
    parser.add_argument(
        "--socket", "-s",
        default=None,
        help="Unix socket path for IPC"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_bridge(socket_path=args.socket)


if __name__ == "__main__":
    main()
