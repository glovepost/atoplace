"""
KiPy Session - Live KiCad Integration via Official IPC API.

This module provides a Session-compatible interface that communicates
directly with a running KiCad 9+ instance using the official KiCad
IPC API (kicad-python/kipy).

Key features:
- Real-time component manipulation (instant visual updates)
- Integration with KiCad's native undo/redo system
- No save/reload cycles required
- Transaction support for grouping operations

Requirements:
- KiCad 9.0+ running with API enabled (Preferences > Plugins)
- kicad-python package (pip install kicad-python)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..board.abstraction import Board

logger = logging.getLogger(__name__)


class KiCadNotRunningError(Exception):
    """KiCad is not running or IPC API is not available."""
    pass


class KiCadConnectionLostError(Exception):
    """Connection to KiCad was lost during operation."""
    pass


class KiPySession:
    """
    Session implementation using kipy for live KiCad communication.

    Provides the same interface as Session/IPCSession, enabling seamless
    switching between backends. All component modifications are immediately
    reflected in KiCad's UI without requiring file save/reload.

    Usage:
        session = KiPySession()
        session.connect()
        session.load()  # Load currently open board from KiCad
        session.update_component("C6", x=10.0, y=15.0, rotation=90.0)
        # Component moves instantly in KiCad!
    """

    MAX_RECONNECT_ATTEMPTS = 3

    def __init__(self):
        """Initialize KiPySession."""
        self._kicad = None
        self._kipy_board = None
        self.board: Optional[Board] = None
        self.source_path: Optional[Path] = None
        self._connected = False
        self._dirty = False
        self._dirty_refs: set = set()
        self._current_commit = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_loaded(self) -> bool:
        """Check if a board is loaded."""
        return self.board is not None and self._connected

    @property
    def is_dirty(self) -> bool:
        """Check if board has unsaved changes."""
        return self._dirty

    @property
    def dirty_components(self) -> List[str]:
        """Get list of modified component references."""
        return list(self._dirty_refs)

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to running KiCad instance.

        Tries multiple socket paths for cross-platform support.

        Returns:
            True if connection successful
        """
        if self._connected:
            return True

        try:
            from kipy import KiCad
        except ImportError:
            logger.error("kipy not installed. Install with: pip install kicad-python")
            return False

        # Try multiple socket paths
        socket_paths = self._get_socket_paths()
        last_error = None

        for path in socket_paths:
            try:
                if path:
                    self._kicad = KiCad(socket_path=path)
                else:
                    self._kicad = KiCad()

                # Verify connection
                self._kicad.ping()
                self._connected = True

                socket_desc = path or "auto-detected"
                logger.info("Connected to KiCad via %s", socket_desc)
                return True

            except Exception as e:
                last_error = e
                logger.debug("Failed to connect via %s: %s", path, e)
                continue

        logger.error("Could not connect to KiCad: %s", last_error)
        return False

    def disconnect(self):
        """Disconnect from KiCad."""
        self._kicad = None
        self._kipy_board = None
        self._connected = False
        self._current_commit = None

    def ensure_connected(self) -> bool:
        """Ensure connection is established, reconnecting if needed."""
        if self._connected:
            return True
        return self.connect()

    def _get_socket_paths(self) -> list:
        """Get list of socket paths to try."""
        # Use centralized socket path logic from backends module
        from .backends import _get_socket_paths
        return _get_socket_paths()

    # =========================================================================
    # Board Operations
    # =========================================================================

    def load(self, path: Path = None) -> "KiPySession":
        """
        Load board from running KiCad instance.

        If path is provided, it's stored as source_path but the board
        data is always read from KiCad's currently open board.

        Args:
            path: Optional path to associate with the board

        Returns:
            Self for chaining
        """
        if not self.ensure_connected():
            raise KiCadNotRunningError(
                "Cannot connect to KiCad. Ensure KiCad 9+ is running "
                "with API enabled (Preferences > Plugins)."
            )

        try:
            # Get board from running KiCad
            self._kipy_board = self._kicad.get_board()

            # Convert to atoplace Board model
            from .kipy_adapter import kipy_board_to_atoplace
            self.board = kipy_board_to_atoplace(self._kipy_board)

            # Store source path
            if path:
                self.source_path = Path(path)
            elif hasattr(self._kipy_board, 'file_path') and self._kipy_board.file_path:
                self.source_path = Path(self._kipy_board.file_path)

            self._dirty = False
            self._dirty_refs.clear()

            logger.info("Loaded board from KiCad: %d components",
                       len(self.board.components))
            return self

        except Exception as e:
            raise RuntimeError(f"Failed to load board from KiCad: {e}")

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save board via KiCad.

        In kipy mode, changes are applied directly to KiCad. Changes are
        synced to KiCad's memory, then user saves via Ctrl+S (Cmd+S on Mac).

        Args:
            path: Output path. If provided, user is instructed to use
                  KiCad's "File > Save As" menu for Save As functionality,
                  since KIPY mode works with the live KiCad instance.

        Returns:
            Path where board should be saved
        """
        if not self.board or not self._connected:
            raise ValueError("No board loaded or not connected")

        # Sync any pending local changes to KiCad
        self._sync_to_kicad()

        if path is not None:
            # Explicit path provided - but KIPY mode can't do Save As directly
            # User needs to use KiCad's native Save As
            logger.warning(
                "KIPY mode: Save As not supported. Use KiCad's 'File > Save As' menu. "
                "Changes are synced to KiCad."
            )
            raise ValueError(
                "Save As not supported in KIPY mode. "
                "Your changes are synced to KiCad. "
                "Use 'File > Save As' in KiCad to save to a different path."
            )
        else:
            # No path - changes synced to KiCad, user saves via Ctrl+S
            logger.info("Changes synced to KiCad. Use Ctrl+S (Cmd+S on Mac) to save.")
            self._dirty = False
            return self.source_path or Path("unknown")

    # =========================================================================
    # Undo/Redo (Delegates to KiCad)
    # =========================================================================

    def checkpoint(self, description: str = ""):
        """
        Create checkpoint for undo.

        In kipy mode, each update_component call creates its own commit
        with the description, so this is largely a no-op. Called for
        API compatibility.
        """
        # In kipy mode, commits are created per-operation
        pass

    def undo(self) -> bool:
        """
        Undo via KiCad's undo system.

        Note: kipy may not expose undo API directly. This refreshes
        the local model from KiCad's current state.
        """
        if not self._connected:
            return False

        try:
            # Refresh local model from KiCad
            self._refresh_from_kicad()
            return True
        except Exception as e:
            logger.error("Undo/refresh failed: %s", e)
            return False

    def redo(self) -> bool:
        """
        Redo via KiCad's redo system.

        Note: kipy may not expose redo API directly. This refreshes
        the local model from KiCad's current state.
        """
        if not self._connected:
            return False

        try:
            self._refresh_from_kicad()
            return True
        except Exception as e:
            logger.error("Redo/refresh failed: %s", e)
            return False

    # =========================================================================
    # Component Updates (Real-time)
    # =========================================================================

    def update_component(self, ref: str, x: float = None, y: float = None,
                         rotation: float = None, locked: bool = None) -> bool:
        """
        Update component position/rotation in KiCad with instant refresh.

        This is the key method for real-time manipulation. Changes appear
        immediately in KiCad's UI and are added to KiCad's undo stack.

        Args:
            ref: Component reference designator
            x: New X position in mm (optional)
            y: New Y position in mm (optional)
            rotation: New rotation in degrees (optional)
            locked: Lock state (optional)

        Returns:
            True if update successful
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        if not self.board:
            raise ValueError("No board loaded. Call load() first.")

        from .kipy_adapter import find_kipy_footprint

        # Find the footprint in kipy
        fp = find_kipy_footprint(self._kipy_board, ref)
        if not fp:
            logger.error("Component not found in KiCad: %s", ref)
            return False

        # Import kipy geometry types
        try:
            from kipy.geometry import Vector2, Angle
            from kipy.util.units import from_mm
        except ImportError:
            from kipy.geometry import Vector2, Angle
            from .kipy_adapter import mm_to_nm as from_mm

        # Begin commit for undo support
        commit = self._kipy_board.begin_commit()

        try:
            # Update position if provided
            if x is not None or y is not None:
                current_pos = fp.position
                new_x = from_mm(x) if x is not None else current_pos.x
                new_y = from_mm(y) if y is not None else current_pos.y
                fp.position = Vector2.from_xy(new_x, new_y)

            # Update rotation if provided
            if rotation is not None:
                fp.orientation = Angle.from_degrees(rotation)

            # Update locked state if provided
            if locked is not None and hasattr(fp, 'locked'):
                fp.locked = locked

            # Apply changes to KiCad (triggers UI refresh)
            self._kipy_board.update_items([fp])
            self._kipy_board.push_commit(commit, f"Move {ref}")

            # Update local model
            if ref in self.board.components:
                comp = self.board.components[ref]
                if x is not None:
                    comp.x = x
                if y is not None:
                    comp.y = y
                if rotation is not None:
                    comp.rotation = rotation
                if locked is not None:
                    comp.locked = locked

            self._dirty = True
            self._dirty_refs.add(ref)

            logger.debug("Updated %s via kipy (instant)", ref)
            return True

        except Exception as e:
            # Rollback on error
            try:
                self._kipy_board.drop_commit(commit)
            except Exception:
                pass
            logger.error("Failed to update %s: %s", ref, e)
            return False

    def update_components(self, updates: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Batch update multiple components in a single transaction.

        All updates are grouped into a single undo step.

        Args:
            updates: List of dicts with keys: ref, x, y, rotation, locked

        Returns:
            Dict mapping ref -> success boolean
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        if not self.board:
            raise ValueError("No board loaded. Call load() first.")

        from .kipy_adapter import find_kipy_footprints

        # Find all footprints
        refs = [u.get('ref') for u in updates if u.get('ref')]
        if not refs:
            return {}

        footprints = find_kipy_footprints(self._kipy_board, refs)

        # Import kipy types
        try:
            from kipy.geometry import Vector2, Angle
            from kipy.util.units import from_mm
        except ImportError:
            from kipy.geometry import Vector2, Angle
            from .kipy_adapter import mm_to_nm as from_mm

        results = {}
        modified_fps = []

        # Process updates first, then commit only if we have changes
        for update in updates:
            ref = update.get('ref')
            if not ref:
                continue

            fp = footprints.get(ref)
            if not fp:
                results[ref] = False
                continue

            # Apply updates
            x = update.get('x')
            y = update.get('y')
            rotation = update.get('rotation')

            if x is not None or y is not None:
                current_pos = fp.position
                new_x = from_mm(x) if x is not None else current_pos.x
                new_y = from_mm(y) if y is not None else current_pos.y
                fp.position = Vector2.from_xy(new_x, new_y)

            if rotation is not None:
                fp.orientation = Angle.from_degrees(rotation)

            modified_fps.append(fp)

            # Update local model
            if ref in self.board.components:
                comp = self.board.components[ref]
                if x is not None:
                    comp.x = x
                if y is not None:
                    comp.y = y
                if rotation is not None:
                    comp.rotation = rotation

            results[ref] = True
            self._dirty_refs.add(ref)

        # Only begin commit if we have changes to apply
        if not modified_fps:
            return results

        commit = self._kipy_board.begin_commit()
        try:
            self._kipy_board.update_items(modified_fps)
            self._kipy_board.push_commit(commit, f"Move {len(modified_fps)} components")
            self._dirty = True
            return results

        except Exception as e:
            # Rollback on error
            try:
                self._kipy_board.drop_commit(commit)
            except Exception:
                pass
            logger.error("Batch update failed: %s", e)
            return {ref: False for ref in refs}

    def mark_modified(self, refs: List[str]):
        """
        Mark components as modified and immediately sync to KiCad.

        In kipy mode, this triggers an instant sync for live feedback.
        In other modes, just marks dirty for later save.
        """
        self._dirty = True
        self._dirty_refs.update(refs)

        # Immediately sync to KiCad for live updates
        if self._connected and self._kipy_board:
            try:
                self._sync_to_kicad()
            except Exception as e:
                logger.warning("Failed to auto-sync to KiCad: %s", e)

    def clear_dirty(self):
        """Clear dirty state."""
        self._dirty_refs.clear()

    # =========================================================================
    # Internal Sync Methods
    # =========================================================================

    def _sync_to_kicad(self):
        """Sync local board changes to KiCad.

        Uses batch update to apply all changes in a single commit for
        better undo behavior and atomicity.
        """
        if not self._dirty_refs or not self._connected or not self.board:
            return

        # Build batch updates from dirty components
        updates = []
        for ref in list(self._dirty_refs):
            comp = self.board.components.get(ref)
            if comp:
                updates.append({
                    'ref': ref,
                    'x': comp.x,
                    'y': comp.y,
                    'rotation': comp.rotation
                })

        if updates:
            try:
                # Use batch update for single commit
                self.update_components(updates)
            except Exception as e:
                logger.error("Failed to sync batch: %s", e)

        self._dirty_refs.clear()

    def _refresh_from_kicad(self):
        """Refresh local model from KiCad's current state."""
        if not self._connected or not self._kicad:
            return

        try:
            self._kipy_board = self._kicad.get_board()
            from .kipy_adapter import kipy_board_to_atoplace
            self.board = kipy_board_to_atoplace(self._kipy_board)
            logger.debug("Refreshed board from KiCad")
        except Exception as e:
            logger.error("Failed to refresh from KiCad: %s", e)

    # =========================================================================
    # Session Info
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "loaded": self.is_loaded,
            "source": str(self.source_path) if self.source_path else None,
            "dirty": self.is_dirty,
            "dirty_count": len(self._dirty_refs),
            "undo_available": True,  # KiCad manages undo
            "redo_available": True,
            "components": len(self.board.components) if self.board else 0,
            "mode": "kipy",
            "connected": self._connected,
        }

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        board_info = f"{len(self.board.components)} components" if self.board else "no board"
        return f"KiPySession({status}, {board_info})"
