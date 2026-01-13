"""
AtoPlace Session State Management

Manages board state persistence including load/save/undo functionality.
Tracks "dirty" state to trigger re-running legalization/routing when needed.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy
import json

from ..board.abstraction import Board


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


class Session:
    """
    Manages the lifecycle of a board editing session.

    Provides:
    - Load/save operations
    - Undo/redo stack
    - Dirty state tracking for incremental updates
    """

    MAX_UNDO_STACK = 50

    def __init__(self):
        self.board: Optional[Board] = None
        self.source_path: Optional[Path] = None
        self._undo_stack: List[BoardSnapshot] = []
        self._redo_stack: List[BoardSnapshot] = []
        self._dirty: bool = False
        self._dirty_refs: set = set()  # Components modified since last legalization

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
        """Get list of components modified since last save/legalize."""
        return list(self._dirty_refs)

    def load(self, path: Path) -> "Session":
        """
        Load a board from file.

        Args:
            path: Path to KiCad PCB file

        Returns:
            Self for chaining
        """
        self.board = Board.from_kicad(path)
        self.source_path = path
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._dirty_refs.clear()

        # Take initial snapshot
        self._save_snapshot("Initial load")

        return self

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save board to file.

        Args:
            path: Output path. If None, uses source path with .placed suffix.

        Returns:
            Path where board was saved
        """
        if not self.board:
            raise ValueError("No board loaded")

        if path is None:
            if self.source_path:
                stem = self.source_path.stem
                if not stem.endswith('.placed'):
                    stem = f"{stem}.placed"
                path = self.source_path.parent / f"{stem}.kicad_pcb"
            else:
                raise ValueError("No path specified and no source path")

        self.board.to_kicad(path)
        self._dirty = False

        return path

    def checkpoint(self, description: str = ""):
        """
        Create a checkpoint for undo.

        Call this before making changes that should be undoable as a unit.
        """
        if not self.board:
            return
        self._save_snapshot(description)

    def undo(self) -> bool:
        """
        Undo last change.

        Returns:
            True if undo was performed, False if nothing to undo
        """
        if len(self._undo_stack) <= 1:  # Keep at least the initial state
            return False

        # Save current state to redo
        self._redo_stack.append(self._take_snapshot("Redo point"))

        # Pop and discard current, then restore previous
        self._undo_stack.pop()
        if self._undo_stack:
            self._restore_snapshot(self._undo_stack[-1])

        self._dirty = True
        return True

    def redo(self) -> bool:
        """
        Redo last undone change.

        Returns:
            True if redo was performed, False if nothing to redo
        """
        if not self._redo_stack:
            return False

        snapshot = self._redo_stack.pop()
        self._undo_stack.append(snapshot)
        self._restore_snapshot(snapshot)

        self._dirty = True
        return True

    def mark_modified(self, refs: List[str]):
        """
        Mark components as modified (for dirty tracking).

        Args:
            refs: List of component references that were modified
        """
        self._dirty = True
        self._dirty_refs.update(refs)

    def clear_dirty(self):
        """Clear dirty state (call after legalization/routing)."""
        self._dirty_refs.clear()

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

        # Clear redo stack on new action
        self._redo_stack.clear()

        # Limit stack size
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
            "dirty": self.is_dirty,
            "dirty_count": len(self._dirty_refs),
            "undo_available": len(self._undo_stack) > 1,
            "redo_available": len(self._redo_stack) > 0,
            "components": len(self.board.components) if self.board else 0,
        }
