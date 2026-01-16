"""
Tests for the AtoPlace Session API.

Tests session state management, undo/redo, and dirty tracking.
"""

import pytest
from pathlib import Path

from atoplace.api.session import Session, BoardSnapshot, ComponentState


class TestSessionBasics:
    """Test basic session properties."""

    def test_new_session_not_loaded(self):
        """New session should have no board loaded."""
        session = Session()
        assert session.is_loaded is False
        assert session.board is None

    def test_session_after_board_set(self, test_board):
        """Session with board should be loaded."""
        session = Session()
        session.board = test_board
        assert session.is_loaded is True

    def test_dirty_initially_false(self, test_board):
        """New session should not be dirty."""
        session = Session()
        session.board = test_board
        session._dirty = False
        assert session.is_dirty is False


class TestCheckpoint:
    """Test checkpoint functionality."""

    def test_checkpoint_creates_snapshot(self, test_board):
        """Checkpoint should add to undo stack."""
        session = Session()
        session.board = test_board
        initial_stack_len = len(session._undo_stack)

        session.checkpoint("Test checkpoint")

        assert len(session._undo_stack) == initial_stack_len + 1

    def test_checkpoint_clears_redo(self, test_board):
        """Checkpoint should clear redo stack."""
        session = Session()
        session.board = test_board
        session._redo_stack.append(BoardSnapshot(component_states={}, description="test"))

        session.checkpoint("Test checkpoint")

        assert len(session._redo_stack) == 0


class TestUndoRedo:
    """Test undo/redo functionality."""

    def test_undo_restores_position(self, test_board):
        """Undo should restore previous position."""
        session = Session()
        session.board = test_board

        # Save initial state
        session.checkpoint("Initial")
        original_x = test_board.components["U1"].x

        # Modify
        session.checkpoint("Move")
        test_board.components["U1"].x = 99.0

        # Undo
        result = session.undo()

        assert result is True
        assert test_board.components["U1"].x == original_x

    def test_undo_nothing_to_undo(self):
        """Undo with empty stack should return False."""
        session = Session()
        result = session.undo()
        assert result is False

    def test_redo_after_undo(self, test_board):
        """Redo should restore undone change."""
        session = Session()
        session.board = test_board

        # Save initial state
        session.checkpoint("Initial")

        # Modify
        session.checkpoint("Move")
        new_x = 99.0
        test_board.components["U1"].x = new_x

        # Save after modification
        session.checkpoint("After move")

        # Undo
        session.undo()

        # Redo
        result = session.redo()

        assert result is True
        assert test_board.components["U1"].x == new_x

    def test_redo_nothing_to_redo(self, test_board):
        """Redo with empty stack should return False."""
        session = Session()
        session.board = test_board
        result = session.redo()
        assert result is False

    def test_max_undo_stack(self, test_board):
        """Undo stack should be limited to MAX_UNDO_STACK."""
        session = Session()
        session.board = test_board

        # Create many checkpoints
        for i in range(session.MAX_UNDO_STACK + 10):
            session.checkpoint(f"Checkpoint {i}")

        assert len(session._undo_stack) <= session.MAX_UNDO_STACK


class TestDirtyTracking:
    """Test dirty state tracking."""

    def test_mark_modified_sets_dirty(self, test_board):
        """Marking modified should set dirty flag."""
        session = Session()
        session.board = test_board
        session._dirty = False

        session.mark_modified(["U1"])

        assert session.is_dirty is True

    def test_mark_modified_tracks_refs(self, test_board):
        """Marking modified should track refs."""
        session = Session()
        session.board = test_board

        session.mark_modified(["U1", "R1"])

        assert "U1" in session.dirty_components
        assert "R1" in session.dirty_components

    def test_clear_dirty(self, test_board):
        """Clearing dirty should reset dirty refs."""
        session = Session()
        session.board = test_board
        session.mark_modified(["U1", "R1"])

        session.clear_dirty()

        assert len(session.dirty_components) == 0


class TestSnapshot:
    """Test snapshot functionality."""

    def test_take_snapshot_captures_positions(self, test_board):
        """Taking snapshot should capture all positions."""
        session = Session()
        session.board = test_board

        snapshot = session._take_snapshot("Test")

        assert "U1" in snapshot.component_states
        assert snapshot.component_states["U1"].x == test_board.components["U1"].x
        assert snapshot.component_states["U1"].y == test_board.components["U1"].y

    def test_restore_snapshot_updates_positions(self, test_board):
        """Restoring snapshot should update positions."""
        session = Session()
        session.board = test_board

        # Take snapshot
        snapshot = session._take_snapshot("Test")
        original_x = test_board.components["U1"].x

        # Modify
        test_board.components["U1"].x = 99.0

        # Restore
        session._restore_snapshot(snapshot)

        assert test_board.components["U1"].x == original_x


class TestGetStats:
    """Test session statistics."""

    def test_stats_not_loaded(self):
        """Stats should show not loaded state."""
        session = Session()
        stats = session.get_stats()

        assert stats["loaded"] is False
        assert stats["components"] == 0

    def test_stats_loaded(self, test_board):
        """Stats should show loaded state."""
        session = Session()
        session.board = test_board
        session.source_path = Path("/tmp/test.kicad_pcb")
        session._undo_stack = [session._take_snapshot("Initial")]
        session.mark_modified(["U1"])

        stats = session.get_stats()

        assert stats["loaded"] is True
        assert stats["source"] == "/tmp/test.kicad_pcb"
        assert stats["dirty"] is True
        assert stats["dirty_count"] == 1
        assert stats["components"] == len(test_board.components)
