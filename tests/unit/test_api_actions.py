"""
Tests for the AtoPlace Layout Actions API.

Tests the atomic placement operations provided by LayoutActions.
"""

import pytest
import math

from atoplace.api.actions import LayoutActions, ActionResult


class TestMoveAbsolute:
    """Test move_absolute action."""

    def test_move_to_position(self, test_board):
        """Moving to absolute position should update coordinates."""
        actions = LayoutActions(test_board)
        result = actions.move_absolute("U1", 75.0, 75.0)

        assert result.success is True
        assert "U1" in result.modified_refs
        assert test_board.components["U1"].x == 75.0
        assert test_board.components["U1"].y == 75.0

    def test_move_with_rotation(self, test_board):
        """Moving with rotation should update both position and rotation."""
        actions = LayoutActions(test_board)
        result = actions.move_absolute("U1", 75.0, 75.0, 90.0)

        assert result.success is True
        assert test_board.components["U1"].rotation == 90.0

    def test_move_nonexistent_component(self, test_board):
        """Moving nonexistent component should fail."""
        actions = LayoutActions(test_board)
        result = actions.move_absolute("NONEXISTENT", 75.0, 75.0)

        assert result.success is False
        assert "not found" in result.message.lower()

    def test_move_locked_component(self, board_with_locked):
        """Moving locked component should fail."""
        actions = LayoutActions(board_with_locked)
        result = actions.move_absolute("U2", 75.0, 75.0)

        assert result.success is False
        assert "locked" in result.message.lower()


class TestMoveRelative:
    """Test move_relative action."""

    def test_move_by_delta(self, test_board):
        """Moving by delta should add to current position."""
        actions = LayoutActions(test_board)
        original_x = test_board.components["U1"].x
        original_y = test_board.components["U1"].y

        result = actions.move_relative("U1", 10.0, -5.0)

        assert result.success is True
        assert test_board.components["U1"].x == original_x + 10.0
        assert test_board.components["U1"].y == original_y - 5.0

    def test_move_relative_nonexistent(self, test_board):
        """Moving nonexistent component should fail."""
        actions = LayoutActions(test_board)
        result = actions.move_relative("NONEXISTENT", 10.0, 10.0)

        assert result.success is False


class TestRotate:
    """Test rotate action."""

    def test_set_rotation(self, test_board):
        """Setting rotation should update rotation value."""
        actions = LayoutActions(test_board)
        result = actions.rotate("U1", 45.0)

        assert result.success is True
        assert test_board.components["U1"].rotation == 45.0

    def test_rotation_wraps(self, test_board):
        """Rotation should wrap at 360 degrees."""
        actions = LayoutActions(test_board)
        result = actions.rotate("U1", 450.0)

        assert result.success is True
        assert test_board.components["U1"].rotation == 90.0

    def test_rotate_locked_component(self, board_with_locked):
        """Rotating locked component should fail."""
        actions = LayoutActions(board_with_locked)
        result = actions.rotate("U2", 90.0)

        assert result.success is False
        assert "locked" in result.message.lower()


class TestPlaceNextTo:
    """Test place_next_to action."""

    def test_place_right(self, test_board):
        """Placing to the right should position correctly."""
        actions = LayoutActions(test_board)
        target_x = test_board.components["U1"].x
        target_width = test_board.components["U1"].width
        source_width = test_board.components["C1"].width
        clearance = 1.0

        result = actions.place_next_to("C1", "U1", "right", clearance)

        assert result.success is True
        expected_x = target_x + (target_width / 2) + clearance + (source_width / 2)
        assert abs(test_board.components["C1"].x - expected_x) < 0.01

    def test_place_left(self, test_board):
        """Placing to the left should position correctly."""
        actions = LayoutActions(test_board)
        result = actions.place_next_to("C1", "U1", "left", 1.0)

        assert result.success is True
        assert test_board.components["C1"].x < test_board.components["U1"].x

    def test_place_top(self, test_board):
        """Placing on top should position correctly."""
        actions = LayoutActions(test_board)
        result = actions.place_next_to("C1", "U1", "top", 1.0)

        assert result.success is True
        assert test_board.components["C1"].y < test_board.components["U1"].y

    def test_place_bottom(self, test_board):
        """Placing on bottom should position correctly."""
        actions = LayoutActions(test_board)
        result = actions.place_next_to("C1", "U1", "bottom", 1.0)

        assert result.success is True
        assert test_board.components["C1"].y > test_board.components["U1"].y

    def test_place_with_alignment(self, test_board):
        """Placing with alignment should align correctly."""
        actions = LayoutActions(test_board)
        result = actions.place_next_to("C1", "U1", "right", 1.0, "center")

        assert result.success is True
        # Center alignment means same Y coordinate
        assert abs(test_board.components["C1"].y - test_board.components["U1"].y) < 0.01

    def test_place_nonexistent_source(self, test_board):
        """Placing nonexistent source should fail."""
        actions = LayoutActions(test_board)
        result = actions.place_next_to("NONEXISTENT", "U1", "right", 1.0)

        assert result.success is False

    def test_place_nonexistent_target(self, test_board):
        """Placing next to nonexistent target should fail."""
        actions = LayoutActions(test_board)
        result = actions.place_next_to("C1", "NONEXISTENT", "right", 1.0)

        assert result.success is False


class TestAlignComponents:
    """Test align_components action."""

    def test_align_x_axis(self, test_board):
        """Aligning on X axis should make Y coordinates match."""
        actions = LayoutActions(test_board)
        result = actions.align_components(["R1", "C1"], "x", "first")

        assert result.success is True
        assert test_board.components["R1"].y == test_board.components["C1"].y

    def test_align_y_axis(self, test_board):
        """Aligning on Y axis should make X coordinates match."""
        actions = LayoutActions(test_board)
        result = actions.align_components(["R1", "C1"], "y", "first")

        assert result.success is True
        assert test_board.components["R1"].x == test_board.components["C1"].x

    def test_align_to_center(self, test_board):
        """Aligning to center should average positions."""
        actions = LayoutActions(test_board)
        r1_y = test_board.components["R1"].y
        c1_y = test_board.components["C1"].y
        expected_y = (r1_y + c1_y) / 2

        result = actions.align_components(["R1", "C1"], "x", "center")

        assert result.success is True
        assert abs(test_board.components["R1"].y - expected_y) < 0.01
        assert abs(test_board.components["C1"].y - expected_y) < 0.01

    def test_align_single_component(self, test_board):
        """Aligning single component should fail."""
        actions = LayoutActions(test_board)
        result = actions.align_components(["R1"], "x", "first")

        assert result.success is False
        assert "at least 2" in result.message.lower()

    def test_align_with_nonexistent(self, test_board):
        """Aligning with nonexistent component should fail."""
        actions = LayoutActions(test_board)
        result = actions.align_components(["R1", "NONEXISTENT"], "x", "first")

        assert result.success is False


class TestGetDims:
    """Test dimension calculation helper."""

    def test_normal_orientation(self, test_board):
        """Normal orientation should return width, height."""
        actions = LayoutActions(test_board)
        comp = test_board.components["U1"]
        comp.rotation = 0

        w, h = actions._get_dims(comp)
        assert w == comp.width
        assert h == comp.height

    def test_rotated_90(self, test_board):
        """90 degree rotation should swap dimensions."""
        actions = LayoutActions(test_board)
        comp = test_board.components["U1"]
        comp.rotation = 90

        w, h = actions._get_dims(comp)
        assert w == comp.height
        assert h == comp.width

    def test_rotated_180(self, test_board):
        """180 degree rotation should keep dimensions."""
        actions = LayoutActions(test_board)
        comp = test_board.components["U1"]
        comp.rotation = 180

        w, h = actions._get_dims(comp)
        assert w == comp.width
        assert h == comp.height
