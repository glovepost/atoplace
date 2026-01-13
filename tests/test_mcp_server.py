"""
Tests for the AtoPlace MCP Server.

Tests cover all MCP tools: board management, placement actions,
discovery, topology, context generation, and validation.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import server components
from atoplace.mcp import server
from atoplace.api.session import Session


class TestResponseHelpers:
    """Test the response helper functions."""

    def test_error_response_format(self):
        """Error responses should have consistent format."""
        result = json.loads(server._error_response("Test error", "test_code"))
        assert result["status"] == "error"
        assert result["code"] == "test_code"
        assert result["message"] == "Test error"

    def test_success_response_format(self):
        """Success responses should have consistent format."""
        result = json.loads(server._success_response({"key": "value"}))
        assert result["status"] == "success"
        assert result["key"] == "value"


class TestValidationHelpers:
    """Test the validation helper functions."""

    def test_validate_ref_empty(self, mcp_session, monkeypatch):
        """Empty ref should return error."""
        monkeypatch.setattr(server, "session", mcp_session)
        assert server._validate_ref("") is not None
        assert "empty" in server._validate_ref("").lower()

    def test_validate_ref_not_found(self, mcp_session, monkeypatch):
        """Non-existent ref should return error with hints."""
        monkeypatch.setattr(server, "session", mcp_session)
        error = server._validate_ref("NONEXISTENT")
        assert error is not None
        assert "not found" in error.lower()

    def test_validate_ref_valid(self, mcp_session, monkeypatch):
        """Valid ref should return None."""
        monkeypatch.setattr(server, "session", mcp_session)
        assert server._validate_ref("U1") is None

    def test_validate_side_invalid(self):
        """Invalid side should return error."""
        error = server._validate_side("diagonal")
        assert error is not None
        assert "top" in error.lower()

    def test_validate_side_valid(self):
        """Valid sides should return None."""
        assert server._validate_side("top") is None
        assert server._validate_side("bottom") is None
        assert server._validate_side("left") is None
        assert server._validate_side("right") is None

    def test_validate_axis_invalid(self):
        """Invalid axis should return error."""
        error = server._validate_axis("z")
        assert error is not None

    def test_validate_axis_valid(self):
        """Valid axes should return None."""
        assert server._validate_axis("x") is None
        assert server._validate_axis("y") is None


class TestBoardManagement:
    """Test board management tools."""

    def test_require_board_without_load(self, empty_session, monkeypatch):
        """Operations should fail without loaded board."""
        monkeypatch.setattr(server, "session", empty_session)
        with pytest.raises(ValueError, match="No board loaded"):
            server._require_board()

    def test_require_board_with_load(self, mcp_session, monkeypatch):
        """Operations should succeed with loaded board."""
        monkeypatch.setattr(server, "session", mcp_session)
        server._require_board()  # Should not raise

    def test_undo_nothing_to_undo(self, mcp_session, monkeypatch):
        """Undo with nothing to undo should report correctly."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.undo())
        assert result["status"] == "success"
        assert result["action"] == "nothing_to_undo"

    def test_redo_nothing_to_redo(self, mcp_session, monkeypatch):
        """Redo with nothing to redo should report correctly."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.redo())
        assert result["status"] == "success"
        assert result["action"] == "nothing_to_redo"


class TestPlacementActions:
    """Test placement action tools."""

    def test_move_component_success(self, mcp_session, monkeypatch):
        """Moving a component should succeed."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.move_component("U1", 60.0, 60.0))
        assert result["status"] == "success"
        assert result["success"] is True
        assert mcp_session.board.components["U1"].x == 60.0
        assert mcp_session.board.components["U1"].y == 60.0

    def test_move_component_with_rotation(self, mcp_session, monkeypatch):
        """Moving with rotation should update rotation."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.move_component("U1", 60.0, 60.0, 90.0))
        assert result["success"] is True
        assert mcp_session.board.components["U1"].rotation == 90.0

    def test_move_component_invalid_ref(self, mcp_session, monkeypatch):
        """Moving invalid ref should return error."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.move_component("INVALID", 60.0, 60.0))
        assert result["status"] == "error"
        assert result["code"] == "invalid_ref"

    def test_move_locked_component(self, board_with_locked, monkeypatch):
        """Moving locked component should fail."""
        session = Session()
        session.board = board_with_locked
        session.source_path = Path("/tmp/test.kicad_pcb")
        session._undo_stack = [session._take_snapshot("Initial")]
        monkeypatch.setattr(server, "session", session)

        result = json.loads(server.move_component("U2", 70.0, 70.0))
        assert result["success"] is False
        assert "locked" in result["message"].lower()

    def test_place_next_to_success(self, mcp_session, monkeypatch):
        """Place next to should position component correctly."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.place_next_to("C1", "U1", "right", 1.0))
        assert result["success"] is True

    def test_place_next_to_invalid_side(self, mcp_session, monkeypatch):
        """Place next to with invalid side should fail."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.place_next_to("C1", "U1", "diagonal", 1.0))
        assert result["status"] == "error"
        assert result["code"] == "invalid_side"

    def test_align_components_success(self, mcp_session, monkeypatch):
        """Aligning components should succeed."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.align_components(["R1", "C1"], "x", "first"))
        assert result["success"] is True

    def test_align_components_invalid_axis(self, mcp_session, monkeypatch):
        """Aligning with invalid axis should fail."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.align_components(["R1", "C1"], "z", "first"))
        assert result["status"] == "error"
        assert result["code"] == "invalid_axis"

    def test_swap_positions_success(self, mcp_session, monkeypatch):
        """Swapping positions should succeed."""
        monkeypatch.setattr(server, "session", mcp_session)
        orig_r1_x = mcp_session.board.components["R1"].x
        orig_c1_x = mcp_session.board.components["C1"].x

        result = json.loads(server.swap_positions("R1", "C1"))
        assert result["success"] is True
        assert mcp_session.board.components["R1"].x == orig_c1_x
        assert mcp_session.board.components["C1"].x == orig_r1_x

    def test_swap_positions_locked_component(self, board_with_locked, monkeypatch):
        """Swapping with locked component should fail."""
        session = Session()
        session.board = board_with_locked
        session.source_path = Path("/tmp/test.kicad_pcb")
        monkeypatch.setattr(server, "session", session)

        result = json.loads(server.swap_positions("U1", "U2"))
        assert result["status"] == "error"
        assert "locked" in result["message"].lower()


class TestDiscoveryTools:
    """Test discovery tools."""

    def test_find_components_by_ref(self, mcp_session, monkeypatch):
        """Finding components by ref should work."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.find_components("U", "ref"))
        assert result["status"] == "success"
        assert result["count"] >= 1

    def test_find_components_by_value(self, mcp_session, monkeypatch):
        """Finding components by value should work."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.find_components("10k", "value"))
        assert result["status"] == "success"

    def test_find_components_invalid_filter(self, mcp_session, monkeypatch):
        """Finding with invalid filter should fail."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.find_components("test", "invalid_field"))
        assert result["status"] == "error"
        assert result["code"] == "invalid_filter"

    def test_get_board_bounds(self, mcp_session, monkeypatch):
        """Getting board bounds should return dimensions."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_board_bounds())
        assert result["status"] == "success"
        assert "width" in result
        assert "height" in result
        assert result["component_count"] > 0

    def test_get_unplaced_components(self, mcp_session, monkeypatch):
        """Getting unplaced components should return list."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_unplaced_components())
        assert result["status"] == "success"
        assert "refs" in result
        assert "count" in result


class TestTopologyTools:
    """Test topology tools."""

    def test_get_connected_components(self, mcp_session, monkeypatch):
        """Getting connected components should return connections."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_connected_components("U1"))
        assert result["status"] == "success"
        assert result["ref"] == "U1"
        assert "connections" in result

    def test_get_connected_components_invalid_ref(self, mcp_session, monkeypatch):
        """Getting connections for invalid ref should fail."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_connected_components("INVALID"))
        assert result["status"] == "error"
        assert result["code"] == "invalid_ref"

    def test_get_critical_nets(self, mcp_session, monkeypatch):
        """Getting critical nets should return power/ground nets."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_critical_nets())
        assert result["status"] == "success"
        assert "critical_nets" in result


class TestContextTools:
    """Test context generation tools."""

    def test_inspect_region(self, mcp_session, monkeypatch):
        """Inspecting region should return geometry data."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.inspect_region(["U1", "R1"]))
        assert "viewport" in result
        assert "objects" in result

    def test_get_board_summary(self, mcp_session, monkeypatch):
        """Getting board summary should return statistics."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_board_summary())
        assert "component_count" in result
        assert "net_count" in result

    def test_get_semantic_grid(self, mcp_session, monkeypatch):
        """Getting semantic grid should return zone mapping."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_semantic_grid())
        assert "zones" in result
        assert "zone_counts" in result

    def test_get_module_map(self, mcp_session, monkeypatch):
        """Getting module map should return hierarchy."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_module_map())
        assert "root" in result
        assert "flat_modules" in result

    def test_render_region(self, mcp_session, monkeypatch):
        """Rendering region should return SVG."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.render_region(["U1", "R1"]))
        assert result["status"] == "success"
        assert "svg" in result
        assert "<svg" in result["svg"]


class TestValidationTools:
    """Test validation tools."""

    def test_check_overlaps_no_overlaps(self, mcp_session, monkeypatch):
        """Checking overlaps on non-overlapping board."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.check_overlaps())
        assert result["status"] == "success"
        assert "overlap_count" in result

    def test_check_overlaps_specific_refs(self, mcp_session, monkeypatch):
        """Checking overlaps for specific refs."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.check_overlaps(["U1", "R1"]))
        assert result["status"] == "success"

    def test_validate_placement(self, mcp_session, monkeypatch):
        """Validating placement should return issues."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.validate_placement())
        assert result["status"] == "success"
        assert "valid" in result
        assert "issues" in result


class TestResources:
    """Test MCP resources."""

    def test_board_summary_resource_no_board(self, empty_session, monkeypatch):
        """Board summary resource without board should return error."""
        monkeypatch.setattr(server, "session", empty_session)
        result = json.loads(server.board_summary_resource())
        assert result["status"] == "error"
        assert result["code"] == "no_board"

    def test_board_modules_resource_no_board(self, empty_session, monkeypatch):
        """Board modules resource without board should return error."""
        monkeypatch.setattr(server, "session", empty_session)
        result = json.loads(server.board_modules_resource())
        assert result["status"] == "error"
        assert result["code"] == "no_board"
