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

    # Note: swap_positions is not implemented in the current server API


class TestDiscoveryTools:
    """Test discovery tools."""

    def test_find_components_by_ref(self, mcp_session, monkeypatch):
        """Finding components by ref should work."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.find_components("U", "ref"))
        # Response uses pagination format (no status field for success)
        assert "total_count" in result
        assert result["count"] >= 1

    def test_find_components_by_value(self, mcp_session, monkeypatch):
        """Finding components by value should work."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.find_components("10k", "value"))
        # Response uses pagination format (no status field for success)
        assert "total_count" in result

    def test_find_components_invalid_filter(self, mcp_session, monkeypatch):
        """Finding with invalid filter should fail."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.find_components("test", "invalid_field"))
        assert result["status"] == "error"
        assert result["code"] == "invalid_filter"

    # Note: get_board_bounds is not implemented in the current server API
    # Board info is available via get_board_summary

    def test_get_unplaced_components(self, mcp_session, monkeypatch):
        """Getting unplaced components should return list."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_unplaced_components())
        # Response uses pagination format (no status field for success)
        assert "refs" in result
        assert "total_count" in result
        assert "count" in result


class TestTopologyTools:
    """Test topology tools."""

    # Note: get_connected_components and get_critical_nets are not implemented
    # in the current server API. Connectivity information is available via
    # inspect_region and detect_modules tools.


class TestContextTools:
    """Test context generation tools."""

    def test_inspect_region(self, mcp_session, monkeypatch):
        """Inspecting region should return correct geometry data."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.inspect_region(["U1", "R1"]))

        # Verify structure
        assert "viewport" in result
        assert "objects" in result

        # Verify viewport structure (center and size)
        # U1 is at (50, 50) with size (4, 5)
        # R1 is at (45, 55) with size (1.6, 0.8)
        viewport = result["viewport"]
        assert "center" in viewport
        assert "size" in viewport
        assert len(viewport["center"]) == 2  # (x, y) tuple
        assert len(viewport["size"]) == 2   # (width, height) tuple

        # Verify viewport encompasses both components
        center_x, center_y = viewport["center"]
        width, height = viewport["size"]
        # Calculate bounds from center and size
        min_x = center_x - width / 2
        max_x = center_x + width / 2
        min_y = center_y - height / 2
        max_y = center_y + height / 2

        # Viewport should include both components (with padding)
        assert min_x < 45  # Should include R1 left edge
        assert max_x > 50  # Should include U1 right edge
        assert min_y < 50  # Should include U1 bottom edge
        assert max_y > 55  # Should include R1 top edge

        # Verify objects contain correct components
        assert len(result["objects"]) >= 2
        refs = [obj["ref"] for obj in result["objects"]]
        assert "U1" in refs
        assert "R1" in refs

        # Verify component data correctness
        # ObjectView uses "location" (tuple), not separate x/y
        u1_obj = next(obj for obj in result["objects"] if obj["ref"] == "U1")
        assert "location" in u1_obj
        assert u1_obj["location"][0] == 50.0  # x coordinate
        assert u1_obj["location"][1] == 50.0  # y coordinate
        assert u1_obj["type"] == "IC"

        r1_obj = next(obj for obj in result["objects"] if obj["ref"] == "R1")
        assert "location" in r1_obj
        assert r1_obj["location"][0] == 45.0  # x coordinate
        assert r1_obj["location"][1] == 55.0  # y coordinate
        assert r1_obj["type"] == "Resistor"

    def test_get_board_summary(self, mcp_session, monkeypatch):
        """Getting board summary should return correct statistics."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.get_board_summary())

        # Verify structure
        assert "component_count" in result
        assert "net_count" in result

        # Verify correct counts from test_board fixture
        # test_board has: U1, R1, C1, J1 = 4 components
        # test_board has: VCC, GND, PB0 = 3 nets
        assert result["component_count"] == 4
        assert result["net_count"] == 3

        # Verify board dimensions if present
        if "board_area" in result:
            # Board is 100x100 mm = 10000 mm²
            assert result["board_area"] == 10000.0

    # Note: get_semantic_grid and get_module_map are not implemented in the
    # current server API. Use detect_modules for module hierarchy information.

    # Note: render_region is not implemented in the current server API.
    # Use inspect_region with include_image=True for SVG visualization.


class TestValidationTools:
    """Test validation tools."""

    def test_check_overlaps_no_overlaps(self, mcp_session, monkeypatch):
        """Checking overlaps on non-overlapping board."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.check_overlaps())
        assert result["status"] == "success"
        # Response uses pagination format with total_count instead of overlap_count
        assert "total_count" in result
        assert "overlaps" in result

    def test_check_overlaps_specific_refs(self, mcp_session, monkeypatch):
        """Checking overlaps for specific refs."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.check_overlaps(["U1", "R1"]))
        assert result["status"] == "success"

    def test_validate_placement(self, mcp_session, monkeypatch):
        """Validating placement should return scores and flags."""
        monkeypatch.setattr(server, "session", mcp_session)
        result = json.loads(server.validate_placement())
        # Response format includes scores and paginated flags
        assert "overall_score" in result
        assert "placement_score" in result
        assert "flags" in result


class TestResources:
    """Test MCP resources."""

    def test_system_prompt_resource(self):
        """System prompt resource should return the system prompt."""
        result = server.system_prompt_resource()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fix_overlaps_prompt_resource(self):
        """Fix overlaps prompt resource should return the prompt."""
        result = server.fix_overlaps_prompt_resource()
        assert isinstance(result, str)
        assert len(result) > 0

    # Note: board_summary_resource and board_modules_resource are not implemented
    # in the current server API. Resources are exposed via MCP @resource decorator
    # and accessed via prompts://system and prompts://fix_overlaps URIs.
