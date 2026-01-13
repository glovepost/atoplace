"""
Tests for the AtoPlace Context Generators.

Tests the Microscope (micro), MacroContext (macro), and VisionContext (vision)
context generators used by the MCP server.
"""

import pytest
import json

from atoplace.mcp.context.micro import Microscope, MicroscopeData
from atoplace.mcp.context.macro import MacroContext, BoardSummary, SemanticGrid, ModuleMap
from atoplace.mcp.context.vision import VisionContext


class TestMicroscope:
    """Test Microscope context generator."""

    def test_inspect_region_returns_data(self, test_board):
        """Inspecting region should return MicroscopeData."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region(["U1", "R1"])

        assert isinstance(data, MicroscopeData)
        assert data.viewport is not None
        assert len(data.objects) == 2

    def test_viewport_has_size(self, test_board):
        """Viewport should have positive size."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region(["U1"])

        assert data.viewport.size[0] > 0
        assert data.viewport.size[1] > 0

    def test_objects_have_positions(self, test_board):
        """Objects should have location data."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region(["U1"])

        obj = data.objects[0]
        assert obj.ref == "U1"
        assert obj.location is not None
        assert obj.bbox is not None

    def test_gap_calculation(self, test_board):
        """Gaps between components should be calculated."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region(["U1", "R1"])

        assert len(data.gaps) == 1
        gap = data.gaps[0]
        assert "U1" in gap.between or "R1" in gap.between

    def test_empty_refs_returns_empty(self, test_board):
        """Empty refs should return empty data."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region([])

        assert len(data.objects) == 0

    def test_grid_alignment_check(self, test_board):
        """Grid alignment should be checked."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region(["U1"])

        # grid_aligned should be boolean
        assert isinstance(data.grid_aligned, bool)

    def test_to_json_serialization(self, test_board):
        """Data should serialize to valid JSON."""
        microscope = Microscope(test_board)
        data = microscope.inspect_region(["U1", "R1"])
        json_str = data.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "viewport" in parsed
        assert "objects" in parsed
        assert "gaps" in parsed


class TestMacroContext:
    """Test MacroContext context generator."""

    def test_get_summary(self, test_board):
        """Getting summary should return BoardSummary."""
        macro = MacroContext(test_board)
        summary = macro.get_summary()

        assert isinstance(summary, BoardSummary)
        assert summary.component_count == len(test_board.components)
        assert summary.net_count == len(test_board.nets)

    def test_summary_detects_power_nets(self, test_board):
        """Summary should detect power nets."""
        macro = MacroContext(test_board)
        summary = macro.get_summary()

        # VCC should be detected as power
        assert len(summary.power_nets) > 0 or len(summary.ground_nets) > 0

    def test_get_semantic_grid(self, test_board):
        """Getting semantic grid should return zone mapping."""
        macro = MacroContext(test_board)
        grid = macro.get_semantic_grid()

        assert isinstance(grid, SemanticGrid)
        assert "zones" in grid.__dict__
        assert len(grid.zones) == 9  # 3x3 grid

    def test_zones_contain_components(self, test_board):
        """Zone mapping should contain component refs."""
        macro = MacroContext(test_board)
        grid = macro.get_semantic_grid()

        # Flatten all zones
        all_refs = []
        for zone_refs in grid.zones.values():
            all_refs.extend(zone_refs)

        # All components should be in some zone
        for ref in test_board.components:
            assert ref in all_refs

    def test_get_module_map(self, test_board):
        """Getting module map should return hierarchy."""
        macro = MacroContext(test_board)
        modules = macro.get_module_map()

        assert isinstance(modules, ModuleMap)
        assert modules.root is not None
        assert modules.root.name == "Board"

    def test_summary_to_json(self, test_board):
        """Summary should serialize to JSON."""
        macro = MacroContext(test_board)
        summary = macro.get_summary()
        json_str = summary.to_json()

        parsed = json.loads(json_str)
        assert "component_count" in parsed
        assert "net_count" in parsed

    def test_grid_to_json(self, test_board):
        """Grid should serialize to JSON."""
        macro = MacroContext(test_board)
        grid = macro.get_semantic_grid()
        json_str = grid.to_json()

        parsed = json.loads(json_str)
        assert "zones" in parsed

    def test_module_map_to_json(self, test_board):
        """Module map should serialize to JSON."""
        macro = MacroContext(test_board)
        modules = macro.get_module_map()
        json_str = modules.to_json()

        parsed = json.loads(json_str)
        assert "root" in parsed
        assert "flat_modules" in parsed


class TestVisionContext:
    """Test VisionContext context generator."""

    def test_render_region_returns_svg(self, test_board):
        """Rendering region should return SVG."""
        vision = VisionContext(test_board)
        image = vision.render_region(["U1", "R1"])

        assert image.svg_content is not None
        assert "<svg" in image.svg_content
        assert "</svg>" in image.svg_content

    def test_render_counts_components(self, test_board):
        """Render should count components correctly."""
        vision = VisionContext(test_board)
        image = vision.render_region(["U1", "R1"])

        assert image.component_count == 2

    def test_render_with_dimensions(self, test_board):
        """Rendering with dimensions should include annotations."""
        vision = VisionContext(test_board)
        image = vision.render_region(["U1", "R1"], show_dimensions=True)

        # Should have annotations
        assert isinstance(image.annotations, list)

    def test_render_full_board(self, test_board):
        """Rendering full board should include all components."""
        vision = VisionContext(test_board)
        image = vision.render_full_board()

        assert image.component_count == len(test_board.components)

    def test_svg_has_valid_structure(self, test_board):
        """SVG should have valid structure."""
        vision = VisionContext(test_board)
        image = vision.render_region(["U1"])

        # Check for common SVG elements
        assert "viewBox" in image.svg_content or "width" in image.svg_content

    def test_render_empty_refs(self, test_board):
        """Rendering empty refs should return empty image."""
        vision = VisionContext(test_board)
        image = vision.render_region([])

        assert image.component_count == 0


class TestContextIntegration:
    """Integration tests for context generators."""

    def test_micro_and_macro_consistency(self, test_board):
        """Micro and macro should report consistent component counts."""
        microscope = Microscope(test_board)
        macro = MacroContext(test_board)

        all_refs = list(test_board.components.keys())
        micro_data = microscope.inspect_region(all_refs)
        macro_summary = macro.get_summary()

        assert len(micro_data.objects) == macro_summary.component_count

    def test_empty_board_handling(self, empty_board):
        """Context generators should handle empty boards."""
        microscope = Microscope(empty_board)
        macro = MacroContext(empty_board)
        vision = VisionContext(empty_board)

        # Should not raise
        micro_data = microscope.inspect_region([])
        summary = macro.get_summary()
        image = vision.render_full_board()

        assert summary.component_count == 0
        assert image.component_count == 0
