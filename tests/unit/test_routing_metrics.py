"""Test routing metrics population in UnifiedVisualizer (Issue #34)."""

import pytest
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional


@dataclass
class MockRouteSegment:
    """Mock route segment for testing."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: int
    width: float
    net_id: Optional[int] = None
    net_name: Optional[str] = None


@dataclass
class MockVia:
    """Mock via for testing."""
    x: float
    y: float
    drill_diameter: float
    pad_diameter: float
    net_id: Optional[int] = None
    net_name: Optional[str] = None


@dataclass
class MockVisualizationFrame:
    """Mock visualization frame from RouteVisualizer."""
    completed_traces: List[MockRouteSegment] = field(default_factory=list)
    completed_vias: List[MockVia] = field(default_factory=list)
    explored_nodes: Set[Tuple[float, float, int]] = field(default_factory=set)
    frontier_nodes: Set[Tuple[float, float, int]] = field(default_factory=set)
    current_path: List[Tuple[float, float, int]] = field(default_factory=list)
    current_net_name: str = ""
    iteration: int = 0
    label: str = ""


def create_mock_board():
    """Create a minimal mock board for testing."""
    board = MagicMock()
    board.outline = None
    board.components = {}
    board.nets = {}
    return board


class TestRoutingMetricsPopulation:
    """Test that routing metrics are properly populated (Issue #34)."""

    def test_wire_length_calculated_from_traces(self):
        """Test that wire_length is calculated from trace segments."""
        from atoplace.visualization.unified import UnifiedVisualizer

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        # Create routing frames with traces
        frames = [
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),  # 10mm horizontal
                        layer=0,
                        width=0.2,
                        net_name="NET1"
                    ),
                    MockRouteSegment(
                        start=(10.0, 0.0),
                        end=(10.0, 5.0),  # 5mm vertical
                        layer=0,
                        width=0.2,
                        net_name="NET1"
                    ),
                ],
                current_net_name="NET1",
                label="Route NET1"
            )
        ]

        viz.add_routing_frames(frames)

        # Check that wire_length is calculated (10 + 5 = 15mm)
        assert len(viz.frames) == 1
        assert abs(viz.frames[0].wire_length - 15.0) < 0.001

    def test_nets_routed_counts_unique_nets(self):
        """Test that nets_routed counts unique nets across frames."""
        from atoplace.visualization.unified import UnifiedVisualizer

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        # Create routing frames with multiple nets
        frames = [
            # Frame 1: Route first net
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0), end=(5.0, 0.0),
                        layer=0, width=0.2, net_name="GND"
                    ),
                ],
                current_net_name="GND",
                label="Route GND"
            ),
            # Frame 2: Route second net (cumulative)
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0), end=(5.0, 0.0),
                        layer=0, width=0.2, net_name="GND"
                    ),
                    MockRouteSegment(
                        start=(0.0, 5.0), end=(5.0, 5.0),
                        layer=0, width=0.2, net_name="VCC"
                    ),
                ],
                current_net_name="VCC",
                label="Route VCC"
            ),
            # Frame 3: Route third net (cumulative)
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0), end=(5.0, 0.0),
                        layer=0, width=0.2, net_name="GND"
                    ),
                    MockRouteSegment(
                        start=(0.0, 5.0), end=(5.0, 5.0),
                        layer=0, width=0.2, net_name="VCC"
                    ),
                    MockRouteSegment(
                        start=(0.0, 10.0), end=(5.0, 10.0),
                        layer=0, width=0.2, net_name="DATA"
                    ),
                ],
                current_net_name="DATA",
                label="Route DATA"
            ),
        ]

        viz.add_routing_frames(frames)

        # Check cumulative nets_routed
        assert viz.frames[0].nets_routed == 1  # GND
        assert viz.frames[1].nets_routed == 2  # GND + VCC
        assert viz.frames[2].nets_routed == 3  # GND + VCC + DATA

    def test_nets_routed_from_vias(self):
        """Test that nets_routed also counts nets from vias."""
        from atoplace.visualization.unified import UnifiedVisualizer

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        # Create routing frame with only vias (no traces)
        frames = [
            MockVisualizationFrame(
                completed_vias=[
                    MockVia(x=5.0, y=5.0, drill_diameter=0.3, pad_diameter=0.6, net_name="NET1"),
                    MockVia(x=10.0, y=5.0, drill_diameter=0.3, pad_diameter=0.6, net_name="NET2"),
                ],
                current_net_name="NET2",
                label="Vias only"
            ),
        ]

        viz.add_routing_frames(frames)

        # Both nets should be counted from vias
        assert viz.frames[0].nets_routed == 2

    def test_nets_routed_in_delta_format(self):
        """Test that nets_routed is included in delta format output."""
        from atoplace.visualization.unified import UnifiedVisualizer

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        frames = [
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0), end=(5.0, 0.0),
                        layer=0, width=0.2, net_name="NET1"
                    ),
                ],
                current_net_name="NET1",
                label="Route NET1"
            ),
        ]

        viz.add_routing_frames(frames)

        # Convert to delta format
        delta_frames = viz._convert_frames_to_delta_format()

        # Check that nets_routed is in the delta
        assert len(delta_frames) == 1
        assert 'nets_routed' in delta_frames[0]
        assert delta_frames[0]['nets_routed'] == 1

    def test_wire_length_diagonal_trace(self):
        """Test wire_length calculation for diagonal traces."""
        from atoplace.visualization.unified import UnifiedVisualizer
        import math

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        # Create routing frame with diagonal trace (3-4-5 triangle)
        frames = [
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0),
                        end=(3.0, 4.0),  # sqrt(9+16) = 5mm
                        layer=0,
                        width=0.2,
                        net_name="DIAG"
                    ),
                ],
                current_net_name="DIAG",
                label="Diagonal"
            ),
        ]

        viz.add_routing_frames(frames)

        # Check diagonal length
        assert abs(viz.frames[0].wire_length - 5.0) < 0.001

    def test_empty_routing_frames(self):
        """Test that empty routing frames have zero metrics."""
        from atoplace.visualization.unified import UnifiedVisualizer

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        # Create empty routing frame
        frames = [
            MockVisualizationFrame(
                completed_traces=[],
                completed_vias=[],
                current_net_name="",
                label="Empty"
            ),
        ]

        viz.add_routing_frames(frames)

        assert viz.frames[0].wire_length == 0.0
        assert viz.frames[0].nets_routed == 0


class TestDeltaFormatComplete:
    """Test that all routing fields are in delta format."""

    def test_all_routing_fields_present(self):
        """Test that all expected routing fields are in delta format."""
        from atoplace.visualization.unified import UnifiedVisualizer

        board = create_mock_board()
        viz = UnifiedVisualizer(board)

        frames = [
            MockVisualizationFrame(
                completed_traces=[
                    MockRouteSegment(
                        start=(0.0, 0.0), end=(5.0, 0.0),
                        layer=0, width=0.2, net_name="NET1"
                    ),
                ],
                completed_vias=[
                    MockVia(x=5.0, y=5.0, drill_diameter=0.3, pad_diameter=0.6, net_name="NET1"),
                ],
                explored_nodes={(1.0, 1.0, 0), (2.0, 2.0, 0)},
                frontier_nodes={(3.0, 3.0, 0)},
                current_path=[(0.0, 0.0, 0), (5.0, 0.0, 0)],
                current_net_name="NET1",
                iteration=42,
                label="Test frame"
            ),
        ]

        viz.add_routing_frames(frames)
        delta_frames = viz._convert_frames_to_delta_format()

        delta = delta_frames[0]

        # Check all expected fields are present
        assert 'traces' in delta
        assert 'vias' in delta
        assert 'astar_explored' in delta
        assert 'astar_frontier' in delta
        assert 'astar_path' in delta
        assert 'current_net' in delta
        assert 'wire_length' in delta
        assert 'nets_routed' in delta

        # Check values are correct
        assert len(delta['traces']) == 1
        assert len(delta['vias']) == 1
        assert delta['nets_routed'] == 1
        assert delta['current_net'] == "NET1"
