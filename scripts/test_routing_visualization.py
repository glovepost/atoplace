#!/usr/bin/env python3
"""Test routing visualization support in the SVG delta viewer.

Tests the routing visualization features added to the unified visualization system:
1. RouteVisualizer.export_svg_delta_html() method
2. Trace rendering with layer visibility
3. Via rendering with pad and drill
4. A* debug visualization (explored, frontier, current path)
"""

import tempfile
import json
from pathlib import Path

from atoplace.routing.visualizer import (
    RouteVisualizer,
    RouteSegment,
    Via,
    VisualizationFrame,
)
from atoplace.routing.spatial_index import Obstacle


def test_routing_visualizer_export_svg_delta():
    """Test that RouteVisualizer can export SVG delta HTML with routing data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create visualizer with board bounds (100mm x 80mm board)
        visualizer = RouteVisualizer(
            board_bounds=(0, 0, 100, 80),
            output_dir=Path(tmpdir),
            scale=10.0,
            margin=5.0
        )

        # Create some test traces
        traces = [
            RouteSegment(start=(10, 10), end=(50, 10), layer=0, width=0.25, net_id=1),  # F.Cu
            RouteSegment(start=(50, 10), end=(50, 50), layer=0, width=0.25, net_id=1),  # F.Cu
            RouteSegment(start=(20, 20), end=(60, 20), layer=1, width=0.25, net_id=2),  # B.Cu
        ]

        # Create some test vias
        vias = [
            Via(x=50, y=30, drill_diameter=0.3, pad_diameter=0.6, net_id=1),
            Via(x=70, y=40, drill_diameter=0.3, pad_diameter=0.6, net_id=2),
        ]

        # Create some A* debug data
        explored_nodes = {(10, 10, 0), (15, 10, 0), (20, 10, 0), (25, 10, 0)}
        frontier_nodes = {(30, 10, 0), (35, 10, 0)}
        current_path = [(10, 10, 0), (15, 10, 0), (20, 10, 0), (25, 10, 0)]

        # Create test obstacles/pads
        pads = [
            Obstacle(min_x=8, min_y=8, max_x=12, max_y=12, layer=0, net_id=1, obstacle_type="pad"),
            Obstacle(min_x=48, min_y=48, max_x=52, max_y=52, layer=0, net_id=1, obstacle_type="pad"),
        ]
        obstacles = [
            Obstacle(min_x=30, min_y=30, max_x=40, max_y=40, layer=0, net_id=None, obstacle_type="keepout"),
        ]

        # Capture frame 1: initial state
        visualizer.capture_frame(
            obstacles=obstacles,
            pads=pads,
            completed_traces=[],
            completed_vias=[],
            current_net="VCC",
            label="Starting routing"
        )

        # Capture frame 2: A* exploration
        visualizer.capture_frame(
            obstacles=obstacles,
            pads=pads,
            completed_traces=[],
            completed_vias=[],
            explored_nodes=explored_nodes,
            frontier_nodes=frontier_nodes,
            current_path=current_path,
            current_net="VCC",
            label="A* exploration"
        )

        # Capture frame 3: completed routing
        visualizer.capture_frame(
            obstacles=obstacles,
            pads=pads,
            completed_traces=traces,
            completed_vias=vias,
            current_net="VCC",
            label="Routing complete"
        )

        # Export to SVG delta HTML
        output_path = visualizer.export_svg_delta_html("routing_test.html")

        assert output_path is not None, "Export should return a path"
        assert output_path.exists(), "Output file should exist"

        # Read and validate HTML content
        html_content = output_path.read_text()

        # Check for essential structural elements
        assert '<svg' in html_content, "Should contain SVG element"
        assert 'traces-group' in html_content, "Should contain traces group"
        assert 'vias-group' in html_content, "Should contain vias group"
        assert 'astar-debug-group' in html_content, "Should contain A* debug group"

        # Check for delta frames data
        assert 'deltaFrames' in html_content, "Should contain deltaFrames JavaScript variable"
        assert 'boardBounds' in html_content, "Should contain boardBounds"

        # Check for copper layer controls (traces/vias now follow copper layers)
        assert 'show-top' in html_content or 'Top (F.Cu)' in html_content, "Should have F.Cu layer control"
        assert 'show-bottom' in html_content or 'Bottom (B.Cu)' in html_content, "Should have B.Cu layer control"
        assert 'show-astar-debug' in html_content, "Should have A* debug control"

        # Check JavaScript rendering functions
        assert 'updateTraces' in html_content, "Should have updateTraces function"
        assert 'updateVias' in html_content, "Should have updateVias function"

        print(f"✓ HTML export created at: {output_path}")
        print(f"  File size: {output_path.stat().st_size:,} bytes")
        print(f"  Frames: {len(visualizer.frames)}")

        # Validate delta frame structure
        # Extract deltaFrames JSON from HTML
        start_marker = "const deltaFrames = "
        end_marker = ";\n\n// Total frames"
        start_idx = html_content.find(start_marker)
        end_idx = html_content.find(end_marker, start_idx)

        if start_idx != -1 and end_idx != -1:
            json_str = html_content[start_idx + len(start_marker):end_idx]
            delta_frames = json.loads(json_str)

            print(f"\n  Delta frames structure:")
            for i, frame in enumerate(delta_frames):
                print(f"    Frame {i}: {frame.get('label', 'N/A')}")
                print(f"      - traces: {len(frame.get('traces', []))}")
                print(f"      - vias: {len(frame.get('vias', []))}")
                print(f"      - A* explored: {len(frame.get('astar_explored', []))}")
                print(f"      - A* frontier: {len(frame.get('astar_frontier', []))}")
                print(f"      - A* path: {len(frame.get('astar_path', []))}")

            # Validate trace data structure
            frame3 = delta_frames[2]  # Completed routing frame
            assert len(frame3['traces']) == 3, f"Expected 3 traces, got {len(frame3['traces'])}"
            assert len(frame3['vias']) == 2, f"Expected 2 vias, got {len(frame3['vias'])}"

            # Check trace format
            trace = frame3['traces'][0]
            assert 'start' in trace, "Trace should have start"
            assert 'end' in trace, "Trace should have end"
            assert 'layer' in trace, "Trace should have layer"
            assert 'width' in trace, "Trace should have width"

            # Check via format
            via = frame3['vias'][0]
            assert 'x' in via, "Via should have x"
            assert 'y' in via, "Via should have y"
            assert 'drill' in via, "Via should have drill"
            assert 'pad' in via, "Via should have pad"

            # Validate A* debug data in frame 2
            frame2 = delta_frames[1]
            assert len(frame2['astar_explored']) > 0, "Frame 2 should have A* explored nodes"
            assert len(frame2['astar_frontier']) > 0, "Frame 2 should have A* frontier nodes"
            assert len(frame2['astar_path']) > 0, "Frame 2 should have A* path"

            print("\n✓ Delta frame data structure is valid")

        return output_path


def test_routing_visualizer_empty_frames():
    """Test that export handles empty frames gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer = RouteVisualizer(
            board_bounds=(0, 0, 50, 50),
            output_dir=Path(tmpdir)
        )

        # No frames captured
        output_path = visualizer.export_svg_delta_html()
        assert output_path is None, "Should return None for empty frames"
        print("✓ Empty frames handled gracefully")


def test_legacy_export_deprecation():
    """Test that legacy export_html_report shows deprecation warning."""
    import warnings

    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer = RouteVisualizer(
            board_bounds=(0, 0, 50, 50),
            output_dir=Path(tmpdir)
        )

        # Add a minimal frame
        visualizer.capture_frame(
            obstacles=[],
            pads=[],
            label="Test"
        )

        # Should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            visualizer.export_html_report()

            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert issubclass(w[0].category, DeprecationWarning)
            assert "export_svg_delta_html" in str(w[0].message)
            print("✓ Deprecation warning emitted for legacy export_html_report()")


def test_coordinate_transforms():
    """Test that board coordinates transform correctly to SVG coordinates."""
    visualizer = RouteVisualizer(
        board_bounds=(10, 20, 110, 120),  # 100x100 board at offset
        scale=10.0,
        margin=5.0
    )

    # Test coordinate transform
    # Point (10, 20) should map to (margin * scale, margin * scale)
    svg_x, svg_y = visualizer._to_svg_coords(10, 20)
    expected_x = 5.0 * 10.0  # margin * scale
    expected_y = 5.0 * 10.0  # margin * scale
    assert svg_x == expected_x, f"X transform failed: {svg_x} != {expected_x}"
    assert svg_y == expected_y, f"Y transform failed: {svg_y} != {expected_y}"

    # Point (60, 70) should map to center of SVG
    svg_x, svg_y = visualizer._to_svg_coords(60, 70)
    expected_x = (60 - 10 + 5) * 10.0  # 550
    expected_y = (70 - 20 + 5) * 10.0  # 550
    assert svg_x == expected_x, f"X transform failed: {svg_x} != {expected_x}"
    assert svg_y == expected_y, f"Y transform failed: {svg_y} != {expected_y}"

    print("✓ Coordinate transforms work correctly")


def test_frame_capture():
    """Test frame capture with all data types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer = RouteVisualizer(
            board_bounds=(0, 0, 100, 100),
            output_dir=Path(tmpdir)
        )

        # Capture frame with all data types
        frame = visualizer.capture_frame(
            obstacles=[Obstacle(0, 0, 10, 10, 0, obstacle_type="keepout")],
            pads=[Obstacle(20, 20, 25, 25, 0, net_id=1, obstacle_type="pad")],
            completed_traces=[RouteSegment((0, 0), (10, 10), 0, 0.25)],
            completed_vias=[Via(5, 5, 0.3, 0.6)],
            explored_nodes={(1, 1, 0), (2, 2, 0)},
            frontier_nodes={(3, 3, 0)},
            current_path=[(0, 0, 0), (1, 1, 0), (2, 2, 0)],
            current_net="TEST_NET",
            label="Test Frame"
        )

        assert frame.iteration == 0
        assert len(frame.obstacles) == 1
        assert len(frame.pads) == 1
        assert len(frame.completed_traces) == 1
        assert len(frame.completed_vias) == 1
        assert len(frame.explored_nodes) == 2
        assert len(frame.frontier_nodes) == 1
        assert len(frame.current_path) == 3
        assert frame.current_net_name == "TEST_NET"
        assert frame.label == "Test Frame"

        assert len(visualizer.frames) == 1
        print("✓ Frame capture works with all data types")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Routing Visualization Support")
    print("=" * 60)
    print()

    test_coordinate_transforms()
    print()

    test_frame_capture()
    print()

    test_routing_visualizer_empty_frames()
    print()

    test_legacy_export_deprecation()
    print()

    output_path = test_routing_visualizer_export_svg_delta()
    print()

    print("=" * 60)
    print("All routing visualization tests passed!")
    print("=" * 60)

    if output_path:
        print(f"\nTo view the test output, open:\n  file://{output_path}")
