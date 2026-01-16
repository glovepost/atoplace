#!/usr/bin/env python3
"""Test routing visualization with real dogtracker board.

Loads the dogtracker default.kicad_pcb and generates routing visualization
to verify the SVG delta viewer works with real PCB data.
"""

from pathlib import Path
import random

from atoplace.board.abstraction import Board
from atoplace.routing.visualizer import (
    RouteVisualizer,
    RouteSegment,
    Via,
    create_visualizer_from_board,
)
from atoplace.routing.spatial_index import Obstacle


def test_routing_visualization_with_dogtracker():
    """Test routing visualization with the dogtracker board.

    Uses create_visualizer_from_board() which now includes the board
    reference, enabling component rendering in the visualization.
    """

    # Load the dogtracker board
    board_path = Path("examples/dogtracker/layouts/default/default.kicad_pcb")
    print(f"Loading board: {board_path}")

    board = Board.from_kicad(str(board_path))
    print(f"  Components: {len(board.components)}")
    print(f"  Nets: {len(board.nets)}")

    # Create visualizer from board (now includes board reference for component rendering)
    visualizer = create_visualizer_from_board(board)
    visualizer.output_dir = Path("placement_debug")
    visualizer.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Board bounds: {visualizer.bounds}")
    print(f"  SVG size: {visualizer.svg_width:.0f}x{visualizer.svg_height:.0f}")
    print(f"  Board reference: {'Yes' if visualizer.board else 'No'}")

    # Extract pads and obstacles from board components
    pads = []
    obstacles = []

    # Build net name to ID mapping
    net_name_to_id = {}
    for i, net_name in enumerate(board.nets.keys()):
        net_name_to_id[net_name] = i + 1

    # Build pads by net for easier lookup
    pads_by_net = {}

    for ref, comp in board.components.items():
        bbox = comp.get_bounding_box()
        # Component body as obstacle
        obstacles.append(Obstacle(
            min_x=bbox[0],
            min_y=bbox[1],
            max_x=bbox[2],
            max_y=bbox[3],
            layer=0 if comp.layer == "F.Cu" else 1,
            obstacle_type="component",
            ref=ref
        ))

        # Component pads
        for pad in comp.pads:
            pad_bbox = pad.get_bounding_box(comp.x, comp.y, comp.rotation)
            net_id = None
            net_name = pad.net
            if net_name and net_name in net_name_to_id:
                net_id = net_name_to_id[net_name]

                # Track pads by net
                if net_name not in pads_by_net:
                    pads_by_net[net_name] = []

            pad_obs = Obstacle(
                min_x=pad_bbox[0],
                min_y=pad_bbox[1],
                max_x=pad_bbox[2],
                max_y=pad_bbox[3],
                layer=0 if comp.layer == "F.Cu" else 1,
                net_id=net_id,
                obstacle_type="pad",
                ref=f"{ref}.{pad.number}"
            )
            pads.append(pad_obs)

            if net_name and net_name in pads_by_net:
                pads_by_net[net_name].append(pad_obs)

    print(f"  Extracted {len(obstacles)} component obstacles")
    print(f"  Extracted {len(pads)} pads")

    # Generate simulated routing data
    # Pick nets with 2+ pads to route
    routable_nets = [(name, pads_list) for name, pads_list in pads_by_net.items() if len(pads_list) >= 2]
    routable_nets = routable_nets[:8]  # Limit to 8 nets

    print(f"  Routable nets (2+ pads): {len(routable_nets)}")

    traces = []
    vias = []

    # Frame 1: Initial state
    visualizer.capture_frame(
        obstacles=obstacles,
        pads=pads,
        label="Initial board state",
        current_net=""
    )

    # Generate simulated routing for each net
    for i, (net_name, net_pads) in enumerate(routable_nets):
        if len(net_pads) < 2:
            continue

        # Simulate A* exploration
        start_pad = net_pads[0]
        end_pad = net_pads[-1]

        # Generate explored nodes between pads
        start_x, start_y = start_pad.center
        end_x, end_y = end_pad.center

        explored = set()
        frontier = set()
        path = []

        # Create exploration pattern
        steps = 20
        for j in range(steps):
            t = j / steps
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            # Add some randomness
            x += random.uniform(-2, 2)
            y += random.uniform(-2, 2)
            explored.add((x, y, 0))
            if j < 5:
                path.append((x, y, 0))

        # Frontier at the edge
        for j in range(5):
            t = (steps + j) / (steps + 5)
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            frontier.add((x, y, 0))

        # Frame: A* exploration for this net
        visualizer.capture_frame(
            obstacles=obstacles,
            pads=pads,
            completed_traces=traces.copy(),
            completed_vias=vias.copy(),
            explored_nodes=explored,
            frontier_nodes=frontier,
            current_path=path,
            label=f"Routing {net_name}",
            current_net=net_name
        )

        # Create trace segments for this net
        # Simple L-route between pads
        mid_x = (start_x + end_x) / 2
        net_id = net_name_to_id.get(net_name, i)

        new_traces = [
            RouteSegment(
                start=(start_x, start_y),
                end=(mid_x, start_y),
                layer=0,
                width=0.25,
                net_id=net_id
            ),
            RouteSegment(
                start=(mid_x, start_y),
                end=(mid_x, end_y),
                layer=0,
                width=0.25,
                net_id=net_id
            ),
            RouteSegment(
                start=(mid_x, end_y),
                end=(end_x, end_y),
                layer=0,
                width=0.25,
                net_id=net_id
            ),
        ]
        traces.extend(new_traces)

        # Add a via for every other net
        if i % 2 == 1:
            vias.append(Via(
                x=mid_x,
                y=(start_y + end_y) / 2,
                drill_diameter=0.3,
                pad_diameter=0.6,
                net_id=net_id
            ))

        # Frame: Net completed
        visualizer.capture_frame(
            obstacles=obstacles,
            pads=pads,
            completed_traces=traces.copy(),
            completed_vias=vias.copy(),
            label=f"{net_name} complete",
            current_net=net_name
        )

    # Final frame
    visualizer.capture_frame(
        obstacles=obstacles,
        pads=pads,
        completed_traces=traces,
        completed_vias=vias,
        label="All routing complete"
    )

    print(f"\nCaptured {len(visualizer.frames)} frames")
    print(f"  Total traces: {len(traces)}")
    print(f"  Total vias: {len(vias)}")

    # Export to SVG delta HTML
    output_path = visualizer.export_svg_delta_html("dogtracker_routing_test.html")

    print(f"\n✓ Created routing visualization: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")

    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Routing Visualization with DogTracker Board")
    print("=" * 60)
    print()

    output_path = test_routing_visualization_with_dogtracker()

    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
    print(f"\nOpen the visualization:")
    print(f"  open {output_path}")
