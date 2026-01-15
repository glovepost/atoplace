#!/usr/bin/env python3
"""Generate Canvas HTML export for comparison."""

import random
import math
from pathlib import Path
from dataclasses import dataclass

# Mock components and board (same as streaming test)
@dataclass
class MockComponent:
    reference: str
    x: float
    y: float
    rotation: float
    width: float
    height: float
    layer: int = 0


class MockBoard:
    def __init__(self, num_components=30):
        self.components = {}
        self.outline = type('obj', (object,), {
            'min_x': 0.0, 'min_y': 0.0,
            'max_x': 120.0, 'max_y': 100.0
        })()

        component_types = ['C', 'R', 'U', 'L', 'D']
        for i in range(num_components):
            comp_type = random.choice(component_types)
            ref = f"{comp_type}{i+1}"
            self.components[ref] = MockComponent(
                reference=ref,
                x=random.uniform(10, 110),
                y=random.uniform(10, 90),
                rotation=random.uniform(0, 360),
                width=random.uniform(2, 10),
                height=random.uniform(2, 10),
            )


def main():
    # Create mock visualizer with captured frames
    from atoplace.placement.visualizer import PlacementVisualizer, ComponentStaticProps

    # Create board
    board = MockBoard(num_components=30)

    # Create visualizer
    class MockVisualizerForExport:
        def __init__(self, board):
            self.board = board
            self.frames = []
            self.delta_frames = []
            self.static_props = {}
            self.use_delta_compression = True

            # Initialize delta compressor
            from atoplace.placement.delta_compression import DeltaCompressor
            self.delta_compressor = DeltaCompressor()

            # Create color manager mock with module colors
            class MockColorManager:
                def __init__(self):
                    self.module_colors = {
                        'power_supply': '#e74c3c',  # Red
                        'analog': '#2ecc71',        # Green
                        'microcontroller': '#3498db',  # Blue
                        'digital': '#9b59b6',       # Purple
                    }

            self.color_manager = MockColorManager()

            # Set board bounds
            self.min_x = board.outline.min_x
            self.min_y = board.outline.min_y
            self.max_x = board.outline.max_x
            self.max_y = board.outline.max_y

            # Create static properties with pads
            for ref, comp in board.components.items():
                # Generate mock pads (2-4 pads per component)
                num_pads = 2 if 'R' in ref or 'C' in ref else 4
                pads = []
                pad_spacing = min(comp.width, comp.height) / 3
                for i in range(num_pads):
                    if num_pads == 2:
                        # Two pads on opposite sides
                        pad_x = -comp.width/4 if i == 0 else comp.width/4
                        pad_y = 0
                    else:
                        # Four pads in corners
                        pad_x = -comp.width/3 if i < 2 else comp.width/3
                        pad_y = -comp.height/3 if i % 2 == 0 else comp.height/3

                    pads.append([pad_x, pad_y, 0.5, 0.5, f"Net{i}"])

                self.static_props[ref] = ComponentStaticProps(
                    width=comp.width,
                    height=comp.height,
                    pads=pads
                )

    # Create visualizer
    viz = MockVisualizerForExport(board)

    # Simulate 100 frames of optimization
    print("Generating 100 frames of optimization data...")
    board_center_x = (board.outline.min_x + board.outline.max_x) / 2
    board_center_y = (board.outline.min_y + board.outline.max_y) / 2

    # Track previous positions for movement trails
    prev_positions = {ref: (comp.x, comp.y) for ref, comp in board.components.items()}

    # Generate some connections (ratsnest) - connect nearby components
    component_list = list(board.components.values())
    connections = []
    for i in range(0, len(component_list)-1, 2):
        comp1 = component_list[i]
        comp2 = component_list[i+1]
        connections.append([comp1.reference, comp2.reference, f"Net{i//2}"])

    for iteration in range(100):
        # Move components toward center
        for comp in board.components.values():
            dx = board_center_x - comp.x
            dy = board_center_y - comp.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.1:
                step_size = 0.3 * (1.0 - iteration / 100)
                comp.x += (dx/dist) * step_size + random.gauss(0, 0.08)
                comp.y += (dy/dist) * step_size + random.gauss(0, 0.08)
                comp.rotation += random.gauss(0, 0.5)

        # Module assignments
        modules = {}
        for comp in board.components.values():
            if 'C' in comp.reference:
                modules[comp.reference] = 'power_supply'
            elif 'R' in comp.reference:
                modules[comp.reference] = 'analog'
            elif 'U' in comp.reference:
                modules[comp.reference] = 'microcontroller'
            elif 'L' in comp.reference:
                modules[comp.reference] = 'power_supply'
            else:
                modules[comp.reference] = 'digital'

        # Forces (every 3 iterations)
        forces = {}
        if iteration % 3 == 0:
            for comp in board.components.values():
                dx = board_center_x - comp.x
                dy = board_center_y - comp.y
                magnitude = math.sqrt(dx*dx + dy*dy)
                if magnitude > 0.1:
                    forces[comp.reference] = [(dx/10, dy/10, "attraction")]

        energy = 1000.0 * (1.0 - iteration / 100)
        max_move = 2.0 * (1.0 - iteration / 100)

        # Get current component positions
        components_dict = {
            ref: (comp.x, comp.y, comp.rotation)
            for ref, comp in board.components.items()
        }

        # Calculate movement trails (delta from previous position)
        movement = {}
        for ref, comp in board.components.items():
            prev_x, prev_y = prev_positions[ref]
            dx = comp.x - prev_x
            dy = comp.y - prev_y
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only show if moved significantly
                movement[ref] = [dx, dy, 0]  # [dx, dy, drotation]

        # Update previous positions
        prev_positions = {ref: (comp.x, comp.y) for ref, comp in board.components.items()}

        # Calculate wire length for ratsnest
        wire_length = 0.0
        for ref1, ref2, net in connections:
            comp1 = board.components[ref1]
            comp2 = board.components[ref2]
            dx = comp2.x - comp1.x
            dy = comp2.y - comp1.y
            wire_length += math.sqrt(dx*dx + dy*dy)

        # Compress frame to delta
        delta = viz.delta_compressor.compress_frame(
            index=iteration,
            label=f"Iteration {iteration}",
            iteration=iteration,
            phase="convergence",
            components=components_dict,
            modules=modules,
            forces=forces,
            overlaps=[],
            movement=movement,
            connections=connections,
            energy=energy,
            max_move=max_move,
            overlap_count=0,
            total_wire_length=wire_length,
        )

        viz.delta_frames.append(delta)

    print(f"Generated {len(viz.delta_frames)} delta frames")

    # Export Canvas HTML
    print("Exporting Canvas HTML...")
    from atoplace.placement.visualizer import PlacementVisualizer
    html_path = PlacementVisualizer.export_canvas_html(viz, filename="canvas_demo.html")

    print(f"Canvas HTML exported to: {html_path}")
    print("\nOpening in browser...")


if __name__ == "__main__":
    main()