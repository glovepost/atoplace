#!/usr/bin/env python3
"""Test Canvas export with a real board (dogtracker)."""

import sys
from pathlib import Path

# Use the real dogtracker board
board_path = Path("examples/dogtracker/layouts/default/default.kicad_pcb")

if not board_path.exists():
    print(f"Error: Board not found at {board_path}")
    sys.exit(1)

print(f"Loading real board: {board_path}")

# Load the board
from atoplace.board import Board
board = Board.from_kicad(str(board_path))

print(f"Board loaded: {len(board.components)} components, {len(board.nets)} nets")

# Run placement optimization with visualization
from atoplace.placement.force_directed import ForceDirectedRefiner, RefinementConfig
from atoplace.placement.visualizer import PlacementVisualizer
from atoplace.placement.module_detector import ModuleDetector

print("\nDetecting modules...")
detector = ModuleDetector(board)
detector.detect()

# Convert to dict format: ref -> module_type_name
modules = {ref: module.name for ref, module in detector._component_to_module.items()}
unique_modules = set(modules.values())
print(f"Detected {len(unique_modules)} module types: {sorted(unique_modules)}")

print("\nRunning placement optimization...")

# Create visualizer (delta compression is enabled by default)
viz = PlacementVisualizer(board)

# Create configuration for optimization
config = RefinementConfig(
    max_iterations=100,
    min_movement=0.05,  # Stop early if components settle
)

# Run optimization
refiner = ForceDirectedRefiner(
    board=board,
    config=config,
    visualizer=viz,
    modules=modules,  # Pass detected modules
)

# Set visualization interval to capture every frame
refiner._viz_interval = 1

result = refiner.refine()

print(f"\nOptimization complete:")
print(f"  Final energy: {result.total_energy:.2f}")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.iteration}")
print(f"  Frames captured: {len(viz.delta_frames)}")

# Export to Canvas HTML
print("\nExporting Canvas HTML...")
html_path = viz.export_canvas_html(filename="dogtracker_canvas.html")

print(f"\nCanvas visualization exported to: {html_path}")
print("Opening in browser...")

import subprocess
subprocess.run(["open", str(html_path)])
