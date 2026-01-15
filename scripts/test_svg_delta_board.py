#!/usr/bin/env python3
"""Test SVG Delta export with a real board (dogtracker)."""

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
from atoplace.board.atopile_adapter import AtopileProjectLoader

project_root = AtopileProjectLoader.find_project_root(board_path)
board = Board.from_kicad(str(board_path))
ato_loader = None
if project_root:
    print(f"Detected atopile project: {project_root}")
    ato_loader = AtopileProjectLoader(project_root)
    ato_loader._apply_component_metadata(board)
    ato_loader._apply_module_hierarchy(board, build_name="default")

print(f"Board loaded: {len(board.components)} components, {len(board.nets)} nets")

# Run placement optimization with visualization
from atoplace.placement.force_directed import ForceDirectedRefiner, RefinementConfig
from atoplace.placement.legalizer import PlacementLegalizer, LegalizerConfig
from atoplace.placement.visualizer import PlacementVisualizer
from atoplace.placement.module_detector import ModuleDetector
from atoplace.dfm.profiles import get_profile_for_layers

print("\nDetecting modules...")
modules = {}
if ato_loader:
    for ref, comp in board.components.items():
        ato_module = comp.properties.get("ato_module")
        if ato_module:
            modules[ref] = ato_module

if modules:
    unique_modules = set(modules.values())
    print(f"Using atopile modules: {sorted(unique_modules)}")
else:
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
    max_iterations=400,
    min_movement=0.01,
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

# Run legalization after refinement
print("\nRunning legalization...")
dfm_profile = get_profile_for_layers(board.layer_count)
strict_clearance = max(0.35, dfm_profile.min_spacing)
legalize_config = LegalizerConfig(
    primary_grid=0.5,
    snap_rotation=True,
    align_passives_only=True,
    min_clearance=strict_clearance,
    edge_clearance=max(0.4, dfm_profile.min_trace_to_edge),
    row_spacing=strict_clearance * 2,
    guarantee_zero_overlaps=True,
    max_displacement_iterations=1000,
    overlap_retry_passes=50,
    escalation_factor=1.3,
    compact_outline=not board.outline.has_outline,
)
legalizer = PlacementLegalizer(board, legalize_config)
legal_result = legalizer.legalize()
print(
    f"  Overlaps resolved: {legal_result.overlaps_resolved} "
    f"(final={legal_result.final_overlaps})"
)

if viz:
    viz.capture_from_board(
        label="After Legalization",
        iteration=0,
        phase="legalization",
        modules=modules,
    )

# Export to SVG Delta HTML
print("\nExporting SVG Delta HTML...")
html_path = viz.export_svg_delta_html(filename="dogtracker_svg_delta.html")

print(f"\nSVG Delta visualization exported to: {html_path}")
print("Opening in browser...")

import subprocess
subprocess.run(["open", str(html_path)])
