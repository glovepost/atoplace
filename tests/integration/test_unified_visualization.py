#!/usr/bin/env python3
"""Test script for unified placement + routing visualization.

This script demonstrates the UnifiedVisualizer by:
1. Loading a KiCad board
2. Running placement refinement and capturing frames
3. Running the real routing algorithm and capturing frames
4. Combining into a single unified HTML visualization
"""

import sys
from pathlib import Path

# Use KiCad's Python for pcbnew support
KICAD_PYTHON = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"


def main():
    from atoplace.board.abstraction import Board
    from atoplace.placement.force_directed import ForceDirectedRefiner
    from atoplace.placement.visualizer import PlacementVisualizer
    from atoplace.placement.module_detector import ModuleDetector
    from atoplace.routing.visualizer import RouteVisualizer, RouteSegment, Via, create_visualizer_from_board
    from atoplace.routing.manager import RoutingManager, RoutingManagerConfig
    from atoplace.dfm.profiles import get_profile
    from atoplace.visualization import UnifiedVisualizer, create_unified_visualizer

    # Find a test board
    board_path = Path("examples/dogtracker/layouts/default/default.kicad_pcb")
    if not board_path.exists():
        # Try alternative path
        board_path = Path("examples/hyperion/layouts/demomatrix/demomatrix.kicad_pcb")
        if not board_path.exists():
            print("No test board found. Please provide a .kicad_pcb file.")
            return 1

    print(f"Loading board: {board_path}")
    board = Board.from_kicad(str(board_path))
    print(f"  Components: {len(board.components)}")
    print(f"  Nets: {len(board.nets)}")

    # Detect modules
    print("\n=== Detecting Modules ===")
    detector = ModuleDetector(board)
    modules = detector.detect()

    # Build module map (ref -> module_name)
    module_map = {}
    for module in modules:
        for ref in module.components:
            module_map[ref] = module.module_type.value

    # Check for atopile project and use atopile module names if available
    # First check for explicit ato_module, then fall back to atopile_address
    is_atopile = any(
        comp.properties.get("ato_module") or comp.properties.get("atopile_address")
        for comp in board.components.values()
    )
    if is_atopile:
        print("  Detected atopile project - using atopile module names")

        # Get project name from board path (e.g., "dogtracker" -> "Dogtracker")
        project_name = board_path.parent.parent.parent.name.capitalize()

        for ref, comp in board.components.items():
            # Check for explicit ato_module first
            ato_module = comp.properties.get("ato_module")
            if ato_module:
                module_map[ref] = ato_module
            else:
                # Extract from atopile_address (e.g., 'power.c_vcc_1' -> 'Dogtracker.power')
                addr = comp.properties.get("atopile_address", "")
                if addr:
                    parts = addr.rsplit(".", 1)
                    if len(parts) > 1:
                        module_name = f"{project_name}.{parts[0]}"
                    else:
                        module_name = project_name
                    module_map[ref] = module_name

    # Print module summary
    module_counts = {}
    for ref, module_name in module_map.items():
        module_counts[module_name] = module_counts.get(module_name, 0) + 1
    for module_name, count in sorted(module_counts.items()):
        print(f"  {module_name}: {count} components")

    # Create placement visualizer and run refinement
    print("\n=== Running Placement Phase ===")
    placement_viz = PlacementVisualizer(board)

    # Create refiner with visualizer and modules attached - frames captured automatically
    from atoplace.placement.force_directed import RefinementConfig
    config = RefinementConfig(max_iterations=50)  # Limit iterations for quick test
    refiner = ForceDirectedRefiner(
        board,
        config=config,
        visualizer=placement_viz,
        modules=module_map  # Pass module assignments for visualization
    )

    # Run refinement (visualizer captures frames automatically)
    final_state = refiner.refine()
    print(f"  Final energy: {final_state.total_energy:.2f}")
    print(f"  Total iterations: {final_state.iteration}")
    print(f"  Converged: {final_state.converged}")
    print(f"  Captured {len(placement_viz.frames)} placement frames after refinement")

    # Run legalization phase
    print("\n=== Running Legalization Phase ===")
    from atoplace.placement.legalizer import PlacementLegalizer, LegalizerConfig

    legalize_config = LegalizerConfig(
        primary_grid=0.5,
        snap_rotation=True,
        align_passives_only=True,
        min_clearance=0.15,
        edge_clearance=0.4,
        row_spacing=0.3,
        guarantee_zero_overlaps=True,
    )
    legalizer = PlacementLegalizer(board, legalize_config)
    legal_result = legalizer.legalize()
    print(f"  Grid snapped: {legal_result.grid_snapped}")
    print(f"  Rows formed: {legal_result.rows_formed}")
    print(f"  Overlaps resolved: {legal_result.overlaps_resolved}")
    print(f"  Final overlaps: {legal_result.final_overlaps}")

    # Capture legalization result frame
    placement_viz.capture_from_board(
        label="After Legalization",
        iteration=0,
        phase="legalization",
        modules=module_map,
    )
    print(f"  Captured {len(placement_viz.frames)} total placement frames")

    # Run real routing algorithm
    print("\n=== Running Routing Phase ===")

    # Configure routing manager with visualization enabled
    routing_config = RoutingManagerConfig(
        visualize=True,  # Enable frame capture
        enable_fanout=False,  # Skip fanout for simpler boards
        enable_pin_optimization=False,
    )

    # Get DFM profile for design rules
    dfm_profile = get_profile("jlcpcb_standard")

    # Create routing manager (it creates its own visualizer internally)
    routing_manager = RoutingManager(
        board=board,
        dfm_profile=dfm_profile,
        config=routing_config,
    )

    # Run routing
    print("  Starting routing...")
    routing_result = routing_manager.route_all()

    print(f"  Routed {routing_result.routed_nets}/{routing_result.total_nets} nets")
    print(f"  Completion rate: {routing_result.completion_rate:.1f}%")
    print(f"  Total trace length: {routing_result.total_length:.1f}mm")
    print(f"  Total vias: {routing_result.total_vias}")

    # Get the visualizer's frames
    route_viz = routing_manager._visualizer
    if route_viz:
        print(f"  Captured {len(route_viz.frames)} routing frames")
    else:
        print("  No routing visualizer created")
        route_viz = create_visualizer_from_board(board)  # Fallback empty visualizer

    # Create unified visualization
    print("\n=== Creating Unified Visualization ===")
    unified_viz = create_unified_visualizer(board)

    # Add placement frames
    unified_viz.add_placement_frames(placement_viz.frames)

    # Add transition marker
    unified_viz.add_transition_frame()

    # Add routing frames
    unified_viz.add_routing_frames(route_viz.frames)

    print(f"  Total unified frames: {len(unified_viz.frames)}")

    # Export unified HTML
    output_path = unified_viz.export_html(
        filename="unified_test.html",
        output_dir="placement_debug"
    )

    if output_path:
        print(f"\n=== SUCCESS ===")
        print(f"Unified visualization exported to: {output_path}")
        print(f"Open in browser: file://{output_path.absolute()}")
    else:
        print("\n=== FAILED ===")
        print("Failed to export visualization")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
