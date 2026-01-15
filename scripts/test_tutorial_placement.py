#!/usr/bin/env python3
"""Test atoplace placement on the tutorial board"""
from pathlib import Path
from atoplace.board.kicad_adapter import load_kicad_board, save_kicad_board
from atoplace.placement.force_directed import ForceDirectedRefiner
from atoplace.placement.constraints import ProximityConstraint
from atoplace.validation.confidence import ConfidenceScorer

# Load the tutorial PCB
pcb_path = Path("examples/tutorial/elec/layout/default/tutorial.kicad_pcb")

print(f"Loading PCB: {pcb_path}")
try:
    board = load_kicad_board(pcb_path)
    print(f"✓ Board loaded successfully")
    print(f"  - Components: {len(board.components)}")
    print(f"  - Nets: {len(board.nets)}")
    print(f"  - Board outline: {board.outline}")

    if board.components:
        print("\nComponents:")
        for comp in board.components[:10]:  # Show first 10
            print(f"  {comp.reference}: {comp.footprint} at ({comp.x:.2f}, {comp.y:.2f})")

        # Try running placement optimization
        print("\nRunning force-directed placement...")
        refiner = ForceDirectedRefiner()

        # Run a quick optimization
        result = refiner.refine(board, iterations=50)
        print(f"✓ Placement completed")
        print(f"  - Iterations: {result.get('iterations', 50)}")
        print(f"  - Final energy: {result.get('energy', 'N/A')}")

        # Assess quality
        print("\nAssessing placement quality...")
        scorer = ConfidenceScorer()
        report = scorer.assess(board)
        print(f"✓ Confidence score: {report.overall_score:.2f}")
        print(f"  - Component spacing: {report.component_spacing_score:.2f}")
        print(f"  - Net connectivity: {report.net_connectivity_score:.2f}")

        # Save result
        output_path = Path("examples/tutorial/test_placed.kicad_pcb")
        save_kicad_board(board, output_path)
        print(f"\n✓ Saved placed board to: {output_path}")
    else:
        print("\n⚠ No components found in the board")
        print("This board appears to be empty - components need to be imported first")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
