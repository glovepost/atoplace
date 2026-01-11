#!/usr/bin/env python3
"""
AtoPlace CLI

Command-line interface for the AI-powered PCB placement and routing tool.

Usage:
    atoplace place <board.kicad_pcb> [options]
    atoplace validate <board.kicad_pcb> [options]
    atoplace report <board.kicad_pcb>
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def check_pcbnew():
    """Check if pcbnew is available."""
    try:
        import pcbnew
        return True
    except ImportError:
        return False


def cmd_place(args):
    """Run placement optimization."""
    from .board.abstraction import Board
    from .board.kicad_adapter import load_kicad_board, save_kicad_board
    from .placement.force_directed import ForceDirectedRefiner, RefinementConfig
    from .placement.module_detector import ModuleDetector
    from .nlp.constraint_parser import ConstraintParser
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile

    pcb_path = Path(args.board)
    if not pcb_path.exists():
        print(f"Error: Board file not found: {pcb_path}")
        return 1

    print(f"Loading board: {pcb_path}")
    board = load_kicad_board(pcb_path)
    print(f"  Components: {len(board.components)}")
    print(f"  Nets: {len(board.nets)}")

    # Detect modules
    print("\nDetecting functional modules...")
    detector = ModuleDetector(board)
    modules = detector.detect()

    for module in modules:
        if module.components:
            print(f"  {module.module_type.value}: {len(module.components)} components")

    # Parse constraints if provided
    constraints = []
    if args.constraints:
        print(f"\nParsing constraints: {args.constraints}")
        parser = ConstraintParser(board)
        parsed, summary = parser.parse_interactive(args.constraints)
        constraints = parsed
        print(summary)

    # Configure refinement
    config = RefinementConfig(
        max_iterations=args.iterations or 500,
        min_movement=0.01,
        damping=0.85,
    )

    if args.grid:
        config.snap_to_grid = True
        config.grid_size = args.grid

    # Run force-directed refinement
    print("\nRunning force-directed placement refinement...")
    refiner = ForceDirectedRefiner(board, config)

    for constraint in constraints:
        refiner.add_constraint(constraint)

    def progress_callback(state):
        if state.iteration % 50 == 0:
            print(f"  Iteration {state.iteration}: energy={state.total_energy:.2f}")

    result = refiner.refine(callback=progress_callback if args.verbose else None)

    if result.converged:
        print(f"  Converged after {result.iteration} iterations")
    else:
        print(f"  Stopped after {result.iteration} iterations (max reached)")

    # Validate result
    print("\nValidating placement...")
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")
    scorer = ConfidenceScorer(dfm_profile)
    report = scorer.assess(board)

    print(report.summary())

    # Save result
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pcb_path.with_suffix('.placed.kicad_pcb')

    if not args.dry_run:
        print(f"\nSaving to: {output_path}")
        save_kicad_board(board, output_path)
        print("Done!")
    else:
        print("\nDry run - not saving changes")

    return 0 if report.overall_score >= 0.7 else 1


def cmd_validate(args):
    """Validate board placement."""
    from .board.kicad_adapter import load_kicad_board
    from .validation.confidence import ConfidenceScorer
    from .validation.pre_route import PreRouteValidator
    from .validation.drc import DRCChecker
    from .dfm.profiles import get_profile

    pcb_path = Path(args.board)
    if not pcb_path.exists():
        print(f"Error: Board file not found: {pcb_path}")
        return 1

    print(f"Loading board: {pcb_path}")
    board = load_kicad_board(pcb_path)

    # Run pre-route validation
    print("\nRunning pre-route validation...")
    pre_validator = PreRouteValidator(board)
    can_proceed, issues = pre_validator.validate()
    print(pre_validator.get_summary())

    # Run DRC
    print("\nRunning DRC checks...")
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")
    drc = DRCChecker(board, dfm_profile)
    passed, violations = drc.run_checks()
    print(drc.get_summary())

    # Run confidence scoring
    print("\nGenerating confidence report...")
    scorer = ConfidenceScorer(dfm_profile)
    report = scorer.assess(board)

    if args.output:
        # Save markdown report
        output_path = Path(args.output)
        output_path.write_text(report.to_markdown())
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + "=" * 60)
        print(report.summary())

    return 0 if passed and can_proceed else 1


def cmd_report(args):
    """Generate detailed report for a board."""
    from .board.kicad_adapter import load_kicad_board
    from .placement.module_detector import ModuleDetector
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile

    pcb_path = Path(args.board)
    if not pcb_path.exists():
        print(f"Error: Board file not found: {pcb_path}")
        return 1

    board = load_kicad_board(pcb_path)

    # Detect modules
    detector = ModuleDetector(board)
    modules = detector.detect()

    # Generate report
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")
    scorer = ConfidenceScorer(dfm_profile)
    report = scorer.assess(board)

    # Print report
    print(report.to_markdown())

    return 0


def cmd_interactive(args):
    """Run interactive constraint session."""
    from .board.kicad_adapter import load_kicad_board, save_kicad_board
    from .nlp.constraint_parser import ConstraintParser, ModificationHandler
    from .placement.force_directed import ForceDirectedRefiner
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile

    pcb_path = Path(args.board)
    if not pcb_path.exists():
        print(f"Error: Board file not found: {pcb_path}")
        return 1

    board = load_kicad_board(pcb_path)
    parser = ConstraintParser(board)
    modifier = ModificationHandler(board)
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")

    print(f"Loaded: {pcb_path}")
    print(f"Components: {len(board.components)}")
    print(f"Enter constraints or modifications. Type 'help' for commands, 'quit' to exit.")
    print()

    constraints = []

    while True:
        try:
            user_input = input("atoplace> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        if user_input.lower() == 'help':
            print("""
Commands:
  help          Show this help
  quit          Exit interactive mode
  list          List current constraints
  clear         Clear all constraints
  apply         Apply constraints and refine placement
  save [path]   Save board
  report        Show confidence report

Constraint examples:
  "Keep C1 close to U1"
  "USB connector on left edge"
  "Group all capacitors together"
  "Separate analog and digital sections"

Modification examples:
  "Move U1 left"
  "Rotate J1 90 degrees"
  "Swap C1 and C2"
            """)
            continue

        if user_input.lower() == 'list':
            if constraints:
                print("Current constraints:")
                for c in constraints:
                    print(f"  - {c.description}")
            else:
                print("No constraints defined.")
            continue

        if user_input.lower() == 'clear':
            constraints = []
            print("Constraints cleared.")
            continue

        if user_input.lower() == 'apply':
            if not constraints:
                print("No constraints to apply.")
                continue

            print("Applying constraints...")
            refiner = ForceDirectedRefiner(board)
            for c in constraints:
                refiner.add_constraint(c)
            result = refiner.refine()
            print(f"Refinement complete ({result.iteration} iterations)")

            scorer = ConfidenceScorer(dfm_profile)
            report = scorer.assess(board)
            print(f"Confidence: {report.overall_score:.0%}")
            continue

        if user_input.lower().startswith('save'):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                save_path = Path(parts[1])
            else:
                save_path = pcb_path.with_suffix('.placed.kicad_pcb')
            save_kicad_board(board, save_path)
            print(f"Saved to: {save_path}")
            continue

        if user_input.lower() == 'report':
            scorer = ConfidenceScorer(dfm_profile)
            report = scorer.assess(board)
            print(report.summary())
            continue

        # Try parsing as constraint
        parsed, summary = parser.parse_interactive(user_input)
        if parsed:
            constraints.extend(parsed)
            print(summary)
        else:
            # Try parsing as modification
            mod = modifier.parse_modification(user_input)
            if mod:
                if modifier.apply_modification(mod):
                    print(f"Applied: {mod['type']} on {mod.get('component', 'components')}")
                else:
                    print("Could not apply modification.")
            else:
                print(f"Could not parse: {user_input}")
                print("Try 'help' for examples.")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AtoPlace - AI-Powered PCB Placement Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atoplace place board.kicad_pcb
  atoplace place board.kicad_pcb --constraints "USB on left edge"
  atoplace validate board.kicad_pcb --dfm jlcpcb_standard
  atoplace interactive board.kicad_pcb
        """,
    )

    parser.add_argument('--version', action='version', version='atoplace 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Place command
    place_parser = subparsers.add_parser('place', help='Run placement optimization')
    place_parser.add_argument('board', help='Path to KiCad PCB file')
    place_parser.add_argument('-o', '--output', help='Output file path')
    place_parser.add_argument('-c', '--constraints', help='Natural language constraints')
    place_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')
    place_parser.add_argument('--iterations', type=int, help='Max iterations (default: 500)')
    place_parser.add_argument('--grid', type=float, help='Grid size for snapping (mm)')
    place_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    place_parser.add_argument('--dry-run', action='store_true', help="Don't save changes")

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate board placement')
    validate_parser.add_argument('board', help='Path to KiCad PCB file')
    validate_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')
    validate_parser.add_argument('-o', '--output', help='Save report to file')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate detailed report')
    report_parser.add_argument('board', help='Path to KiCad PCB file')
    report_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive',
                                                help='Interactive constraint session')
    interactive_parser.add_argument('board', help='Path to KiCad PCB file')
    interactive_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Check for pcbnew
    if not check_pcbnew():
        print("Error: pcbnew not available.")
        print("Run with KiCad's Python interpreter:")
        print("  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m atoplace ...")
        return 1

    # Dispatch command
    commands = {
        'place': cmd_place,
        'validate': cmd_validate,
        'report': cmd_report,
        'interactive': cmd_interactive,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
