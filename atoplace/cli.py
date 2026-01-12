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
from typing import Dict, List, Optional


def check_pcbnew():
    """Check if pcbnew is available."""
    try:
        import pcbnew
        return True
    except ImportError:
        return False


def load_board_from_path(board_arg: str, build: Optional[str] = None):
    """
    Load a board from a path, auto-detecting atopile projects.

    Args:
        board_arg: Path to .kicad_pcb file or atopile project directory
        build: Optional build name for atopile projects

    Returns:
        Tuple of (board, source_path, is_atopile)
    """
    from .board.atopile_adapter import (
        AtopileProjectLoader,
        detect_board_source,
    )
    from .board.kicad_adapter import load_kicad_board

    path = Path(board_arg)

    # Check if this is an atopile project
    if path.is_dir() and (path / "ato.yaml").exists():
        print(f"Detected atopile project: {path}")
        loader = AtopileProjectLoader(path)
        build_name = build or "default"

        try:
            board_path = loader.get_board_path(build_name)
            print(f"  Build: {build_name}")
            print(f"  Board: {board_path}")

            if not board_path.exists():
                print(f"Error: Board file not found: {board_path}")
                print("Run 'ato build' first to generate the board.")
                return None, None, True

            board = loader.load_board(build_name)
            return board, board_path, True

        except ValueError as e:
            print(f"Error: {e}")
            return None, None, True

    # Check for project root if path doesn't exist directly
    if not path.exists():
        project_root = AtopileProjectLoader.find_project_root(path)
        if project_root:
            return load_board_from_path(str(project_root), build)

        print(f"Error: Path not found: {path}")
        return None, None, False

    # Direct .kicad_pcb file
    if path.suffix == ".kicad_pcb":
        print(f"Loading KiCad board: {path}")
        board = load_kicad_board(path)
        return board, path, False

    # Directory without ato.yaml - look for .kicad_pcb files
    if path.is_dir():
        kicad_files = list(path.glob("*.kicad_pcb"))
        if kicad_files:
            board_path = kicad_files[0]
            print(f"Found KiCad board: {board_path}")
            board = load_kicad_board(board_path)
            return board, board_path, False

    print(f"Error: Cannot load board from: {path}")
    return None, None, False


def cmd_place(args):
    """Run placement optimization."""
    from .board.abstraction import Board
    from .board.kicad_adapter import save_kicad_board
    from .placement.force_directed import ForceDirectedRefiner, RefinementConfig
    from .placement.legalizer import PlacementLegalizer, LegalizerConfig
    from .placement.module_detector import ModuleDetector
    from .nlp.constraint_parser import ConstraintParser
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile

    # Load board with auto-detection
    build_name = getattr(args, 'build', None)
    board, pcb_path, is_atopile = load_board_from_path(args.board, build_name)

    if board is None:
        return 1
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

    # Add atopile module grouping constraints if requested
    if getattr(args, 'use_ato_modules', False) and is_atopile:
        from .placement.constraints import GroupingConstraint
        print("\nApplying atopile module grouping constraints...")

        # Collect components by their ato_module property
        modules_to_components: Dict[str, List[str]] = {}
        for ref, comp in board.components.items():
            ato_module = comp.properties.get("ato_module")
            if ato_module:
                if ato_module not in modules_to_components:
                    modules_to_components[ato_module] = []
                modules_to_components[ato_module].append(ref)

        # Create grouping constraints for modules with 2+ components
        for module_name, comp_refs in modules_to_components.items():
            if len(comp_refs) >= 2:
                constraint = GroupingConstraint(
                    components=comp_refs,
                    max_spread=15.0,  # 15mm max spread for module grouping
                    description=f"Group atopile module: {module_name}"
                )
                constraints.append(constraint)
                print(f"  {module_name}: {len(comp_refs)} components")

    # Get DFM profile for spacing rules
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")

    # Configure refinement with DFM-aware spacing
    config = RefinementConfig(
        max_iterations=args.iterations or 500,
        min_movement=0.01,
        damping=0.85,
        min_clearance=dfm_profile.min_spacing,
        preferred_clearance=dfm_profile.min_spacing * 2,  # 2x min for comfort
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

    # Run legalization pass (Manhattan aesthetics)
    if not getattr(args, 'skip_legalization', False):
        print("\nRunning legalization pass...")
        legalize_config = LegalizerConfig(
            primary_grid=args.grid if args.grid else 0.5,
            snap_rotation=True,
            align_passives_only=True,
            # Use DFM profile spacing to avoid reintroducing violations
            min_clearance=dfm_profile.min_spacing,
            row_spacing=dfm_profile.min_spacing * 1.5,  # Comfortable row spacing
        )
        legalizer = PlacementLegalizer(board, legalize_config)
        legal_result = legalizer.legalize()

        print(f"  Grid snapped: {legal_result.grid_snapped} components")
        print(f"  Rows formed: {legal_result.rows_formed} ({legal_result.components_aligned} components aligned)")
        print(f"  Overlaps resolved: {legal_result.overlaps_resolved} in {legal_result.iterations_used} iterations")
        if legal_result.final_overlaps > 0:
            print(f"  Warning: {legal_result.final_overlaps} overlaps remaining")
        if legal_result.locked_conflicts:
            print(f"  Warning: {len(legal_result.locked_conflicts)} locked component conflicts (cannot resolve):")
            for ref1, ref2 in legal_result.locked_conflicts[:5]:  # Show first 5
                print(f"    - {ref1} overlaps {ref2}")
            if len(legal_result.locked_conflicts) > 5:
                print(f"    - ... and {len(legal_result.locked_conflicts) - 5} more")

    # Validate result
    print("\nValidating placement...")
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
    from .validation.confidence import ConfidenceScorer
    from .validation.pre_route import PreRouteValidator
    from .validation.drc import DRCChecker
    from .dfm.profiles import get_profile

    # Load board with auto-detection
    build_name = getattr(args, 'build', None)
    board, pcb_path, is_atopile = load_board_from_path(args.board, build_name)

    if board is None:
        return 1

    # Run pre-route validation
    print("\nRunning pre-route validation...")
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")
    pre_validator = PreRouteValidator(board, dfm_profile)
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
    from .placement.module_detector import ModuleDetector
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile

    # Load board with auto-detection
    build_name = getattr(args, 'build', None)
    board, pcb_path, is_atopile = load_board_from_path(args.board, build_name)

    if board is None:
        return 1

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
    from .board.kicad_adapter import save_kicad_board
    from .nlp.constraint_parser import ConstraintParser, ModificationHandler
    from .placement.force_directed import ForceDirectedRefiner
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile

    # Load board with auto-detection
    build_name = getattr(args, 'build', None)
    board, pcb_path, is_atopile = load_board_from_path(args.board, build_name)

    if board is None:
        return 1

    parser = ConstraintParser(board)
    modifier = ModificationHandler(board)
    dfm_profile = get_profile(args.dfm or "jlcpcb_standard")

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
        # Always print summary to show warnings even if no constraints found
        if summary and summary != "No constraints found.":
            print(summary)
        if not parsed:
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
  # KiCad board files
  atoplace place board.kicad_pcb
  atoplace place board.kicad_pcb --constraints "USB on left edge"

  # Atopile projects (auto-detected)
  atoplace place .                              # Uses default build
  atoplace place ~/projects/my-sensor --build custom
  atoplace place . --use-ato-modules            # Use atopile hierarchy

  # Validation and reports
  atoplace validate board.kicad_pcb --dfm jlcpcb_standard
  atoplace interactive .
        """,
    )

    parser.add_argument('--version', action='version', version='atoplace 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Place command
    place_parser = subparsers.add_parser('place', help='Run placement optimization')
    place_parser.add_argument('board', help='Path to KiCad PCB file or atopile project directory')
    place_parser.add_argument('-o', '--output', help='Output file path')
    place_parser.add_argument('-c', '--constraints', help='Natural language constraints')
    place_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')
    place_parser.add_argument('--build', help='Build name for atopile projects (default: default)')
    place_parser.add_argument('--iterations', type=int, help='Max iterations (default: 500)')
    place_parser.add_argument('--grid', type=float, help='Grid size for snapping (mm)')
    place_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    place_parser.add_argument('--dry-run', action='store_true', help="Don't save changes")
    place_parser.add_argument('--use-ato-modules', action='store_true',
                              help='Use atopile module hierarchy for grouping constraints')
    place_parser.add_argument('--skip-legalization', action='store_true',
                              help='Skip the legalization pass (grid snapping, alignment)')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate board placement')
    validate_parser.add_argument('board', help='Path to KiCad PCB file or atopile project directory')
    validate_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')
    validate_parser.add_argument('--build', help='Build name for atopile projects (default: default)')
    validate_parser.add_argument('-o', '--output', help='Save report to file')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate detailed report')
    report_parser.add_argument('board', help='Path to KiCad PCB file or atopile project directory')
    report_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')
    report_parser.add_argument('--build', help='Build name for atopile projects (default: default)')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive',
                                                help='Interactive constraint session')
    interactive_parser.add_argument('board', help='Path to KiCad PCB file or atopile project directory')
    interactive_parser.add_argument('--dfm', help='DFM profile (default: jlcpcb_standard)')
    interactive_parser.add_argument('--build', help='Build name for atopile projects (default: default)')

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
