#!/usr/bin/env python3
"""
AtoPlace CLI

Command-line interface for the AI-powered PCB placement and routing tool.
"""

from __future__ import annotations

import io
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="AtoPlace - AI-powered PCB placement and validation CLI.",
)

_LOG_FILE_HANDLE: Optional[io.TextIOBase] = None
logger = logging.getLogger(__name__)


@dataclass
class CLIContext:
    console: Console
    verbose: bool
    log_path: Optional[Path]


class _AnsiStrippingTee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, data):
        if not data:
            return 0
        # Handle both str and bytes (click sometimes writes bytes)
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        for stream in self._streams:
            if stream is _LOG_FILE_HANDLE:
                stream.write(_strip_ansi(data))
            else:
                stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_PATTERN.sub("", text)


def _configure_logging(
    *,
    verbose: bool,
    log_file: Optional[Path],
    log_dir: Path,
    console: Console,
) -> Path:
    log_path = log_file if log_file else log_dir / (
        f"atoplace-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    console_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        show_level=True,
        rich_tracebacks=verbose,
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(console_handler)

    global _LOG_FILE_HANDLE
    _LOG_FILE_HANDLE = log_path.open("a", encoding="utf-8")
    sys.stdout = _AnsiStrippingTee(sys.stdout, _LOG_FILE_HANDLE)
    sys.stderr = _AnsiStrippingTee(sys.stderr, _LOG_FILE_HANDLE)

    return log_path


def _get_context(ctx: typer.Context) -> CLIContext:
    return ctx.obj  # type: ignore[return-value]


def _render_board_summary(console: Console, board, outline_note: Optional[str]):
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Components", str(len(board.components)))
    table.add_row("Nets", str(len(board.nets)))
    table.add_row("Layers", str(board.layer_count))
    if outline_note:
        table.add_row("Outline", outline_note)
    console.print(Panel(table, title="Board", border_style="cyan"))


def _render_module_summary(console: Console, modules) -> None:
    module_type_counts: Dict[str, int] = {}
    module_type_components: Dict[str, int] = {}
    for module in modules:
        if module.components:
            mtype = module.module_type.value
            module_type_counts[mtype] = module_type_counts.get(mtype, 0) + 1
            module_type_components[mtype] = (
                module_type_components.get(mtype, 0) + len(module.components)
            )

    if not module_type_counts:
        console.print(Panel("No modules detected", border_style="yellow"))
        return

    table = Table(title="Detected Modules", box=box.SIMPLE, header_style="bold")
    table.add_column("Module")
    table.add_column("Components", justify="right")
    table.add_column("Groups", justify="right")

    for mtype in sorted(module_type_counts):
        table.add_row(
            mtype,
            str(module_type_components[mtype]),
            str(module_type_counts[mtype]),
        )

    console.print(table)


def _render_constraint_summary(console: Console, constraints) -> None:
    if not constraints:
        console.print(Panel("No constraints provided", border_style="yellow"))
        return

    table = Table(title="Constraints", box=box.SIMPLE, header_style="bold")
    table.add_column("Description")
    table.add_column("Priority", justify="right")

    for constraint in constraints:
        table.add_row(
            getattr(constraint, "description", "(unnamed)"),
            getattr(constraint, "priority", "preferred"),
        )

    console.print(table)


def _render_legalization_summary(console: Console, legal_result) -> None:
    table = Table(title="Legalization", box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Grid snapped", str(legal_result.grid_snapped))
    table.add_row("Rows formed", str(legal_result.rows_formed))
    table.add_row("Components aligned", str(legal_result.components_aligned))
    table.add_row("Overlaps resolved", str(legal_result.overlaps_resolved))
    table.add_row("Overlap iterations", str(legal_result.iterations_used))
    table.add_row("Final overlaps", str(legal_result.final_overlaps))
    console.print(table)

    if legal_result.locked_conflicts:
        conflicts = "\n".join(
            f"- {ref1} overlaps {ref2}"
            for ref1, ref2 in legal_result.locked_conflicts[:5]
        )
        if len(legal_result.locked_conflicts) > 5:
            conflicts += f"\n- ... and {len(legal_result.locked_conflicts) - 5} more"
        console.print(Panel(conflicts, title="Locked Conflicts", border_style="yellow"))


def _render_validation_summary(console: Console, report) -> None:
    console.print(Panel(report.summary(), title="Confidence", border_style="green"))


def _render_drc_summary(console: Console, drc) -> None:
    console.print(Panel(drc.get_summary(), title="DRC", border_style="magenta"))


def _render_pre_route_summary(console: Console, pre_validator) -> None:
    console.print(Panel(pre_validator.get_summary(), title="Pre-Route", border_style="magenta"))


def check_pcbnew() -> bool:
    """Check if pcbnew is available."""
    try:
        import pcbnew  # noqa: F401

        return True
    except ImportError:
        return False


def load_board_from_path(
    board_arg: str,
    build: Optional[str] = None,
    console: Optional[Console] = None,
    apply_lock: bool = False,
    only_locked: bool = False,
) -> Tuple[Optional[object], Optional[Path], bool, Optional[object]]:
    """
    Load a board from a path, auto-detecting atopile projects.

    Args:
        board_arg: Path to board file or atopile project directory
        build: Optional atopile build name
        console: Console for output
        apply_lock: If True, apply saved positions from atoplace.lock
        only_locked: If True, only apply positions marked as locked

    Returns:
        Tuple of (board, source_path, is_atopile, loader)
        The loader is returned for atopile projects to enable lock file saving.
    """
    from .board.atopile_adapter import AtopileProjectLoader
    from .board.kicad_adapter import load_kicad_board

    emit = console.print if console else print
    path = Path(board_arg)

    if path.is_dir() and (path / "ato.yaml").exists():
        emit(f"Detected atopile project: [bold]{path}[/bold]")
        loader = AtopileProjectLoader(path)
        build_name = build or "default"

        try:
            board_path = loader.get_board_path(build_name)
            emit(f"  Build: [bold]{build_name}[/bold]")
            emit(f"  Board: [bold]{board_path}[/bold]")

            if not board_path.exists():
                emit(f"[red]Error:[/red] Board file not found: {board_path}")
                emit("Run 'ato build' first to generate the board.")
                return None, None, True, None

            # Check for lock file
            if apply_lock:
                lock = loader.get_lock(build_name)
                if lock:
                    locked_count = len(lock.get_locked_refs())
                    emit(f"  Lock file: [bold]{len(lock.components)}[/bold] positions ({locked_count} locked)")
                else:
                    emit("  Lock file: [dim]not found[/dim]")

            board = loader.load_board(build_name, apply_lock=apply_lock, only_locked=only_locked)
            return board, board_path, True, loader

        except ValueError as exc:
            emit(f"[red]Error:[/red] {exc}")
            return None, None, True, None

    if not path.exists():
        project_root = AtopileProjectLoader.find_project_root(path)
        if project_root:
            return load_board_from_path(str(project_root), build, console, apply_lock, only_locked)

        emit(f"[red]Error:[/red] Path not found: {path}")
        return None, None, False, None

    if path.suffix == ".kicad_pcb":
        # Check if this kicad_pcb file is inside an atopile project
        project_root = AtopileProjectLoader.find_project_root(path.parent)
        if project_root:
            emit(f"Detected atopile project: [bold]{project_root}[/bold]")
            loader = AtopileProjectLoader(project_root)

            # Try to determine which build this file belongs to
            build_name = build or "default"

            # Check if this file matches any build's board path
            try:
                expected_board = loader.get_board_path(build_name)
                if path.resolve() == expected_board.resolve():
                    emit(f"  Build: [bold]{build_name}[/bold]")

                    # Check for lock file
                    if apply_lock:
                        lock = loader.get_lock(build_name)
                        if lock:
                            locked_count = len(lock.get_locked_refs())
                            emit(f"  Lock file: [bold]{len(lock.components)}[/bold] positions ({locked_count} locked)")
                        else:
                            emit("  Lock file: [dim]not found[/dim]")

                    board = loader.load_board(build_name, apply_lock=apply_lock, only_locked=only_locked)
                    return board, path, True, loader
            except ValueError:
                pass

            # Fall back to loading as plain KiCad but with atopile enrichment attempt
            emit(f"Loading KiCad board: [bold]{path}[/bold] (with atopile enrichment)")
            board = load_kicad_board(path)
            try:
                # Apply atopile metadata: component values and module hierarchy
                loader._apply_component_metadata(board)
                loader._apply_module_hierarchy(board, build_name)
                return board, path, True, loader
            except Exception as e:
                emit(f"[yellow]Warning:[/yellow] Could not apply atopile metadata: {e}")
                return board, path, False, None

        emit(f"Loading KiCad board: [bold]{path}[/bold]")
        board = load_kicad_board(path)
        return board, path, False, None

    if path.is_dir():
        kicad_files = sorted(path.glob("*.kicad_pcb"))
        if kicad_files:
            if len(kicad_files) > 1:
                emit(
                    f"Found {len(kicad_files)} KiCad boards - using first alphabetically:"
                )
                for kicad in kicad_files:
                    emit(f"  - {kicad.name}")
            board_path = kicad_files[0]
            emit(f"Loading KiCad board: [bold]{board_path}[/bold]")
            board = load_kicad_board(board_path)
            return board, board_path, False, None

    emit(f"[red]Error:[/red] Cannot load board from: {path}")
    return None, None, False, None


def _generate_full_validation_report(report, pre_validator, drc, pre_route_passed, drc_passed):
    """Generate comprehensive validation report including all checks."""
    lines = [
        "# Validation Report",
        "",
        "## Pre-Route Validation",
        "",
        f"**Status:** {'PASSED' if pre_route_passed else 'FAILED'}",
        "",
    ]

    if pre_validator.issues:
        lines.append("### Issues")
        lines.append("")
        for issue in pre_validator.issues:
            severity_emoji = {"error": "X", "warning": "!", "info": "i"}
            lines.append(
                f"- [{severity_emoji.get(issue.severity, '?')}] [{issue.category}] {issue.message}"
            )
            if issue.location:
                lines.append(f"  - Location: {issue.location}")
        lines.append("")
    else:
        lines.append("No pre-route issues found.")
        lines.append("")

    lines.extend(
        [
            "## DRC Checks",
            "",
            f"**Status:** {'PASSED' if drc_passed else 'FAILED'}",
            "",
        ]
    )

    violations = drc.violations if hasattr(drc, "violations") else []
    if violations:
        lines.append("### Violations")
        lines.append("")
        for violation in violations:
            lines.append(f"- [{violation.rule}] {violation.message}")
            if violation.location:
                lines.append(f"  - Location: {violation.location}")
        lines.append("")
    else:
        lines.append("No DRC violations found.")
        lines.append("")

    lines.extend(["---", "", report.to_markdown()])
    return "\n".join(lines)


@app.callback()
def _app_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logs."),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Write debug logs to this file."
    ),
    log_dir: Path = typer.Option(
        Path("logs"),
        "--log-dir",
        help="Directory for debug logs.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show version and exit.",
        callback=lambda value: _version_callback(value),
        is_eager=True,
    ),
):
    """AtoPlace - AI-powered PCB placement tool."""
    console = Console(no_color=no_color)
    log_path = _configure_logging(
        verbose=verbose,
        log_file=log_file,
        log_dir=log_dir,
        console=console,
    )
    ctx.obj = CLIContext(console=console, verbose=verbose, log_path=log_path)


def _version_callback(value: Optional[bool]) -> None:
    if value:
        typer.echo("atoplace 0.1.0")
        raise typer.Exit()


@app.command()
def place(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    constraints: Optional[str] = typer.Option(
        None, "--constraints", "-c", help="Placement constraints (natural language)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for placed board."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not save output."),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-i", help="Max refinement iterations."
    ),
    grid: Optional[float] = typer.Option(
        None, "--grid", help="Snap-to-grid size in mm."
    ),
    skip_legalization: bool = typer.Option(
        False,
        "--skip-legalization",
        help="Skip legalization pass (grid snap, alignment).",
    ),
    use_ato_modules: bool = typer.Option(
        False,
        "--use-ato-modules",
        help="Use atopile module grouping constraints.",
    ),
    build: Optional[str] = typer.Option(
        None,
        "--build",
        help="Atopile build name (default: default).",
    ),
    auto_outline: bool = typer.Option(
        False,
        "--auto-outline",
        help="Auto-generate outline when missing (component bounding box).",
    ),
    outline_margin: float = typer.Option(
        5.0,
        "--outline-margin",
        help="Margin for auto-generated outline in mm.",
    ),
    compact_outline: bool = typer.Option(
        False,
        "--compact-outline",
        help="Iteratively compact outline to minimum feasible size.",
    ),
    outline_clearance: float = typer.Option(
        0.25,
        "--outline-clearance",
        help="Clearance used when compacting outline in mm.",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        help="Generate interactive HTML visualization for debugging.",
    ),
    # Lock file options for atopile sidecar persistence
    use_lock: bool = typer.Option(
        False,
        "--use-lock",
        help="Apply saved positions from atoplace.lock before placement.",
    ),
    only_locked: bool = typer.Option(
        False,
        "--only-locked",
        help="Only apply positions marked as locked (user-approved) from lock file.",
    ),
    save_lock: bool = typer.Option(
        True,
        "--save-lock/--no-save-lock",
        help="Save positions to atoplace.lock after placement (default: True for atopile projects).",
    ),
    lock_all: bool = typer.Option(
        False,
        "--lock-all",
        help="Mark all positions as locked (user-approved) when saving lock file.",
    ),
):
    """Run placement optimization."""
    from .board.kicad_adapter import save_kicad_board
    from .placement.force_directed import ForceDirectedRefiner, RefinementConfig
    from .placement.legalizer import PlacementLegalizer, LegalizerConfig
    from .placement.module_detector import ModuleDetector
    from .placement.visualizer import PlacementVisualizer
    from .nlp.constraint_parser import ConstraintParser
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile, get_profile_for_layers

    context = _get_context(ctx)
    console = context.console

    console.print(Rule("Load Board", style="cyan"))
    board_obj, pcb_path, is_atopile, ato_loader = load_board_from_path(
        str(board), build, console, apply_lock=use_lock, only_locked=only_locked
    )
    if board_obj is None or pcb_path is None:
        raise typer.Exit(code=1)

    # Determine build name for lock file operations
    build_name = build or "default"

    outline_note = "present" if board_obj.outline.has_outline else "missing"
    if not board_obj.outline.has_outline:
        if compact_outline:
            board_obj.outline = board_obj.compact_outline(
                initial_margin=10.0,
                min_margin=1.0,
                clearance=outline_clearance,
                shrink_step=0.5,
            )
            outline_note = (
                f"compacted to {board_obj.outline.width:.1f}x"
                f"{board_obj.outline.height:.1f}mm"
            )
        elif auto_outline:
            board_obj.outline = board_obj.generate_outline_from_components(
                margin=outline_margin
            )
            outline_note = (
                f"auto {board_obj.outline.width:.1f}x"
                f"{board_obj.outline.height:.1f}mm"
            )
        else:
            outline_note = "missing (boundary checks disabled)"

    _render_board_summary(console, board_obj, outline_note)

    logger.debug(
        "Board loaded: path=%s components=%d nets=%d layers=%d atopile=%s",
        pcb_path,
        len(board_obj.components),
        len(board_obj.nets),
        board_obj.layer_count,
        is_atopile,
    )

    console.print(Rule("Detect Modules", style="cyan"))
    detector = ModuleDetector(board_obj)
    modules = detector.detect()
    _render_module_summary(console, modules)

    # Create visualizer if requested
    visualizer = None
    module_map = {}  # ref -> module_type
    if visualize:
        visualizer = PlacementVisualizer(board_obj)
        # Build module map from detected modules (heuristic-based)
        for module in modules:
            for ref in module.components:
                module_map[ref] = module.module_type.value

        # If atopile project, override with atopile module names for visualization
        # This provides more meaningful module names than heuristic detection
        if is_atopile:
            for ref, comp in board_obj.components.items():
                ato_module = comp.properties.get("ato_module")
                if ato_module:
                    module_map[ref] = ato_module

    constraints_list = []
    if constraints:
        console.print(Rule("Constraints", style="cyan"))
        parser = ConstraintParser(board_obj)
        parsed, summary = parser.parse_interactive(constraints)
        constraints_list = parsed
        console.print(Panel(summary, border_style="blue"))
        _render_constraint_summary(console, constraints_list)

    if use_ato_modules and is_atopile:
        from .placement.constraints import GroupingConstraint

        console.print(Rule("Atopile Module Grouping", style="cyan"))
        modules_to_components: Dict[str, List[str]] = {}
        for ref, comp in board_obj.components.items():
            ato_module = comp.properties.get("ato_module")
            if ato_module:
                modules_to_components.setdefault(ato_module, []).append(ref)

        table = Table(box=box.SIMPLE, header_style="bold")
        table.add_column("Module")
        table.add_column("Components", justify="right")
        for module_name, comp_refs in modules_to_components.items():
            if len(comp_refs) >= 2:
                constraint = GroupingConstraint(
                    components=comp_refs,
                    max_spread=15.0,
                    optimize_bbox=True,  # Enable tight module bounding box
                    bbox_strength=1.0,
                    min_clearance=0.25,  # Default clearance, updated after DFM profile
                    description=f"Group atopile module: {module_name}",
                )
                constraints_list.append(constraint)
                table.add_row(module_name, str(len(comp_refs)))
        console.print(table)

    console.print(Rule("DFM Profile", style="cyan"))
    if dfm:
        try:
            dfm_profile = get_profile(dfm)
        except ValueError as exc:
            from .dfm.profiles import list_profiles

            console.print(
                Panel(
                    f"[red]Invalid DFM profile[/red]: {exc}\n"
                    f"Available: {', '.join(list_profiles())}",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
    else:
        dfm_profile = get_profile_for_layers(board_obj.layer_count)

    dfm_table = Table(box=box.SIMPLE, show_header=False)
    dfm_table.add_column("Field", style="bold")
    dfm_table.add_column("Value")
    dfm_table.add_row("Profile", dfm_profile.name)
    dfm_table.add_row("Min spacing", f"{dfm_profile.min_spacing:.3f} mm")
    console.print(Panel(dfm_table, border_style="cyan"))

    config = RefinementConfig(
        max_iterations=iterations or 500,
        min_movement=0.01,
        damping=0.85,
        min_clearance=dfm_profile.min_spacing,
        edge_clearance=dfm_profile.min_trace_to_edge,
        preferred_clearance=dfm_profile.min_spacing * 2,
        lock_placed=True,
    )
    if grid:
        config.snap_to_grid = True
        config.grid_size = grid

    logger.debug(
        "Refinement configured: iterations=%d grid=%s snap=%s min_clearance=%.3f preferred=%.3f",
        config.max_iterations,
        f"{config.grid_size:.3f}mm" if config.grid_size else "none",
        config.snap_to_grid,
        config.min_clearance,
        config.preferred_clearance,
    )

    console.print(Rule("Refinement", style="cyan"))
    refiner = ForceDirectedRefiner(
        board_obj, config,
        visualizer=visualizer,
        modules=module_map,
    )
    for constraint in constraints_list:
        refiner.add_constraint(constraint)

    result = None
    if context.verbose:
        with console.status("Running force-directed refinement..."):
            result = refiner.refine()
    else:
        progress_columns = [
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ]
        with Progress(*progress_columns, console=console) as progress:
            task_id = progress.add_task(
                "Refining placement", total=config.max_iterations
            )
            last_update = -1

            def progress_callback(state):
                nonlocal last_update
                if state.iteration - last_update >= 5 or state.iteration == 0:
                    progress.update(
                        task_id,
                        completed=state.iteration,
                        description=f"Refining (E={state.total_energy:.1f})",
                    )
                    last_update = state.iteration

            result = refiner.refine(callback=progress_callback)
            progress.update(task_id, completed=result.iteration)

    if result is None:
        raise typer.Exit(code=1)

    refinement_table = Table(box=box.SIMPLE, show_header=False)
    refinement_table.add_column("Metric", style="bold")
    refinement_table.add_column("Value", justify="right")
    refinement_table.add_row("Converged", "yes" if result.converged else "no")
    refinement_table.add_row("Iterations", str(result.iteration))
    refinement_table.add_row("Energy", f"{result.total_energy:.2f}")
    if config.snap_to_grid:
        refinement_table.add_row("Grid", f"{config.grid_size:.2f} mm")
    console.print(Panel(refinement_table, title="Refinement Summary", border_style="green"))

    logger.debug(
        "Refinement result: converged=%s iterations=%d energy=%.3f",
        result.converged,
        result.iteration,
        result.total_energy,
    )

    if not skip_legalization:
        console.print(Rule("Legalization", style="cyan"))
        # Use STRICT overlap prevention settings
        # Use at least 0.35mm clearance, or DFM profile if stricter
        strict_clearance = max(0.35, dfm_profile.min_spacing)
        # Only compact outline if board had no explicit outline to begin with
        # This preserves user-designed board shapes while auto-generating
        # outlines for boards that had none
        should_compact_outline = not board_obj.outline.has_outline or compact_outline
        legalize_config = LegalizerConfig(
            primary_grid=grid if grid else 0.5,
            snap_rotation=True,
            align_passives_only=True,
            min_clearance=strict_clearance,  # Strict clearance to prevent any overlaps
            edge_clearance=max(0.4, dfm_profile.min_trace_to_edge),
            row_spacing=strict_clearance * 2,
            guarantee_zero_overlaps=True,  # CRITICAL: Ensure no overlapping components
            max_displacement_iterations=1000,  # More iterations for thorough resolution
            overlap_retry_passes=50,  # More passes to guarantee resolution
            escalation_factor=1.3,  # Gentler escalation for better results
            compact_outline=should_compact_outline,  # Only compact when appropriate
        )
        legalizer = PlacementLegalizer(
            board_obj, legalize_config, constraints=constraints_list
        )
        with console.status("Running legalization pass..."):
            legal_result = legalizer.legalize()
        _render_legalization_summary(console, legal_result)
        logger.debug(
            "Legalizer result: snapped=%d rows=%d aligned=%d overlaps_resolved=%d final_overlaps=%d",
            legal_result.grid_snapped,
            legal_result.rows_formed,
            legal_result.components_aligned,
            legal_result.overlaps_resolved,
            legal_result.final_overlaps,
        )

        # Capture legalization result frame
        if visualizer:
            visualizer.capture_from_board(
                label="After Legalization",
                iteration=0,
                phase="legalization",
                modules=module_map,
            )

    console.print(Rule("Validation", style="cyan"))
    scorer = ConfidenceScorer(dfm_profile)
    report = scorer.assess(board_obj)
    _render_validation_summary(console, report)

    output_path = output if output else pcb_path.with_suffix(".placed.kicad_pcb")

    # Reposition ref des text to avoid overlaps
    with console.status("Repositioning reference designators..."):
        repositioned = board_obj.reposition_ref_des_text(
            clearance=0.2,
            pad_clearance=dfm_profile.min_silk_to_pad if hasattr(dfm_profile, 'min_silk_to_pad') else 0.15
        )
    if repositioned > 0:
        console.print(f"  Repositioned {repositioned} reference designators")

    if dry_run:
        console.print(Panel("Dry run - output not saved", border_style="yellow"))
    else:
        with console.status("Saving placement..."):
            save_kicad_board(board_obj, output_path)
        console.print(Panel(f"Saved to {output_path}", border_style="green"))

        # Save lock file for atopile projects (sidecar persistence)
        if is_atopile and save_lock and ato_loader:
            console.print(Rule("Lock File", style="cyan"))
            with console.status("Saving placement lock file..."):
                lock_saved = ato_loader.save_lock(
                    board_obj,
                    build_name=build_name,
                    lock_all=lock_all,
                    preserve_locked=True,
                )
            if lock_saved:
                lock_path = ato_loader.get_lock_path(build_name)
                lock = ato_loader.get_lock(build_name)
                locked_count = len(lock.get_locked_refs()) if lock else 0
                console.print(
                    Panel(
                        f"Saved to {lock_path}\n"
                        f"  {len(lock.components) if lock else 0} positions"
                        f" ({locked_count} locked)",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel("Failed to save lock file", border_style="yellow")
                )

    # Export visualization if enabled
    if visualizer:
        viz_path = visualizer.export_html_report(
            filename="placement_debug.html",
            output_dir="placement_debug",
        )
        console.print(Panel(f"Visualization: {viz_path}", border_style="magenta"))

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))
    raise typer.Exit(code=0 if report.overall_score >= 0.7 else 1)


@app.command()
def validate(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write markdown report to file."
    ),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
):
    """Validate board placement."""
    from .validation.confidence import ConfidenceScorer
    from .validation.pre_route import PreRouteValidator
    from .validation.drc import DRCChecker
    from .dfm.profiles import get_profile, get_profile_for_layers

    context = _get_context(ctx)
    console = context.console

    console.print(Rule("Load Board", style="cyan"))
    board_obj, _, _, _ = load_board_from_path(str(board), build, console)
    if board_obj is None:
        raise typer.Exit(code=1)

    if dfm:
        try:
            dfm_profile = get_profile(dfm)
        except ValueError as exc:
            from .dfm.profiles import list_profiles

            console.print(
                Panel(
                    f"[red]Invalid DFM profile[/red]: {exc}\n"
                    f"Available: {', '.join(list_profiles())}",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
    else:
        dfm_profile = get_profile_for_layers(board_obj.layer_count)

    console.print(Rule("Pre-Route Validation", style="cyan"))
    pre_validator = PreRouteValidator(board_obj, dfm_profile)
    can_proceed, _ = pre_validator.validate()
    _render_pre_route_summary(console, pre_validator)

    console.print(Rule("DRC", style="cyan"))
    drc = DRCChecker(board_obj, dfm_profile)
    drc_passed, _ = drc.run_checks()
    _render_drc_summary(console, drc)

    console.print(Rule("Confidence", style="cyan"))
    scorer = ConfidenceScorer(dfm_profile)
    report = scorer.assess(board_obj)
    _render_validation_summary(console, report)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        full_report = _generate_full_validation_report(
            report, pre_validator, drc, can_proceed, drc_passed
        )
        output.write_text(full_report)
        console.print(Panel(f"Report saved to {output}", border_style="green"))

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))
    confidence_ok = report.overall_score >= 0.7
    raise typer.Exit(code=0 if drc_passed and can_proceed and confidence_ok else 1)


@app.command()
def route(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file (default: overwrites input)."
    ),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
    greedy: float = typer.Option(
        2.0, "--greedy", "-g",
        help="A* greedy multiplier (1=optimal, 2-3=fast). Default: 2.0"
    ),
    grid: float = typer.Option(
        0.1, "--grid", help="Routing grid size in mm. Default: 0.1"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Generate routing visualization."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Don't save routed traces to file."
    ),
):
    """Route all nets on a board using A* pathfinding.

    Uses weighted A* (greedy multiplier) for fast routing with acceptable
    path quality. Supports multi-layer routing with vias.

    Traces are saved to the output file (or input file if no output specified).

    Example:
        atoplace route board.kicad_pcb --greedy 2.5 --visualize
        atoplace route board.kicad_pcb -o routed_board.kicad_pcb
    """
    from .routing import (
        AStarRouter,
        RouterConfig,
        ObstacleMapBuilder,
        NetOrderer,
        create_visualizer_from_board,
    )
    from .dfm.profiles import get_profile, get_profile_for_layers

    context: CLIContext = ctx.obj
    console = context.console

    console.print(Rule("Routing"))

    # Load board
    board_obj, pcb_path, _, _ = load_board_from_path(str(board), build, console)
    if board_obj is None:
        raise typer.Exit(code=1)

    # Get DFM profile
    if dfm:
        try:
            dfm_profile = get_profile(dfm)
        except ValueError as e:
            from .dfm.profiles import list_profiles
            console.print(f"[red]Error:[/] {e}")
            console.print(f"Available profiles: {', '.join(list_profiles())}")
            raise typer.Exit(code=1)
    else:
        dfm_profile = get_profile_for_layers(board_obj.layer_count)

    console.print(f"DFM Profile: [cyan]{dfm_profile.name}[/]")

    # Build obstacle map
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Building obstacle map...", total=None)
        builder = ObstacleMapBuilder(board_obj, dfm_profile)
        obstacle_index = builder.build()
        nets = builder.get_net_pads()

    stats = builder.get_routing_stats()
    console.print(f"Nets to route: [cyan]{len(nets)}[/] ({stats['total_pads']} pads)")
    console.print(f"Routing difficulty: [cyan]{stats['estimated_difficulty']}[/]")

    # Create visualizer if requested
    viz = None
    if visualize:
        viz = create_visualizer_from_board(board_obj)
        console.print("Visualization: [cyan]enabled[/]")

    # Configure router with board's layer count and default DFM values
    config = RouterConfig(
        greedy_weight=greedy,
        grid_size=grid,
        trace_width=dfm_profile.min_trace_width,
        clearance=dfm_profile.min_spacing,
        layer_count=board_obj.layer_count,
    )

    # Build lookup for per-net trace width and clearance from board's net definitions
    # This allows respecting KiCad's net-class settings
    net_trace_widths = {}
    net_clearances = {}
    for net_name, net_obj in board_obj.nets.items():
        if net_obj.trace_width is not None:
            net_trace_widths[net_name] = net_obj.trace_width
        if net_obj.clearance is not None:
            net_clearances[net_name] = net_obj.clearance

    if net_trace_widths or net_clearances:
        console.print(f"Per-net rules: [cyan]{len(net_trace_widths)} widths, {len(net_clearances)} clearances[/]")

    # Create router
    router = AStarRouter(obstacle_index, config, viz)

    # Order nets by difficulty
    orderer = NetOrderer(obstacle_index)
    ordered_nets = orderer.order_nets(nets)

    # Route each net
    results = {}
    success_count = 0
    total_length = 0.0
    total_vias = 0
    net_id_to_name = {}  # For saving traces later

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Routing nets...", total=len(ordered_nets))

        for net in ordered_nets:
            # Use per-net trace width and clearance if available, else DFM defaults
            trace_width = net_trace_widths.get(net.net_name, config.trace_width)
            net_clearance = net_clearances.get(net.net_name, config.clearance)

            # Track net_id to name mapping for saving traces
            net_id_to_name[net.net_id] = net.net_name

            result = router.route_net(
                pads=net.pads,
                net_name=net.net_name,
                net_id=net.net_id,
                trace_width=trace_width,
                clearance=net_clearance
            )
            results[net.net_name] = result

            if result.success:
                success_count += 1
                total_length += result.total_length
                total_vias += result.via_count

                # Add routed traces as obstacles for subsequent nets
                for seg in result.segments:
                    seg.net_id = net.net_id
                    router.add_routed_trace(seg)
                for via in result.vias:
                    via.net_id = net.net_id
                    router.add_routed_via(via)

            progress.update(task, advance=1)

    # Display results
    console.print()
    table = Table(title="Routing Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Nets routed", f"{success_count}/{len(nets)}")
    table.add_row("Success rate", f"{100*success_count/max(len(nets),1):.1f}%")
    table.add_row("Total trace length", f"{total_length:.1f}mm")
    table.add_row("Total vias", str(total_vias))

    console.print(table)

    # Show failed nets
    failed = [name for name, r in results.items() if not r.success]
    if failed:
        console.print(f"\n[yellow]Failed nets ({len(failed)}):[/]")
        for name in failed[:10]:  # Show first 10
            console.print(f"  - {name}: {results[name].failure_reason}")
        if len(failed) > 10:
            console.print(f"  ... and {len(failed) - 10} more")

    # Export visualization
    if viz:
        # Capture final state
        viz.capture_frame(
            obstacles=[],
            pads=[],
            completed_traces=[seg for r in results.values() for seg in r.segments],
            completed_vias=[v for r in results.values() for v in r.vias],
            label="Final routing"
        )
        report_path = viz.export_html_report("routing_result.html")
        console.print(Panel(f"Visualization: {report_path}", border_style="green"))

    # Save routed traces to KiCad file
    if not dry_run and success_count > 0:
        from .board.kicad_adapter import save_routed_traces
        import shutil

        # Collect all successful traces and vias
        all_segments = [seg for r in results.values() if r.success for seg in r.segments]
        all_vias = [via for r in results.values() if r.success for via in r.vias]

        # Determine output path
        output_path = output if output else pcb_path
        if output_path != pcb_path:
            # Copy source to output first to preserve all elements
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pcb_path, output_path)

        try:
            save_routed_traces(
                output_path,
                all_segments,
                all_vias,
                net_id_to_name,
                board_obj.layer_count
            )
            console.print(Panel(
                f"Saved {len(all_segments)} traces, {len(all_vias)} vias to {output_path}",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Failed to save traces:[/] {e}")
    elif dry_run:
        console.print("[yellow]Dry run - traces not saved[/]")

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))

    # Exit code: 0 if >80% routed, 1 otherwise
    success_rate = success_count / max(len(nets), 1)
    raise typer.Exit(code=0 if success_rate >= 0.8 else 1)


@app.command()
def report(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
):
    """Generate a detailed report for a board."""
    from .placement.module_detector import ModuleDetector
    from .validation.confidence import ConfidenceScorer
    from .validation.pre_route import PreRouteValidator
    from .validation.drc import DRCChecker
    from .dfm.profiles import get_profile, get_profile_for_layers

    context = _get_context(ctx)
    console = context.console

    console.print(Rule("Load Board", style="cyan"))
    board_obj, _, _, _ = load_board_from_path(str(board), build, console)
    if board_obj is None:
        raise typer.Exit(code=1)

    detector = ModuleDetector(board_obj)
    modules = detector.detect()

    if dfm:
        try:
            dfm_profile = get_profile(dfm)
        except ValueError as exc:
            from .dfm.profiles import list_profiles

            console.print(
                Panel(
                    f"[red]Invalid DFM profile[/red]: {exc}\n"
                    f"Available: {', '.join(list_profiles())}",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
    else:
        dfm_profile = get_profile_for_layers(board_obj.layer_count)

    console.print(Rule("Pre-Route Validation", style="cyan"))
    pre_validator = PreRouteValidator(board_obj, dfm_profile)
    can_proceed, _ = pre_validator.validate()
    _render_pre_route_summary(console, pre_validator)

    console.print(Rule("DRC", style="cyan"))
    drc = DRCChecker(board_obj, dfm_profile)
    drc_passed, _ = drc.run_checks()
    _render_drc_summary(console, drc)

    console.print(Rule("Confidence", style="cyan"))
    scorer = ConfidenceScorer(dfm_profile)
    report_obj = scorer.assess(board_obj)
    _render_validation_summary(console, report_obj)

    console.print(Rule("Modules", style="cyan"))
    _render_module_summary(console, modules)

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))
    confidence_ok = report_obj.overall_score >= 0.7
    raise typer.Exit(code=0 if drc_passed and can_proceed and confidence_ok else 1)


@app.command("mcp")
def mcp_serve(
    ctx: typer.Context,
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport mode: 'stdio' (default) or 'sse' for HTTP.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level for MCP server (DEBUG, INFO, WARNING, ERROR).",
    ),
    ipc: bool = typer.Option(
        False,
        "--ipc",
        "-i",
        help="Use IPC mode to communicate with KiCad bridge (for Python 3.9 compatibility).",
    ),
    socket: Optional[str] = typer.Option(
        None,
        "--socket",
        "-s",
        help="Unix socket path for IPC communication.",
    ),
    launch: bool = typer.Option(
        False,
        "--launch",
        help="Auto-launch both KiCad bridge and MCP server (recommended for IPC mode).",
    ),
):
    """Start the MCP server for LLM agent integration.

    The MCP (Model Context Protocol) server exposes AtoPlace tools to LLM agents
    like Claude. Tools include board management, placement actions, discovery,
    topology analysis, context generation, and validation.

    Modes:
        - Direct: Uses pcbnew directly (requires KiCad's Python 3.9)
        - IPC: Communicates with KiCad bridge process (works with Python 3.10+)
        - Launch: Starts both bridge and server automatically (recommended)

    Examples:
        atoplace mcp                    # Start with stdio transport (direct mode)
        atoplace mcp --ipc              # Start in IPC mode (needs bridge running)
        atoplace mcp --launch           # Auto-launch bridge and server
        atoplace mcp --log-level DEBUG  # Enable debug logging
    """
    import logging as mcp_logging

    context = _get_context(ctx)
    console = context.console

    # Configure MCP-specific logging
    mcp_logger = mcp_logging.getLogger("atoplace.mcp")
    mcp_logger.setLevel(getattr(mcp_logging, log_level.upper(), mcp_logging.INFO))

    # Handle launch mode
    if launch:
        console.print(
            Panel(
                "Launching KiCad bridge and MCP server...\n\n"
                "This will start:\n"
                "  1. KiCad bridge (Python 3.9 with pcbnew)\n"
                "  2. MCP server (Python 3.10+ with FastMCP)",
                border_style="cyan",
            )
        )
        try:
            from .mcp.launcher import main as launcher_main
            launcher_main()
        except ImportError as e:
            console.print(
                Panel(
                    f"[red]Failed to import launcher[/red]: {e}",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
        return

    mode = "IPC" if ipc else "direct"
    console.print(
        Panel(
            f"Starting MCP server (transport={transport}, mode={mode}, log_level={log_level})",
            border_style="cyan",
        )
    )

    if ipc:
        console.print(
            "[yellow]IPC mode: Make sure the KiCad bridge is running![/yellow]\n"
            "Start the bridge with: atoplace-mcp-bridge"
        )

    try:
        from .mcp.server import mcp, MCP_AVAILABLE, configure_session

        if not MCP_AVAILABLE:
            console.print(
                Panel(
                    "[red]MCP package not installed[/red]\n\n"
                    "Install with: pip install mcp",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

        # Configure session based on mode
        if ipc:
            configure_session(use_ipc=True, socket_path=socket)

        # Run the MCP server
        mcp.run()

    except ImportError as e:
        console.print(
            Panel(
                f"[red]Failed to import MCP server[/red]: {e}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command()
def interactive(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
):
    """Run interactive constraint session."""
    from .board.kicad_adapter import save_kicad_board
    from .nlp.constraint_parser import ConstraintParser, ModificationHandler
    from .placement.force_directed import ForceDirectedRefiner, RefinementConfig
    from .placement.legalizer import PlacementLegalizer, LegalizerConfig
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile, get_profile_for_layers

    context = _get_context(ctx)
    console = context.console

    console.print(Rule("Load Board", style="cyan"))
    board_obj, pcb_path, _, _ = load_board_from_path(str(board), build, console)
    if board_obj is None or pcb_path is None:
        raise typer.Exit(code=1)

    if dfm:
        try:
            dfm_profile = get_profile(dfm)
        except ValueError as exc:
            from .dfm.profiles import list_profiles

            console.print(
                Panel(
                    f"[red]Invalid DFM profile[/red]: {exc}\n"
                    f"Available: {', '.join(list_profiles())}",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
    else:
        dfm_profile = get_profile_for_layers(board_obj.layer_count)

    parser = ConstraintParser(board_obj)
    modifier = ModificationHandler(board_obj)
    constraints_list = []

    console.print(
        Panel(
            "Enter constraints or modifications. Type 'help' for commands, 'quit' to exit.",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("atoplace").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "help":
            console.print(
                Panel(
                    """
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
                    """,
                    border_style="blue",
                )
            )
            continue

        if user_input.lower() == "list":
            _render_constraint_summary(console, constraints_list)
            continue

        if user_input.lower() == "clear":
            constraints_list = []
            console.print(Panel("Constraints cleared", border_style="yellow"))
            continue

        if user_input.lower() == "apply":
            if not constraints_list:
                console.print(Panel("No constraints to apply", border_style="yellow"))
                continue

            console.print(Rule("Refinement", style="cyan"))
            config = RefinementConfig(
                min_clearance=dfm_profile.min_spacing,
                edge_clearance=dfm_profile.min_trace_to_edge,
                preferred_clearance=dfm_profile.min_spacing * 2,
                lock_placed=True,
            )
            refiner = ForceDirectedRefiner(board_obj, config)
            for c in constraints_list:
                refiner.add_constraint(c)
            with console.status("Running refinement..."):
                result = refiner.refine()
            console.print(
                Panel(
                    f"Refinement complete ({result.iteration} iterations)",
                    border_style="green",
                )
            )

            console.print(Rule("Legalization", style="cyan"))
            # Only compact outline if board has no explicit outline
            should_compact = not board_obj.outline.has_outline
            legalize_config = LegalizerConfig(
                primary_grid=0.5,
                snap_rotation=True,
                align_passives_only=True,
                min_clearance=dfm_profile.min_spacing,
                edge_clearance=dfm_profile.min_trace_to_edge,
                row_spacing=dfm_profile.min_spacing * 1.5,
                compact_outline=should_compact,
            )
            legalizer = PlacementLegalizer(
                board_obj, legalize_config, constraints=constraints_list
            )
            with console.status("Running legalization..."):
                legal_result = legalizer.legalize()
            _render_legalization_summary(console, legal_result)

            scorer = ConfidenceScorer(dfm_profile)
            report_obj = scorer.assess(board_obj)
            _render_validation_summary(console, report_obj)
            continue

        if user_input.lower().startswith("save"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                save_path = Path(parts[1])
            else:
                save_path = pcb_path.with_suffix(".placed.kicad_pcb")
            save_kicad_board(board_obj, save_path)
            console.print(Panel(f"Saved to {save_path}", border_style="green"))
            continue

        if user_input.lower() == "report":
            scorer = ConfidenceScorer(dfm_profile)
            report_obj = scorer.assess(board_obj)
            _render_validation_summary(console, report_obj)
            continue

        parsed, summary = parser.parse_interactive(user_input)
        if parsed:
            constraints_list.extend(parsed)
        if summary and summary != "No constraints found.":
            console.print(Panel(summary, border_style="blue"))
        if not parsed:
            mod = modifier.parse_modification(user_input)
            if mod:
                if modifier.apply_modification(mod):
                    console.print(
                        Panel(
                            f"Applied: {mod['type']} on {mod.get('component', 'components')}",
                            border_style="green",
                        )
                    )
                else:
                    console.print(Panel("Could not apply modification", border_style="red"))
            else:
                console.print(Panel(f"Could not parse: {user_input}", border_style="red"))

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))


def main():
    app()


if __name__ == "__main__":
    main()
