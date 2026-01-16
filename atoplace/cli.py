#!/usr/bin/env python3
"""
AtoPlace CLI

Command-line interface for the AI-powered PCB placement and routing tool.
"""

from __future__ import annotations

import atexit
import io
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
import math
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

    # Register cleanup handler to close file on exit
    def _cleanup_log_file():
        global _LOG_FILE_HANDLE
        if _LOG_FILE_HANDLE is not None:
            try:
                _LOG_FILE_HANDLE.close()
            except Exception:
                pass  # Best effort cleanup, don't raise during shutdown
            _LOG_FILE_HANDLE = None

    atexit.register(_cleanup_log_file)

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


def _get_version() -> str:
    """Get package version from metadata."""
    try:
        from importlib.metadata import version
        return version("atoplace")
    except Exception:
        return "0.1.0"  # Fallback for development


def _version_callback(value: Optional[bool]) -> None:
    if value:
        typer.echo(f"atoplace {_get_version()}")
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
        True,
        "--use-ato-modules/--no-use-ato-modules",
        help="Use atopile module grouping and placement hints.",
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
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Enable real-time streaming visualization via WebSocket.",
    ),
    stream_port: int = typer.Option(
        8765,
        "--stream-port",
        help="WebSocket port for streaming visualization (default: 8765).",
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
    import asyncio
    from .board.kicad_adapter import save_kicad_board
    from .placement.force_directed import ForceDirectedRefiner, RefinementConfig
    from .placement.legalizer import PlacementLegalizer, LegalizerConfig
    from .placement.module_detector import ModuleDetector
    from .placement.visualizer import PlacementVisualizer
    from .nlp.constraint_parser import ConstraintParser
    from .validation.confidence import ConfidenceScorer
    from .dfm.profiles import get_profile, get_profile_for_layers

    # Import streaming visualizer if needed
    if stream:
        try:
            from .placement.streaming_visualizer import StreamingVisualizer
        except ImportError:
            console = _get_context(ctx).console
            console.print(
                Panel(
                    "[red]Streaming requires websockets package[/red]\n\n"
                    "Install with: pip install 'atoplace[streaming]'",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

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
    streaming_viz = None
    module_map = {}  # ref -> module_type
    if visualize or stream:
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

        if stream:
            # Use streaming visualizer for real-time WebSocket updates
            streaming_viz = StreamingVisualizer(
                board_obj,
                host="localhost",
                port=stream_port,
                max_fps=10.0,
            )
            visualizer = streaming_viz.visualizer  # Use underlying visualizer for captures
            console.print(
                Panel(
                    f"[cyan]Streaming enabled[/cyan]\n\n"
                    f"WebSocket: ws://localhost:{stream_port}\n"
                    f"Open placement_debug/stream_viewer.html in your browser",
                    border_style="cyan",
                )
            )
        else:
            visualizer = PlacementVisualizer(board_obj)

    constraints_list = []
    if constraints:
        console.print(Rule("Constraints", style="cyan"))
        parser = ConstraintParser(board_obj)
        parsed, summary = parser.parse_interactive(constraints)
        constraints_list = parsed
        console.print(Panel(summary, border_style="blue"))
        _render_constraint_summary(console, constraints_list)

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

    # Build module-derived constraints (atopile groupings + heuristic hints)
    def _estimate_module_spread(component_refs: List[str]) -> float:
        """Estimate a reasonable grouping radius based on component areas and clearance."""
        total_area = 0.0
        for ref in component_refs:
            comp = board_obj.components.get(ref)
            if not comp:
                continue
            bbox = comp.get_bounding_box_with_pads()
            width = max(0.1, bbox[2] - bbox[0])
            height = max(0.1, bbox[3] - bbox[1])
            total_area += width * height + (dfm_profile.min_spacing * 4)
        if total_area <= 0:
            return 15.0
        radius = math.sqrt(total_area / math.pi)
        return max(10.0, radius * 1.6)

    def _nearest_edge(comp_ref: str) -> str:
        """Pick the nearest board edge for a component."""
        outline = board_obj.outline
        comp = board_obj.components.get(comp_ref)
        if not comp or not outline.has_outline:
            return "left"
        distances = {
            "left": abs((comp.x - outline.origin_x)),
            "right": abs((outline.origin_x + outline.width) - comp.x),
            "top": abs((comp.y - outline.origin_y)),
            "bottom": abs((outline.origin_y + outline.height) - comp.y),
        }
        return min(distances, key=distances.get)

    def _collect_atopile_groupings() -> List[object]:
        """Create grouping + separation constraints from atopile module metadata."""
        from .placement.constraints import GroupingConstraint, SeparationConstraint

        constraints: List[object] = []
        modules_to_components: Dict[str, List[str]] = {}
        for ref, comp in board_obj.components.items():
            ato_module = comp.properties.get("ato_module")
            if ato_module:
                modules_to_components.setdefault(ato_module, []).append(ref)

        if not modules_to_components:
            return constraints

        table = Table(box=box.SIMPLE, header_style="bold")
        table.add_column("Module")
        table.add_column("Components", justify="right")
        module_spread: Dict[str, float] = {}

        for module_name, comp_refs in modules_to_components.items():
            if len(comp_refs) < 2:
                continue
            spread = _estimate_module_spread(comp_refs)
            module_spread[module_name] = spread
            constraint = GroupingConstraint(
                components=comp_refs,
                max_spread=spread,
                optimize_bbox=True,
                bbox_strength=1.1,
                min_clearance=dfm_profile.min_spacing,
                description=f"Group atopile module: {module_name}",
            )
            constraints.append(constraint)
            table.add_row(module_name, str(len(comp_refs)))

        # Optional separation between top-level modules (bounded to avoid N^2 explosions)
        module_names = list(module_spread.keys())
        if 2 <= len(module_names) <= 8:
            roots: Dict[str, List[str]] = {}
            for name in module_names:
                root = name.split(".")[0]
                roots.setdefault(root, []).append(name)

            processed_pairs = set()
            for names in roots.values():
                for i, left in enumerate(names):
                    for right in names[i + 1:]:
                        pair_key = tuple(sorted((left, right)))
                        if pair_key in processed_pairs:
                            continue
                        processed_pairs.add(pair_key)
                        spread_a = module_spread[left]
                        spread_b = module_spread[right]
                        refs_a = modules_to_components.get(left, [])
                        refs_b = modules_to_components.get(right, [])
                        if not refs_a or not refs_b:
                            continue
                        constraints.append(
                            SeparationConstraint(
                                group_a=refs_a,
                                group_b=refs_b,
                                min_separation=max(spread_a, spread_b) * 0.5,
                                description=f"Separate modules {left} and {right}",
                            )
                        )

        if table.rows:
            console.print(Rule("Atopile Module Grouping", style="cyan"))
            console.print(table)

        return constraints

    def _build_hint_constraints() -> List[object]:
        """Translate ModuleDetector placement hints into constraints."""
        from .placement.constraints import (
            EdgeConstraint,
            ProximityConstraint,
        )
        hint_constraints: List[object] = []

        # Identify anchor modules/components for hints
        mcu_refs = [
            m.primary_component
            for m in modules
            if m.module_type.name.lower() == "microcontroller" and m.primary_component
        ]
        connector_refs = [
            m.primary_component
            for m in modules
            if m.module_type.name.lower() == "connector" and m.primary_component
        ]

        for module in modules:
            if not module.placement_hints:
                continue
            primary_ref = module.primary_component
            if not primary_ref:
                continue

            edge_hint = module.placement_hints.get("edge_placement")
            if edge_hint:
                edge = _nearest_edge(primary_ref)
                hint_constraints.append(
                    EdgeConstraint(
                        component_ref=primary_ref,
                        edge=edge,
                        offset=max(dfm_profile.min_trace_to_edge, dfm_profile.min_spacing * 1.5),
                        priority="required" if edge_hint == "required" else "preferred",
                        description=f"Place {primary_ref} near {edge} edge",
                    )
                )

            if module.placement_hints.get("close_to_mcu") and mcu_refs:
                target = mcu_refs[0]
                hint_constraints.append(
                    ProximityConstraint(
                        target_ref=primary_ref,
                        anchor_ref=target,
                        ideal_distance=5.0,
                        max_distance=15.0,
                        priority="required",
                        description=f"Keep {primary_ref} close to MCU {target}",
                    )
                )

            if module.placement_hints.get("near_input") and connector_refs:
                target = connector_refs[0]
                hint_constraints.append(
                    ProximityConstraint(
                        target_ref=primary_ref,
                        anchor_ref=target,
                        ideal_distance=8.0,
                        max_distance=20.0,
                        priority="preferred",
                        description=f"Keep {primary_ref} near connector {target}",
                    )
                )

        return hint_constraints

    module_constraints: List[object] = []
    if is_atopile and use_ato_modules:
        module_constraints.extend(_collect_atopile_groupings())
    module_constraints.extend(_build_hint_constraints())

    if module_constraints:
        constraints_list.extend(module_constraints)
        _render_constraint_summary(console, module_constraints)

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

    # Streaming refinement - run server in background thread
    if stream and streaming_viz:
        import threading
        import queue

        # Queue for frames to stream
        frame_queue = queue.Queue(maxsize=100)
        stop_event = threading.Event()
        server_ready = threading.Event()

        async def streaming_server_loop():
            """Run WebSocket server and stream frames from queue."""
            try:
                await streaming_viz.start_streaming(generate_viewer=True)
                await streaming_viz.send_status("info", "Starting placement optimization")
                server_ready.set()

                while not stop_event.is_set():
                    try:
                        # Get frame from queue with timeout
                        frame_data = frame_queue.get(timeout=0.1)
                        if frame_data is None:  # Poison pill
                            break
                        await streaming_viz.server.broadcast_frame(frame_data)
                    except queue.Empty:
                        await asyncio.sleep(0.05)

                await streaming_viz.send_status("complete", "Optimization complete")
                await asyncio.sleep(0.3)  # Brief pause for final message

            finally:
                await streaming_viz.stop_streaming()

        def run_server_thread():
            """Run async server in thread."""
            asyncio.run(streaming_server_loop())

        # Start server thread
        server_thread = threading.Thread(target=run_server_thread, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        console.print("Starting streaming server...")
        server_ready.wait(timeout=5.0)

        def streaming_callback(state):
            # Build frame data
            if state.iteration % 5 == 0:  # Stream every 5th iteration
                max_vel = 0.0
                for vx, vy in state.velocities.values():
                    vel = math.sqrt(vx * vx + vy * vy)
                    max_vel = max(max_vel, vel)

                frame_data = {
                    "index": state.iteration,
                    "label": f"Iteration {state.iteration}",
                    "iteration": state.iteration,
                    "phase": "refinement",
                    "components": {
                        ref: [comp.x, comp.y, comp.rotation]
                        for ref, comp in board_obj.components.items()
                    },
                    "modules": module_map,
                    "forces": {},
                    "energy": state.total_energy,
                    "max_move": max_vel,
                    "overlap_count": 0,
                    "total_wire_length": 0.0,
                }
                try:
                    frame_queue.put_nowait(frame_data)
                except queue.Full:
                    pass  # Skip frame if queue full

        # Run refinement with streaming callback
        console.print("Running refinement with streaming... (open browser to view)")
        result = refiner.refine(callback=streaming_callback)

        # Signal server to stop and wait
        stop_event.set()
        frame_queue.put(None)  # Poison pill
        server_thread.join(timeout=3.0)

    elif context.verbose:
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

    # Export visualization if enabled (using SVG delta for best quality/performance)
    if visualizer:
        viz_path = visualizer.export_svg_delta_html(
            filename="placement_debug.html",
            output_dir="placement_debug",
        )
        console.print(Panel(f"Visualization: {viz_path}", border_style="magenta"))

        if stream:
            console.print(
                Panel(
                    "Streaming visualization was available during optimization.\n"
                    "Static visualization also exported for later review.",
                    border_style="cyan",
                )
            )

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
def fanout(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    component: Optional[str] = typer.Option(
        None, "-c", "--component", help="Specific component to fanout (e.g., U1). If not specified, auto-detects BGAs."
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file (default: overwrites input)."
    ),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
    strategy: str = typer.Option(
        "auto", "--strategy", "-s",
        help="Fanout strategy: 'auto' (detect from pitch), 'dogbone', or 'vip'."
    ),
    no_escape: bool = typer.Option(
        False, "--no-escape", help="Skip escape routing (only generate vias and pad-to-via traces)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Don't save fanout to file."
    ),
):
    """Generate BGA/FPGA fanout patterns.

    Automatically generates escape routing for high-density BGA packages:
    - Detects BGA components on the board
    - Creates dogbone or via-in-pad patterns based on pitch
    - Assigns escape layers using the onion model
    - Routes escape traces to clear routing space

    Example:
        atoplace fanout board.kicad_pcb --component U1
        atoplace fanout board.kicad_pcb --strategy dogbone
    """
    from .routing.fanout import FanoutGenerator, FanoutStrategy
    from .dfm.profiles import get_profile, get_profile_for_layers

    context: CLIContext = ctx.obj
    console = context.console

    console.print(Rule("BGA Fanout"))

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

    # Parse strategy
    strategy_map = {
        "auto": FanoutStrategy.AUTO,
        "dogbone": FanoutStrategy.DOGBONE,
        "vip": FanoutStrategy.VIP,
    }
    if strategy.lower() not in strategy_map:
        console.print(f"[red]Invalid strategy:[/] {strategy}")
        console.print(f"Available: {', '.join(strategy_map.keys())}")
        raise typer.Exit(code=1)
    fanout_strategy = strategy_map[strategy.lower()]

    # Create generator
    generator = FanoutGenerator(board_obj, dfm_profile)

    # Detect or use specified component
    if component:
        if component not in board_obj.components:
            console.print(f"[red]Component not found:[/] {component}")
            raise typer.Exit(code=1)
        components_to_fanout = [component]
    else:
        components_to_fanout = generator.detect_bgas()
        if not components_to_fanout:
            console.print("[yellow]No BGA components detected[/]")
            raise typer.Exit(code=0)

    console.print(f"Components to fanout: [cyan]{', '.join(components_to_fanout)}[/]")

    # Generate fanout
    all_results = {}
    total_vias = 0
    total_traces = 0
    total_warnings = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating fanout...", total=len(components_to_fanout))

        for ref in components_to_fanout:
            result = generator.fanout_component(
                ref,
                strategy=fanout_strategy,
                include_escape=not no_escape,
            )
            all_results[ref] = result

            if result.success:
                total_vias += len(result.vias)
                total_traces += len(result.traces)
                total_warnings.extend(result.warnings)

            progress.update(task, advance=1)

    # Display results
    console.print()
    table = Table(title="Fanout Results", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Strategy")
    table.add_column("Pitch", justify="right")
    table.add_column("Rings", justify="right")
    table.add_column("Vias", justify="right")
    table.add_column("Traces", justify="right")
    table.add_column("Status")

    for ref, result in all_results.items():
        if result.success:
            status = "[green]OK[/]"
        else:
            status = f"[red]FAIL[/] ({result.failure_reason})"

        table.add_row(
            ref,
            result.strategy_used.value if result.success else "-",
            f"{result.pitch_detected:.2f}mm" if result.success else "-",
            str(result.ring_count) if result.success else "-",
            str(len(result.vias)) if result.success else "-",
            str(len(result.traces)) if result.success else "-",
            status,
        )

    console.print(table)

    # Summary
    success_count = sum(1 for r in all_results.values() if r.success)
    console.print(f"\nSuccess: [cyan]{success_count}/{len(all_results)}[/]")
    console.print(f"Total vias: [cyan]{total_vias}[/]")
    console.print(f"Total traces: [cyan]{total_traces}[/]")

    # Warnings
    if total_warnings:
        console.print(f"\n[yellow]Warnings ({len(total_warnings)}):[/]")
        for warn in total_warnings[:10]:
            console.print(f"  - {warn}")
        if len(total_warnings) > 10:
            console.print(f"  ... and {len(total_warnings) - 10} more")

    # Save fanout
    if not dry_run and success_count > 0:
        # Collect all vias and traces
        all_vias = []
        all_traces = []
        for result in all_results.values():
            if result.success:
                all_vias.extend(result.vias)
                all_traces.extend(result.traces)

        # For now, we'll need to extend the KiCad adapter to save fanout
        # This is a placeholder - actual implementation depends on KiCad API
        console.print(f"\n[yellow]Note:[/] Fanout visualization generated. "
                     f"KiCad file saving for fanout vias/traces not yet implemented.")
        console.print(f"Generated {len(all_vias)} vias and {len(all_traces)} traces for review.")

    elif dry_run:
        console.print("[yellow]Dry run - fanout not saved[/]")

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))

    raise typer.Exit(code=0 if success_count > 0 else 1)


@app.command()
def pinswap(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    component: Optional[str] = typer.Option(
        None, "-c", "--component", help="Specific component to optimize (e.g., U1). If not specified, optimizes all."
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file for constraint updates."
    ),
    format: str = typer.Option(
        "xdc", "--format", "-f",
        help="Constraint output format: 'xdc' (Xilinx), 'qsf' (Intel), 'tcl', 'csv', 'json'."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
    min_improvement: float = typer.Option(
        5.0, "--min-improvement",
        help="Minimum improvement percentage to apply swaps. Default: 5.0"
    ),
    analyze_only: bool = typer.Option(
        False, "--analyze", "-a",
        help="Only analyze swap potential, don't apply changes."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Don't save constraint file or modify board."
    ),
):
    """Optimize pin assignments to reduce routing complexity.

    Detects swappable pin groups on FPGAs, MCUs, and connectors, then uses
    bipartite matching to find optimal pin-to-net assignments that minimize
    wire crossings and total wire length.

    This is Phase 0 of the routing pipeline - run before actual routing.

    Examples:
        atoplace pinswap board.kicad_pcb --analyze
        atoplace pinswap board.kicad_pcb -c U1 --format xdc
        atoplace pinswap board.kicad_pcb -o constraints.xdc
    """
    from .routing.pinswapper import PinSwapper, SwapConfig, ConstraintFormat

    context: CLIContext = ctx.obj
    console = context.console

    console.print(Rule("Pin Swap Optimization"))

    # Load board
    board_obj, pcb_path, _, _ = load_board_from_path(str(board), build, console)
    if board_obj is None:
        raise typer.Exit(code=1)

    # Parse output format
    format_map = {
        "xdc": ConstraintFormat.XDC,
        "qsf": ConstraintFormat.QSF,
        "tcl": ConstraintFormat.TCL,
        "csv": ConstraintFormat.NETLIST,
        "json": ConstraintFormat.JSON,
    }
    if format.lower() not in format_map:
        console.print(f"[red]Invalid format:[/] {format}")
        console.print(f"Available: {', '.join(format_map.keys())}")
        raise typer.Exit(code=1)
    constraint_format = format_map[format.lower()]

    # Configure swapper
    config = SwapConfig(
        min_improvement=min_improvement,
    )
    swapper = PinSwapper(board_obj, config)

    # Analyze or optimize
    if analyze_only or component:
        console.print(Rule("Analysis", style="cyan"))

        if component:
            if component not in board_obj.components:
                console.print(f"[red]Component not found:[/] {component}")
                raise typer.Exit(code=1)
            components_to_analyze = [component]
        else:
            # Get all components with swap groups
            groups = swapper._detector.detect_all()
            components_to_analyze = list(groups.keys())

        if not components_to_analyze:
            console.print("[yellow]No swappable components found[/]")
            raise typer.Exit(code=0)

        for ref in components_to_analyze:
            analysis = swapper.analyze_component(ref)

            if "error" in analysis:
                console.print(f"[red]{ref}:[/] {analysis['error']}")
                continue

            table = Table(title=f"{ref} ({analysis['footprint']})", box=box.ROUNDED)
            table.add_column("Swap Group", style="cyan")
            table.add_column("Type")
            table.add_column("Pins", justify="right")
            table.add_column("Connected", justify="right")
            table.add_column("Potential", justify="right")
            table.add_column("Confidence", justify="right")

            for group in analysis.get("groups", []):
                table.add_row(
                    group["name"],
                    group["type"],
                    str(group["pins"]),
                    str(group["connected_pins"]),
                    group["potential_improvement"],
                    f"{group['confidence']:.0%}",
                )

            console.print(table)
            console.print(f"Current crossings: [cyan]{analysis['current_crossings']}[/]")
            console.print()

        if analyze_only:
            raise typer.Exit(code=0)

    # Run optimization
    console.print(Rule("Optimization", style="cyan"))

    if component:
        results = {component: swapper.optimize_component(component, apply=not dry_run)}
    else:
        results = swapper.optimize_all(apply=not dry_run)

    # Display results
    console.print()
    table = Table(title="Pin Swap Results", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Groups", justify="right")
    table.add_column("Swaps", justify="right")
    table.add_column("Crossing Δ", justify="right")
    table.add_column("Wire Δ", justify="right")
    table.add_column("Status")

    total_swaps = 0
    for ref, result in results.items():
        if result.success:
            total_swaps += result.total_swaps
            crossing_delta = result.original_crossings - result.final_crossings
            crossing_pct = f"{result.crossing_improvement:.1f}%" if result.original_crossings > 0 else "N/A"
            wire_pct = f"{result.wire_improvement:.1f}%" if result.original_wire_length > 0 else "N/A"
            status = "[green]OK[/]"
        else:
            crossing_delta = 0
            crossing_pct = "N/A"
            wire_pct = "N/A"
            status = f"[red]FAIL[/] ({result.failure_reason})"

        table.add_row(
            ref,
            str(result.groups_detected),
            str(result.total_swaps),
            f"-{crossing_delta} ({crossing_pct})" if crossing_delta > 0 else crossing_pct,
            wire_pct,
            status,
        )

    console.print(table)

    # Get crossing analysis
    crossing_result = swapper.get_crossing_analysis()
    console.print(f"\nTotal ratsnest crossings: [cyan]{crossing_result.total_crossings}[/]")
    console.print(f"Crossing density: [cyan]{crossing_result.crossing_density:.2f}[/] crossings/edge")

    # Summary
    console.print(f"\n[bold]Total swaps performed:[/bold] {total_swaps}")

    # Export constraints
    if total_swaps > 0 and not dry_run:
        if output:
            swapper.export_constraints(output, constraint_format)
            console.print(f"[green]Constraints saved to:[/green] {output}")
        else:
            console.print(Rule("Constraint Preview", style="cyan"))
            preview = swapper.get_constraint_preview(constraint_format)
            console.print(preview[:2000])  # Limit preview size
            if len(preview) > 2000:
                console.print(f"... ({len(preview) - 2000} more characters)")

    elif dry_run:
        console.print("[yellow]Dry run - no changes saved[/]")

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))

    raise typer.Exit(code=0 if total_swaps > 0 else 1)


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


@app.command()
def route(
    ctx: typer.Context,
    board: Path = typer.Argument(..., help="KiCad PCB file or atopile project dir."),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output PCB file path."
    ),
    dfm: Optional[str] = typer.Option(
        None, "--dfm", help="DFM profile name (e.g., jlcpcb_standard)."
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Atopile build name (default: default)."
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Generate routing visualization."
    ),
    diff_pair: Optional[List[str]] = typer.Option(
        None, "--diff-pair", "-d",
        help="Diff pair in format NAME:POS_NET:NEG_NET. Can specify multiple."
    ),
    critical_net: Optional[List[str]] = typer.Option(
        None, "--critical", "-c",
        help="Net name to route with high priority. Can specify multiple."
    ),
    detect_diff_pairs: bool = typer.Option(
        False, "--detect-diff-pairs", help="Auto-detect differential pairs from net names."
    ),
    skip_fanout: bool = typer.Option(
        False, "--skip-fanout", help="Skip BGA fanout phase."
    ),
    greedy: float = typer.Option(
        2.0, "--greedy", "-g",
        help="A* greedy multiplier (1=optimal, 2-3=fast). Default: 2.0"
    ),
    grid: float = typer.Option(
        0.1, "--grid", help="Routing grid size in mm. Default: 0.1"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Don't save routed traces to file."
    ),
):
    """Route all nets on a PCB board.

    Uses the multi-phase routing pipeline:
    1. Fanout & Escape: BGA/QFN escape routing
    2. Critical Nets: Differential pairs
    3. General Routing: A* with greedy multiplier

    Routed traces are saved to the output file (or input file if no output specified).

    Examples:
        atoplace route board.kicad_pcb
        atoplace route board.kicad_pcb --visualize
        atoplace route board.kicad_pcb -d USB:USB_D+:USB_D-
        atoplace route board.kicad_pcb --detect-diff-pairs
        atoplace route board.kicad_pcb --greedy 2.5 --grid 0.05
        atoplace route board.kicad_pcb -o routed_board.kicad_pcb --dry-run
    """
    from .routing import RoutingManager, RoutingManagerConfig, DiffPairDetector, RouterConfig
    from .dfm.profiles import get_profile, get_profile_for_layers

    context = _get_context(ctx)
    console = context.console

    console.print(Rule("Load Board", style="cyan"))
    board_obj, pcb_path, _, _ = load_board_from_path(str(board), build, console)
    if board_obj is None or pcb_path is None:
        raise typer.Exit(code=1)

    # Get DFM profile
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

    # Display board summary
    _render_board_summary(console, board_obj, None)

    # Configure router with user-specified parameters
    router_config = RouterConfig(
        greedy_weight=greedy,
        grid_size=grid,
        trace_width=dfm_profile.min_trace_width,
        clearance=dfm_profile.min_spacing,
        layer_count=board_obj.layer_count,
    )

    # Configure routing manager
    config = RoutingManagerConfig(
        visualize=visualize,
        enable_fanout=not skip_fanout,
        output_dir=str(output.parent) if output else ".",
        router_config=router_config,
    )
    manager = RoutingManager(board_obj, dfm_profile, config)

    console.print(f"[cyan]Greedy multiplier:[/cyan] {greedy}")
    console.print(f"[cyan]Grid size:[/cyan] {grid}mm")

    # Auto-detect diff pairs if requested
    if detect_diff_pairs:
        console.print(Rule("Detect Differential Pairs", style="cyan"))
        net_names = list(board_obj.nets.keys())
        detector = DiffPairDetector(net_names)
        detected_pairs = detector.detect()

        if detected_pairs:
            table = Table(title=f"Detected {len(detected_pairs)} Diff Pairs", box=box.SIMPLE)
            table.add_column("Name")
            table.add_column("Positive")
            table.add_column("Negative")
            table.add_column("Pattern")

            for pair in detected_pairs:
                manager.add_diff_pair(pair.name, pair.positive_net, pair.negative_net)
                table.add_row(pair.name, pair.positive_net, pair.negative_net, pair.pattern.value)

            console.print(table)
        else:
            console.print("[yellow]No differential pairs detected[/yellow]")

    # Add manually specified diff pairs
    if diff_pair:
        for dp_str in diff_pair:
            parts = dp_str.split(":")
            if len(parts) >= 3:
                name, pos_net, neg_net = parts[0], parts[1], parts[2]
                manager.add_diff_pair(name, pos_net, neg_net)
                console.print(f"[green]Added diff pair:[/green] {name} ({pos_net}/{neg_net})")
            else:
                console.print(f"[red]Invalid diff pair format:[/red] {dp_str}")

    # Add critical nets
    if critical_net:
        manager.set_critical_nets(list(critical_net))
        console.print(f"[green]Critical nets:[/green] {', '.join(critical_net)}")

    # Run routing with progress display
    console.print(Rule("Routing", style="cyan"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Routing...", total=100)

        def progress_callback(phase: str, pct: float):
            progress.update(task, completed=int(pct * 100), description=f"[cyan]{phase}[/cyan]")

        manager.set_progress_callback(progress_callback)
        result = manager.route_all()

    # Display results
    console.print(Rule("Routing Results", style="cyan"))

    # Summary table
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total Nets", str(result.total_nets))
    table.add_row("Routed", f"[green]{result.routed_nets}[/green]")
    table.add_row("Failed", f"[red]{result.failed_nets}[/red]" if result.failed_nets > 0 else "0")
    table.add_row("Completion", f"{result.completion_rate:.1f}%")
    table.add_row("Total Length", f"{result.total_length:.1f} mm")
    table.add_row("Total Vias", str(result.total_vias))
    table.add_row("Phases", ", ".join(p.value for p in result.phases_completed))

    console.print(Panel(table, title="Summary", border_style="cyan"))

    # Show failed nets if any
    if result.failed_nets > 0:
        failed_table = Table(title="Failed Nets", box=box.SIMPLE)
        failed_table.add_column("Net")
        failed_table.add_column("Reason")

        count = 0
        for net_name, net_result in result.net_results.items():
            if not net_result.success and count < 20:
                failed_table.add_row(net_name, net_result.failure_reason or "Unknown")
                count += 1

        if count == 20:
            failed_table.add_row("...", f"({result.failed_nets - 20} more)")

        console.print(failed_table)

    # Show errors and warnings
    if result.errors:
        for error in result.errors:
            console.print(f"[red]Error:[/red] {error}")

    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

    # Save routed traces to KiCad file
    if result.routed_nets > 0 and not dry_run:
        from .board.kicad_adapter import save_routed_traces
        import shutil

        # Collect all successful traces and vias from routing results
        all_segments = []
        all_vias = []
        net_id_to_name = {}

        for net_name, net_result in result.net_results.items():
            if net_result.success:
                all_segments.extend(net_result.segments)
                all_vias.extend(net_result.vias)
                # Build net_id to name mapping from segments/vias
                for seg in net_result.segments:
                    if seg.net_id is not None:
                        net_id_to_name[seg.net_id] = net_name
                for via in net_result.vias:
                    if via.net_id is not None:
                        net_id_to_name[via.net_id] = net_name

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
            console.print(
                Panel(
                    "[yellow]Note:[/yellow] Routing completed but trace saving failed. "
                    "Use the visualization output to review routing.",
                    border_style="yellow"
                )
            )
    elif dry_run and result.routed_nets > 0:
        console.print("[yellow]Dry run - traces not saved to file[/yellow]")

    if visualize:
        viz_path = Path(config.output_dir) / "routing_result.html"
        console.print(f"[green]Visualization:[/green] {viz_path}")

    console.print(Panel(f"Debug log: {context.log_path}", border_style="blue"))

    raise typer.Exit(code=0 if result.success else 1)


def main():
    app()


if __name__ == "__main__":
    main()
