"""
AtoPlace Action Plugins for KiCad PCB Editor

Provides three action plugins:
- AtoPlacePlaceAction: Run force-directed placement optimization
- AtoPlaceValidateAction: Validate current placement
- AtoPlaceReportAction: Generate detailed placement report
"""

import os
import sys
from pathlib import Path

import pcbnew
import wx


def get_atoplace_path():
    """Get the path to the atoplace package.

    The plugin expects atoplace to be installed or available in the Python path.
    If not found, returns None.
    """
    # Check if atoplace is already importable
    try:
        import atoplace
        return Path(atoplace.__file__).parent.parent
    except ImportError:
        pass

    # Check relative to this plugin (development setup)
    plugin_dir = Path(__file__).parent
    project_root = plugin_dir.parent

    if (project_root / "atoplace" / "__init__.py").exists():
        return project_root

    return None


def ensure_atoplace_available():
    """Ensure atoplace package is available for import."""
    atoplace_path = get_atoplace_path()

    if atoplace_path and str(atoplace_path) not in sys.path:
        sys.path.insert(0, str(atoplace_path))

    # Verify import works
    try:
        import atoplace
        return True
    except ImportError as e:
        wx.MessageBox(
            f"AtoPlace package not found.\n\n"
            f"Please install atoplace:\n"
            f"  pip install -e /path/to/atoplace\n\n"
            f"Error: {e}",
            "AtoPlace Error",
            wx.OK | wx.ICON_ERROR
        )
        return False


class AtoPlaceBaseAction(pcbnew.ActionPlugin):
    """Base class for AtoPlace action plugins."""

    def defaults(self):
        """Set default plugin metadata."""
        self.name = "AtoPlace"
        self.category = "Placement"
        self.description = "AI-powered PCB placement optimization"
        self.show_toolbar_button = False

        # Icon path (relative to plugin directory)
        plugin_dir = Path(__file__).parent
        icon_path = plugin_dir / "icon.png"
        if icon_path.exists():
            self.icon_file_name = str(icon_path)

    def get_board_wrapper(self):
        """Get the current board wrapped in atoplace's Board abstraction."""
        if not ensure_atoplace_available():
            return None

        from atoplace.board.abstraction import Board
        from atoplace.board.kicad_adapter import load_kicad_board

        # Get the current board from pcbnew
        kicad_board = pcbnew.GetBoard()
        if not kicad_board:
            wx.MessageBox(
                "No board is currently open.",
                "AtoPlace Error",
                wx.OK | wx.ICON_ERROR
            )
            return None

        # Get board file path
        board_path = kicad_board.GetFileName()
        if not board_path:
            wx.MessageBox(
                "Board has not been saved. Please save the board first.",
                "AtoPlace Error",
                wx.OK | wx.ICON_ERROR
            )
            return None

        # Load through atoplace's abstraction
        try:
            board = load_kicad_board(Path(board_path))
            return board
        except Exception as e:
            wx.MessageBox(
                f"Failed to load board:\n{e}",
                "AtoPlace Error",
                wx.OK | wx.ICON_ERROR
            )
            return None

    def get_dfm_profile(self, board):
        """Get appropriate DFM profile for the board."""
        from atoplace.dfm.profiles import get_profile_for_layers
        return get_profile_for_layers(board.layer_count)


class AtoPlacePlaceAction(AtoPlaceBaseAction):
    """Action plugin to run placement optimization."""

    def defaults(self):
        super().defaults()
        self.name = "AtoPlace: Optimize Placement"
        self.description = "Run force-directed placement optimization with legalization"

    def Run(self):
        """Execute placement optimization."""
        if not ensure_atoplace_available():
            return

        board = self.get_board_wrapper()
        if not board:
            return

        # Show progress dialog
        progress = wx.ProgressDialog(
            "AtoPlace",
            "Optimizing placement...",
            maximum=100,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
        )

        try:
            from atoplace.placement.force_directed import (
                ForceDirectedRefiner,
                RefinementConfig,
            )
            from atoplace.placement.legalizer import (
                PlacementLegalizer,
                LegalizerConfig,
            )

            dfm_profile = self.get_dfm_profile(board)

            # Configure refinement
            config = RefinementConfig(
                min_clearance=dfm_profile.min_spacing,
                preferred_clearance=dfm_profile.min_spacing * 1.5,
                lock_placed=True,
            )

            progress.Update(10, "Running force-directed refinement...")

            # Run force-directed refinement
            refiner = ForceDirectedRefiner(board, config)
            result = refiner.refine()

            if not progress.Update(50, "Running legalization...")[0]:
                return  # User cancelled

            # Run legalization
            legal_config = LegalizerConfig(
                min_clearance=dfm_profile.min_spacing,
            )
            legalizer = PlacementLegalizer(board, legal_config)
            legal_result = legalizer.legalize()

            progress.Update(80, "Saving changes...")

            # Save the board
            from atoplace.board.kicad_adapter import save_kicad_board
            kicad_board = pcbnew.GetBoard()
            board_path = Path(kicad_board.GetFileName())
            save_kicad_board(board, board_path)

            # Refresh the view
            pcbnew.Refresh()

            progress.Update(100, "Complete!")

            # Show results
            wx.MessageBox(
                f"Placement optimization complete!\n\n"
                f"Force-directed refinement:\n"
                f"  - Iterations: {result.iterations}\n"
                f"  - Final energy: {result.final_energy:.2f}\n\n"
                f"Legalization:\n"
                f"  - Components snapped to grid: {legal_result.grid_snapped}\n"
                f"  - Rows formed: {legal_result.rows_formed}\n"
                f"  - Overlaps resolved: {legal_result.overlaps_resolved}\n"
                f"  - Remaining overlaps: {legal_result.final_overlaps}",
                "AtoPlace Complete",
                wx.OK | wx.ICON_INFORMATION
            )

        except Exception as e:
            wx.MessageBox(
                f"Placement optimization failed:\n{e}",
                "AtoPlace Error",
                wx.OK | wx.ICON_ERROR
            )
        finally:
            progress.Destroy()


class AtoPlaceValidateAction(AtoPlaceBaseAction):
    """Action plugin to validate current placement."""

    def defaults(self):
        super().defaults()
        self.name = "AtoPlace: Validate Placement"
        self.description = "Run pre-route validation and DRC checks"

    def Run(self):
        """Execute placement validation."""
        if not ensure_atoplace_available():
            return

        board = self.get_board_wrapper()
        if not board:
            return

        # Show progress dialog
        progress = wx.ProgressDialog(
            "AtoPlace Validation",
            "Running validation checks...",
            maximum=100,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )

        try:
            from atoplace.validation.pre_route import PreRouteValidator
            from atoplace.validation.drc import DRCChecker
            from atoplace.validation.confidence import ConfidenceScorer

            dfm_profile = self.get_dfm_profile(board)

            progress.Update(20, "Running pre-route validation...")

            # Pre-route validation
            pre_route = PreRouteValidator(board, dfm_profile)
            pre_route_ok, pre_route_issues = pre_route.validate()

            progress.Update(50, "Running DRC checks...")

            # DRC checks
            drc = DRCChecker(board, dfm_profile)
            drc_ok, drc_violations = drc.run_checks()

            progress.Update(80, "Calculating confidence score...")

            # Confidence scoring
            scorer = ConfidenceScorer(board, dfm_profile)
            report = scorer.assess()

            progress.Update(100, "Complete!")

            # Build result message
            status_icon = wx.ICON_INFORMATION
            status_text = "PASSED"

            if not pre_route_ok or not drc_ok or report.overall_score < 0.7:
                status_icon = wx.ICON_WARNING
                status_text = "ISSUES FOUND"

            pre_route_summary = f"{len([i for i in pre_route_issues if i.severity == 'error'])} errors, " \
                               f"{len([i for i in pre_route_issues if i.severity == 'warning'])} warnings"

            drc_summary = f"{len([v for v in drc_violations if v.severity == 'error'])} errors, " \
                         f"{len([v for v in drc_violations if v.severity == 'warning'])} warnings"

            wx.MessageBox(
                f"Validation {status_text}\n\n"
                f"Confidence Score: {report.overall_score:.1%}\n\n"
                f"Pre-Route Validation: {pre_route_summary}\n"
                f"DRC Checks: {drc_summary}\n\n"
                f"Use 'AtoPlace: Generate Report' for detailed results.",
                "AtoPlace Validation",
                wx.OK | status_icon
            )

        except Exception as e:
            wx.MessageBox(
                f"Validation failed:\n{e}",
                "AtoPlace Error",
                wx.OK | wx.ICON_ERROR
            )
        finally:
            progress.Destroy()


class AtoPlaceReportAction(AtoPlaceBaseAction):
    """Action plugin to generate detailed placement report."""

    def defaults(self):
        super().defaults()
        self.name = "AtoPlace: Generate Report"
        self.description = "Generate a detailed placement quality report"

    def Run(self):
        """Generate and display placement report."""
        if not ensure_atoplace_available():
            return

        board = self.get_board_wrapper()
        if not board:
            return

        # Ask where to save the report
        kicad_board = pcbnew.GetBoard()
        board_path = Path(kicad_board.GetFileName())
        default_report_path = board_path.with_suffix(".placement_report.md")

        with wx.FileDialog(
            None,
            "Save Placement Report",
            defaultDir=str(board_path.parent),
            defaultFile=default_report_path.name,
            wildcard="Markdown files (*.md)|*.md|All files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            report_path = Path(dlg.GetPath())

        # Show progress dialog
        progress = wx.ProgressDialog(
            "AtoPlace Report",
            "Generating report...",
            maximum=100,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )

        try:
            from atoplace.validation.pre_route import PreRouteValidator
            from atoplace.validation.drc import DRCChecker
            from atoplace.validation.confidence import ConfidenceScorer
            from atoplace.placement.module_detector import ModuleDetector

            dfm_profile = self.get_dfm_profile(board)

            progress.Update(15, "Running pre-route validation...")
            pre_route = PreRouteValidator(board, dfm_profile)
            pre_route_ok, pre_route_issues = pre_route.validate()

            progress.Update(35, "Running DRC checks...")
            drc = DRCChecker(board, dfm_profile)
            drc_ok, drc_violations = drc.run_checks()

            progress.Update(55, "Calculating confidence score...")
            scorer = ConfidenceScorer(board, dfm_profile)
            report = scorer.assess()

            progress.Update(75, "Detecting modules...")
            detector = ModuleDetector(board)
            modules = detector.detect_modules()

            progress.Update(90, "Writing report...")

            # Generate report markdown
            md_content = self._generate_report_markdown(
                board, dfm_profile, pre_route_issues, drc_violations,
                report, modules, pre_route_ok, drc_ok
            )

            # Write report
            report_path.write_text(md_content)

            progress.Update(100, "Complete!")

            wx.MessageBox(
                f"Report saved to:\n{report_path}\n\n"
                f"Confidence Score: {report.overall_score:.1%}",
                "AtoPlace Report",
                wx.OK | wx.ICON_INFORMATION
            )

        except Exception as e:
            wx.MessageBox(
                f"Report generation failed:\n{e}",
                "AtoPlace Error",
                wx.OK | wx.ICON_ERROR
            )
        finally:
            progress.Destroy()

    def _generate_report_markdown(self, board, dfm_profile, pre_route_issues,
                                   drc_violations, confidence_report, modules,
                                   pre_route_ok, drc_ok):
        """Generate the full report markdown content."""
        lines = [
            f"# AtoPlace Placement Report",
            f"",
            f"**Board:** {board.name}",
            f"**DFM Profile:** {dfm_profile.name}",
            f"**Components:** {len(board.components)}",
            f"**Nets:** {len(board.nets)}",
            f"",
            f"---",
            f"",
            f"## Summary",
            f"",
            f"| Check | Status |",
            f"|-------|--------|",
            f"| Pre-Route Validation | {'PASS' if pre_route_ok else 'FAIL'} |",
            f"| DRC Checks | {'PASS' if drc_ok else 'FAIL'} |",
            f"| Confidence Score | {confidence_report.overall_score:.1%} |",
            f"",
            f"---",
            f"",
            f"## Pre-Route Validation",
            f"",
        ]

        if pre_route_issues:
            for issue in pre_route_issues:
                prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}
                lines.append(f"- **[{prefix.get(issue.severity, 'INFO')}]** [{issue.category}] {issue.message}")
        else:
            lines.append("No issues found.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## DRC Violations",
            f"",
        ])

        if drc_violations:
            for v in drc_violations:
                prefix = "ERROR" if v.severity == "error" else "WARN"
                lines.append(f"- **[{prefix}]** {v.message}")
        else:
            lines.append("No violations found.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Confidence Report",
            f"",
            confidence_report.to_markdown(),
            f"",
            f"---",
            f"",
            f"## Detected Modules",
            f"",
        ])

        if modules:
            for module in modules:
                comp_list = ", ".join(sorted(module.components)[:5])
                if len(module.components) > 5:
                    comp_list += f", ... ({len(module.components)} total)"
                lines.append(f"- **{module.module_type.value}**: {comp_list}")
        else:
            lines.append("No functional modules detected.")

        lines.extend([
            f"",
            f"---",
            f"",
            f"*Generated by AtoPlace*",
        ])

        return "\n".join(lines)
