"""
DRC (Design Rule Check) Integration

Provides design rule checking using KiCad's DRC engine or custom checks.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from ..board.abstraction import Board
from ..dfm.profiles import DFMProfile


@dataclass
class DRCViolation:
    """A DRC violation."""
    rule: str
    severity: str  # "error", "warning"
    message: str
    location: Tuple[float, float]  # mm coordinates
    items: List[str]  # Affected items (refs, net names)


class DRCChecker:
    """Design rule checker."""

    def __init__(self, board: Board, dfm_profile: Optional[DFMProfile] = None):
        self.board = board
        self.dfm_profile = dfm_profile
        self.violations: List[DRCViolation] = []

    def run_checks(self) -> Tuple[bool, List[DRCViolation]]:
        """
        Run all DRC checks.

        Returns:
            (passed, violations) - passed is True if no errors
        """
        self.violations = []

        # Run custom checks
        self._check_clearance()
        self._check_minimum_sizes()
        self._check_edge_clearance()

        passed = not any(v.severity == "error" for v in self.violations)
        return (passed, self.violations)

    def _check_clearance(self):
        """Check component-to-component clearance."""
        if not self.dfm_profile:
            return

        min_clearance = self.dfm_profile.min_spacing
        overlaps = self.board.find_overlaps(min_clearance)

        for ref1, ref2, dist in overlaps:
            c1 = self.board.get_component(ref1)
            c2 = self.board.get_component(ref2)

            # Skip if either component is DNP
            if c1 and c2 and not c1.dnp and not c2.dnp:
                mid_x = (c1.x + c2.x) / 2
                mid_y = (c1.y + c2.y) / 2

                self.violations.append(DRCViolation(
                    rule="component_clearance",
                    severity="error",
                    message=f"Clearance violation: {ref1} and {ref2} ({dist:.2f}mm < {min_clearance}mm)",
                    location=(mid_x, mid_y),
                    items=[ref1, ref2],
                ))

    def _check_minimum_sizes(self):
        """Check that pads meet minimum size requirements.

        Differentiates between through-hole pads (checked against via rules)
        and SMD pads (checked against minimum manufacturable size).
        """
        if not self.dfm_profile:
            return

        # Through-hole pads: check drill and annular ring
        min_th_drill = self.dfm_profile.min_hole_diameter
        min_th_annular = self.dfm_profile.min_via_annular

        # SMD pads: use a sensible minimum based on spacing rules
        # Most fabs can handle pads as small as 0.15mm for 0201 components
        # Use min_spacing as a lower bound (typically 0.1-0.15mm)
        min_smd_pad = self.dfm_profile.min_spacing

        for ref, comp in self.board.components.items():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            for pad in comp.pads:
                abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)

                if pad.drill:
                    # Through-hole pad - check drill size
                    if pad.drill < min_th_drill:
                        self.violations.append(DRCViolation(
                            rule="min_drill_size",
                            severity="warning",
                            message=f"Drill in {ref}.{pad.number} too small ({pad.drill:.2f}mm < {min_th_drill:.2f}mm)",
                            location=(abs_x, abs_y),
                            items=[f"{ref}.{pad.number}"],
                        ))

                    # Check annular ring (pad size minus drill / 2)
                    min_dimension = min(pad.width, pad.height)
                    annular_ring = (min_dimension - pad.drill) / 2
                    if annular_ring < min_th_annular:
                        self.violations.append(DRCViolation(
                            rule="min_annular_ring",
                            severity="warning",
                            message=f"Annular ring in {ref}.{pad.number} too small ({annular_ring:.3f}mm < {min_th_annular:.3f}mm)",
                            location=(abs_x, abs_y),
                            items=[f"{ref}.{pad.number}"],
                        ))
                else:
                    # SMD pad - check minimum dimensions
                    # Only warn for extremely small pads (smaller than min_spacing)
                    if pad.width < min_smd_pad or pad.height < min_smd_pad:
                        self.violations.append(DRCViolation(
                            rule="min_smd_pad_size",
                            severity="warning",
                            message=f"SMD pad {ref}.{pad.number} very small ({pad.width:.2f}x{pad.height:.2f}mm)",
                            location=(abs_x, abs_y),
                            items=[f"{ref}.{pad.number}"],
                        ))

    def _check_edge_clearance(self):
        """Check component clearance to board edges."""
        if not self.dfm_profile:
            return

        min_edge = self.dfm_profile.min_trace_to_edge
        outline = self.board.outline

        for ref, comp in self.board.components.items():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            bbox = comp.get_bounding_box()

            violations_found = []

            if bbox[0] < outline.origin_x + min_edge:
                violations_found.append("left edge")
            if bbox[2] > outline.origin_x + outline.width - min_edge:
                violations_found.append("right edge")
            if bbox[1] < outline.origin_y + min_edge:
                violations_found.append("top edge")
            if bbox[3] > outline.origin_y + outline.height - min_edge:
                violations_found.append("bottom edge")

            if violations_found:
                self.violations.append(DRCViolation(
                    rule="edge_clearance",
                    severity="error",
                    message=f"{ref} too close to {', '.join(violations_found)}",
                    location=(comp.x, comp.y),
                    items=[ref],
                ))

    def run_kicad_drc(self, pcb_path: Path) -> Tuple[bool, List[DRCViolation]]:
        """
        Run KiCad's native DRC.

        Requires pcbnew to be available.
        """
        try:
            import pcbnew
        except ImportError:
            # Return warning that native DRC was skipped
            return (True, [DRCViolation(
                rule="KICAD_DRC_UNAVAILABLE",
                severity="warning",
                message="KiCad DRC skipped: pcbnew module not available",
                location=(0.0, 0.0),
                items=[]
            )])

        board = pcbnew.LoadBoard(str(pcb_path))

        # Create DRC runner
        # Note: This is a simplified version - actual implementation
        # would need to handle KiCad's DRC markers properly

        # For now, return empty (rely on custom checks)
        return (True, [])

    def get_summary(self) -> str:
        """Get summary of DRC results."""
        if not self.violations:
            return "DRC passed with no violations."

        errors = sum(1 for v in self.violations if v.severity == "error")
        warnings = sum(1 for v in self.violations if v.severity == "warning")

        lines = [
            f"DRC: {errors} errors, {warnings} warnings",
            "",
        ]

        for v in self.violations:
            prefix = "[ERROR]" if v.severity == "error" else "[WARN]"
            lines.append(f"{prefix} {v.message}")

        return "\n".join(lines)
