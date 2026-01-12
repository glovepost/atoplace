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
        """Check component-to-component clearance.

        Only checks clearance between components on the same layer (top vs bottom).
        Components on opposite sides of the board are allowed to overlap.

        Uses the effective clearance which is the maximum of:
        - Board's design rule default clearance (if available)
        - DFM profile minimum spacing
        """
        if not self.dfm_profile:
            return

        # Determine effective minimum clearance
        # Board's design rules take precedence if they're stricter
        dfm_clearance = self.dfm_profile.min_spacing
        board_clearance = self.board.default_clearance or 0.0

        min_clearance = max(dfm_clearance, board_clearance)
        clearance_source = "board" if board_clearance > dfm_clearance else "DFM"

        # Use layer-aware overlap detection with pad extents for accuracy
        # - check_layers: avoid false positives for components on opposite sides
        # - include_pads: catch overlaps where pads protrude beyond body
        overlaps = self.board.find_overlaps(min_clearance, check_layers=True, include_pads=True)

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
                    message=f"Clearance violation: {ref1} and {ref2} ({dist:.2f}mm < {min_clearance}mm [{clearance_source}])",
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

        # SMD pads: use an absolute minimum for truly unrealistic pads
        # Most fabs can handle 0201 pads (~0.15mm x 0.3mm), so we use 0.05mm
        # as the floor to catch only obviously erroneous geometry.
        # Note: This is intentionally NOT tied to min_spacing, as spacing rules
        # govern clearance between features, not minimum pad dimensions.
        min_smd_pad = 0.05  # 50 microns - below any realistic SMD pad

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
        """Check component clearance to board edges.

        Uses BoardOutline.contains_point() which properly handles:
        - Polygon outlines (non-rectangular boards)
        - Board cutouts/holes
        - Margin enforcement for edge clearance
        """
        if not self.dfm_profile:
            return

        min_edge = self.dfm_profile.min_trace_to_edge
        outline = self.board.outline

        for ref, comp in self.board.components.items():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            bbox = comp.get_bounding_box()

            # Check all 4 corners of bounding box against board outline with margin
            corners = [
                (bbox[0], bbox[1]),  # min_x, min_y (bottom-left)
                (bbox[2], bbox[1]),  # max_x, min_y (bottom-right)
                (bbox[0], bbox[3]),  # min_x, max_y (top-left)
                (bbox[2], bbox[3]),  # max_x, max_y (top-right)
            ]

            violating_corners = 0
            for cx, cy in corners:
                if not outline.contains_point(cx, cy, min_edge):
                    violating_corners += 1

            if violating_corners > 0:
                self.violations.append(DRCViolation(
                    rule="edge_clearance",
                    severity="error",
                    message=f"{ref} too close to board edge or cutout ({violating_corners} corners violate {min_edge:.2f}mm clearance)",
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
