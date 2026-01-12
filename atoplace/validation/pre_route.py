"""
Pre-Routing Validation

Validates board state before attempting autorouting to catch issues early.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, FrozenSet
from ..board.abstraction import Board, Component


@dataclass
class PreRouteIssue:
    """An issue found during pre-route validation."""
    severity: str  # "error", "warning", "info"
    category: str  # "connectivity", "placement", "footprint"
    message: str
    location: str  # Component or net reference


class PreRouteValidator:
    """Validates board before routing."""

    def __init__(self, board: Board):
        self.board = board
        self.issues: List[PreRouteIssue] = []

    def validate(self) -> Tuple[bool, List[PreRouteIssue]]:
        """
        Run all pre-route validations.

        Returns:
            (can_proceed, issues) - can_proceed is False if errors found
        """
        self.issues = []

        # Run all checks
        self._check_unconnected_pads()
        self._check_single_pad_nets()
        self._check_missing_footprints()
        self._check_overlapping_pads()
        self._check_power_connections()

        # Determine if we can proceed
        has_errors = any(i.severity == "error" for i in self.issues)

        return (not has_errors, self.issues)

    def _check_unconnected_pads(self):
        """Check for pads with no net assigned."""
        for ref, comp in self.board.components.items():
            for pad in comp.pads:
                if not pad.net:
                    # Some pads (like mounting holes) are expected to be unconnected
                    if pad.number not in ['', 'MP', 'NC', 'N/C']:
                        self.issues.append(PreRouteIssue(
                            severity="warning",
                            category="connectivity",
                            message=f"Pad {pad.number} on {ref} has no net assigned",
                            location=f"{ref}.{pad.number}",
                        ))

    def _check_single_pad_nets(self):
        """Check for nets connected to only one pad."""
        for net_name, net in self.board.nets.items():
            if len(net.connections) == 1:
                conn = net.connections[0]
                self.issues.append(PreRouteIssue(
                    severity="warning",
                    category="connectivity",
                    message=f"Net '{net_name}' has only one connection ({conn[0]}.{conn[1]})",
                    location=net_name,
                ))

    def _check_missing_footprints(self):
        """Check for components without valid footprints."""
        for ref, comp in self.board.components.items():
            if not comp.footprint:
                self.issues.append(PreRouteIssue(
                    severity="error",
                    category="footprint",
                    message=f"Component {ref} has no footprint assigned",
                    location=ref,
                ))

            if not comp.pads:
                self.issues.append(PreRouteIssue(
                    severity="error",
                    category="footprint",
                    message=f"Component {ref} has no pads",
                    location=ref,
                ))

    def _check_overlapping_pads(self):
        """Check for pads that overlap between different components.

        Uses a coarse grid for candidate finding, then verifies with
        actual pad geometry to avoid false positives.
        """
        import math

        # Build spatial index of all pads (coarse grid for candidate pairs)
        # Use larger grid size and check neighboring cells too
        grid_size = 1.0  # mm (coarser grid, then verify precisely)
        pad_info: Dict[Tuple[int, int], List[Tuple[str, str, float, float, float, float]]] = {}
        # Store: (ref, pad_num, abs_x, abs_y, half_width, half_height)

        for ref, comp in self.board.components.items():
            for pad in comp.pads:
                abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                grid_x = int(abs_x / grid_size)
                grid_y = int(abs_y / grid_size)

                # Store pad info with dimensions
                half_w = pad.width / 2
                half_h = pad.height / 2
                info = (ref, pad.number, abs_x, abs_y, half_w, half_h)

                # Add to current and neighboring cells for broad-phase
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        key = (grid_x + dx, grid_y + dy)
                        if key not in pad_info:
                            pad_info[key] = []
                        pad_info[key].append(info)

        # Check for actual overlaps with precise geometry
        checked_pairs: Set[Tuple[str, str, str, str]] = set()

        for cell_pads in pad_info.values():
            if len(cell_pads) < 2:
                continue

            for i, pad1 in enumerate(cell_pads):
                ref1, num1, x1, y1, hw1, hh1 = pad1
                for pad2 in cell_pads[i+1:]:
                    ref2, num2, x2, y2, hw2, hh2 = pad2

                    # Skip same component
                    if ref1 == ref2:
                        continue

                    # Skip if already checked this pair
                    pair_key = tuple(sorted([(ref1, num1), (ref2, num2)]))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    # Check actual overlap using axis-aligned bounding box
                    # Add small clearance (0.05mm) to avoid false positives
                    clearance = 0.05
                    dx = abs(x1 - x2)
                    dy = abs(y1 - y2)

                    # For circular pads, use radius
                    # For rectangular, use half-dimensions
                    # Use simplified check: if centers are closer than sum of half-sizes
                    min_dx = hw1 + hw2 - clearance
                    min_dy = hh1 + hh2 - clearance

                    if dx < min_dx and dy < min_dy:
                        self.issues.append(PreRouteIssue(
                            severity="error",
                            category="placement",
                            message=f"Overlapping pads: {ref1}.{num1} and {ref2}.{num2}",
                            location=f"({(x1+x2)/2:.2f}, {(y1+y2)/2:.2f})",
                        ))

    def _check_power_connections(self):
        """Verify power and ground net connectivity."""
        power_nets = self.board.get_power_nets()
        ground_nets = self.board.get_ground_nets()

        # Check that ICs have power connections
        ics = self.board.get_components_by_prefix('U')
        for ic in ics:
            ic_nets = ic.get_connected_nets()

            has_power = any(n in [net.name for net in power_nets] for n in ic_nets)
            has_ground = any(n in [net.name for net in ground_nets] for n in ic_nets)

            if not has_power:
                self.issues.append(PreRouteIssue(
                    severity="warning",
                    category="connectivity",
                    message=f"IC {ic.reference} may be missing power connection",
                    location=ic.reference,
                ))

            if not has_ground:
                self.issues.append(PreRouteIssue(
                    severity="warning",
                    category="connectivity",
                    message=f"IC {ic.reference} may be missing ground connection",
                    location=ic.reference,
                ))

    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if not self.issues:
            return "Pre-route validation passed with no issues."

        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        infos = sum(1 for i in self.issues if i.severity == "info")

        lines = [
            f"Pre-route validation: {errors} errors, {warnings} warnings, {infos} info",
            "",
        ]

        for issue in self.issues:
            prefix = {"error": "[ERROR]", "warning": "[WARN]", "info": "[INFO]"}
            lines.append(f"{prefix[issue.severity]} {issue.message}")

        return "\n".join(lines)
