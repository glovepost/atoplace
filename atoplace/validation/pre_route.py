"""
Pre-Routing Validation

Validates board state before attempting autorouting to catch issues early.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
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
        """Check for pads that overlap between different components."""
        # Build spatial index of all pads
        pad_locations: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}
        grid_size = 0.1  # mm

        for ref, comp in self.board.components.items():
            for pad in comp.pads:
                abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                grid_x = int(abs_x / grid_size)
                grid_y = int(abs_y / grid_size)

                key = (grid_x, grid_y)
                if key not in pad_locations:
                    pad_locations[key] = []
                pad_locations[key].append((ref, pad.number))

        # Check for multiple pads in same grid cell
        for (gx, gy), pads in pad_locations.items():
            if len(pads) > 1:
                # Check if from different components
                refs = set(p[0] for p in pads)
                if len(refs) > 1:
                    pad_str = ", ".join(f"{r}.{p}" for r, p in pads)
                    self.issues.append(PreRouteIssue(
                        severity="error",
                        category="placement",
                        message=f"Overlapping pads detected: {pad_str}",
                        location=f"({gx * grid_size:.1f}, {gy * grid_size:.1f})",
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
