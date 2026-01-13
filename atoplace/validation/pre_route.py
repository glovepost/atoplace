"""
Pre-Routing Validation

Validates board state before attempting autorouting to catch issues early.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, FrozenSet, Optional
from ..board.abstraction import Board, Component
from ..dfm.profiles import DFMProfile


@dataclass
class PreRouteIssue:
    """An issue found during pre-route validation."""
    severity: str  # "error", "warning", "info"
    category: str  # "connectivity", "placement", "footprint"
    message: str
    location: str  # Component or net reference


class PreRouteValidator:
    """Validates board before routing."""

    def __init__(self, board: Board, dfm_profile: Optional[DFMProfile] = None):
        self.board = board
        self.dfm_profile = dfm_profile
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
            if comp.dnp:  # Skip Do Not Populate components
                continue
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
            if comp.dnp:  # Skip Do Not Populate components
                continue
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

        Uses a spatial grid for candidate finding, then verifies with
        actual pad geometry. Grid size is based on minimum pad dimensions
        to ensure fine-pitch components are properly checked.

        Uses DFM profile's min_spacing for clearance if available.
        """
        import math

        # Determine minimum clearance from DFM profile or use sensible default
        if self.dfm_profile:
            min_clearance = self.dfm_profile.min_spacing
        else:
            min_clearance = 0.1  # Default 0.1mm clearance

        # Calculate grid size based on the smallest pad dimension in the design
        # This ensures fine-pitch components (0.5mm, 0.4mm pitch) are not missed
        min_pad_dim = float('inf')
        for ref, comp in self.board.components.items():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            for pad in comp.pads:
                min_pad_dim = min(min_pad_dim, pad.width, pad.height)

        # Grid size should be at least 2x the minimum pad dimension
        # but not too small to avoid performance issues
        if min_pad_dim == float('inf'):
            grid_size = 0.5  # Default if no pads found
        else:
            grid_size = max(0.2, min(1.0, min_pad_dim * 2))

        # Build spatial index of all pads
        # Store: (ref, pad_num, abs_x, abs_y, half_width, half_height, layer, is_through_hole)
        # where half-dimensions account for pad rotation
        pad_info: Dict[Tuple[int, int], List[Tuple[str, str, float, float, float, float, str, bool]]] = {}

        for ref, comp in self.board.components.items():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            for pad in comp.pads:
                # Get pad bounding box which accounts for pad rotation
                bbox = pad.get_bounding_box(comp.x, comp.y, comp.rotation)
                abs_x = (bbox[0] + bbox[2]) / 2  # Center X
                abs_y = (bbox[1] + bbox[3]) / 2  # Center Y
                half_w = (bbox[2] - bbox[0]) / 2  # Half-width of AABB
                half_h = (bbox[3] - bbox[1]) / 2  # Half-height of AABB

                # Determine layer and through-hole status
                is_through_hole = pad.drill is not None and pad.drill > 0
                pad_layer = pad.layer.value if pad.layer else "F.Cu"

                # Use floor() instead of int() for correct bucketing with negative coordinates
                # int() truncates toward zero, so -0.5 -> 0, but floor(-0.5) -> -1
                grid_x = math.floor(abs_x / grid_size)
                grid_y = math.floor(abs_y / grid_size)

                info = (ref, pad.number, abs_x, abs_y, half_w, half_h, pad_layer, is_through_hole)

                # Add to current and neighboring cells for broad-phase collision
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        key = (grid_x + dx, grid_y + dy)
                        if key not in pad_info:
                            pad_info[key] = []
                        pad_info[key].append(info)

        # Check for actual overlaps with precise geometry
        checked_pairs: Set[FrozenSet[Tuple[str, str]]] = set()

        for cell_pads in pad_info.values():
            if len(cell_pads) < 2:
                continue

            for i, pad1 in enumerate(cell_pads):
                ref1, num1, x1, y1, hw1, hh1, layer1, th1 = pad1
                for pad2 in cell_pads[i+1:]:
                    ref2, num2, x2, y2, hw2, hh2, layer2, th2 = pad2

                    # Skip same component - pads within a component can be close
                    if ref1 == ref2:
                        continue

                    # Skip if already checked this pair
                    pair_key = frozenset([(ref1, num1), (ref2, num2)])
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    # Layer-aware collision detection:
                    # - Through-hole pads conflict with everything (both layers)
                    # - SMD pads only conflict with pads on the same layer
                    if not th1 and not th2:
                        # Both are SMD - check if they're on the same layer
                        if layer1 != layer2:
                            # Different layers, no collision possible
                            continue

                    # Check clearance using axis-aligned bounding box
                    dx = abs(x1 - x2)
                    dy = abs(y1 - y2)

                    # Required separation: sum of half-widths plus clearance
                    required_dx = hw1 + hw2 + min_clearance
                    required_dy = hh1 + hh2 + min_clearance

                    # Pads overlap if both axes are within required separation
                    if dx < required_dx and dy < required_dy:
                        # Calculate actual overlap/clearance violation
                        overlap_x = required_dx - dx
                        overlap_y = required_dy - dy
                        overlap = min(overlap_x, overlap_y)

                        self.issues.append(PreRouteIssue(
                            severity="error",
                            category="placement",
                            message=f"Pad clearance violation: {ref1}.{num1} and {ref2}.{num2} "
                                    f"(clearance: {max(0, min(dx - hw1 - hw2, dy - hh1 - hh2)):.3f}mm < {min_clearance:.3f}mm)",
                            location=f"({(x1+x2)/2:.2f}, {(y1+y2)/2:.2f})",
                        ))

    def _check_power_connections(self):
        """Verify power and ground net connectivity."""
        power_nets = self.board.get_power_nets()
        ground_nets = self.board.get_ground_nets()

        # Check that ICs have power connections
        ics = self.board.get_components_by_prefix('U')
        for ic in ics:
            if ic.dnp:  # Skip Do Not Populate components
                continue
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
        """Get a summary of validation results including category and location context."""
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
            # Include category and location for actionable output
            lines.append(f"{prefix[issue.severity]} [{issue.category}] {issue.message}")
            if issue.location:
                lines.append(f"    Location: {issue.location}")

        return "\n".join(lines)
