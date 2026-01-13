"""
Confidence Scoring System

Assesses the quality of a PCB design and flags areas requiring human review.
Based on rules from layout_rules_research.md and industry best practices.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

from ..board.abstraction import Board, Component, Net
from ..dfm.profiles import DFMProfile, get_profile


class Severity(Enum):
    """Severity levels for design flags."""
    INFO = "info"           # Informational note
    WARNING = "warning"     # Should be reviewed
    ERROR = "error"         # Likely problem
    CRITICAL = "critical"   # Must be fixed


class FlagCategory(Enum):
    """Categories of design flags."""
    PLACEMENT = "placement"
    ROUTING = "routing"
    DFM = "dfm"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    EMI = "emi"


@dataclass
class DesignFlag:
    """A flagged issue or note about the design."""
    severity: Severity
    category: FlagCategory
    location: str          # Component ref, net name, or board region
    message: str
    suggested_action: Optional[str] = None
    confidence: float = 1.0  # How confident we are this is an issue (0-1)
    rule_source: str = ""    # Where this rule comes from

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "location": self.location,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
            "rule_source": self.rule_source,
        }


@dataclass
class ConfidenceReport:
    """Complete confidence assessment for a design."""
    overall_score: float = 1.0   # 0-1, where 1 = high confidence
    flags: List[DesignFlag] = field(default_factory=list)

    # Sub-scores
    placement_score: float = 1.0
    routing_score: float = 1.0
    dfm_score: float = 1.0
    electrical_score: float = 1.0

    # Statistics
    component_count: int = 0
    net_count: int = 0
    layer_count: int = 0

    def human_review_required(self) -> bool:
        """Check if design needs human review."""
        has_critical = any(f.severity == Severity.CRITICAL for f in self.flags)
        has_errors = any(f.severity == Severity.ERROR for f in self.flags)
        low_confidence = self.overall_score < 0.7
        return has_critical or has_errors or low_confidence

    def get_flags_by_severity(self, severity: Severity) -> List[DesignFlag]:
        """Get all flags of a specific severity."""
        return [f for f in self.flags if f.severity == severity]

    def get_flags_by_category(self, category: FlagCategory) -> List[DesignFlag]:
        """Get all flags in a specific category."""
        return [f for f in self.flags if f.category == category]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Overall Confidence: {self.overall_score:.0%}",
            f"  Placement: {self.placement_score:.0%}",
            f"  Routing: {self.routing_score:.0%}",
            f"  DFM: {self.dfm_score:.0%}",
            f"  Electrical: {self.electrical_score:.0%}",
            "",
            f"Components: {self.component_count}",
            f"Nets: {self.net_count}",
            f"Layers: {self.layer_count}",
            "",
        ]

        if self.flags:
            lines.append("Flags:")

            # Sort by severity
            severity_order = [Severity.CRITICAL, Severity.ERROR,
                              Severity.WARNING, Severity.INFO]

            for severity in severity_order:
                severity_flags = self.get_flags_by_severity(severity)
                if severity_flags:
                    icon = {
                        Severity.CRITICAL: "[CRITICAL]",
                        Severity.ERROR: "[ERROR]",
                        Severity.WARNING: "[WARNING]",
                        Severity.INFO: "[INFO]",
                    }[severity]

                    for flag in severity_flags:
                        lines.append(f"  {icon} [{flag.category.value}] {flag.message}")
                        if flag.suggested_action:
                            lines.append(f"      -> {flag.suggested_action}")
        else:
            lines.append("No issues found.")

        if self.human_review_required():
            lines.append("")
            lines.append("*** HUMAN REVIEW REQUIRED ***")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Design Confidence Report",
            "",
            "## Summary",
            "",
            f"| Metric | Score |",
            f"|--------|-------|",
            f"| Overall | {self.overall_score:.0%} |",
            f"| Placement | {self.placement_score:.0%} |",
            f"| Routing | {self.routing_score:.0%} |",
            f"| DFM | {self.dfm_score:.0%} |",
            f"| Electrical | {self.electrical_score:.0%} |",
            "",
            "## Statistics",
            "",
            f"- Components: {self.component_count}",
            f"- Nets: {self.net_count}",
            f"- Layers: {self.layer_count}",
            "",
        ]

        if self.flags:
            lines.extend([
                "## Issues",
                "",
            ])

            for severity in [Severity.CRITICAL, Severity.ERROR,
                             Severity.WARNING, Severity.INFO]:
                severity_flags = self.get_flags_by_severity(severity)
                if severity_flags:
                    lines.append(f"### {severity.value.title()}")
                    lines.append("")
                    for flag in severity_flags:
                        lines.append(f"- **[{flag.category.value}]** {flag.message}")
                        if flag.suggested_action:
                            lines.append(f"  - Action: {flag.suggested_action}")
                        if flag.rule_source:
                            lines.append(f"  - Source: {flag.rule_source}")
                    lines.append("")

        if self.human_review_required():
            lines.extend([
                "---",
                "",
                "> **Note:** This design requires human review before manufacturing.",
                "",
            ])

        return "\n".join(lines)


class ConfidenceScorer:
    """
    Score design confidence based on rules and heuristics.

    Implements checks based on layout_rules_research.md:
    - Decoupling capacitor placement
    - Power delivery network quality
    - Component overlap detection
    - High-speed signal identification
    - DFM rule compliance
    """

    def __init__(self, dfm_profile: Optional[DFMProfile] = None):
        """
        Initialize scorer.

        Args:
            dfm_profile: DFM profile for checking manufacturability
        """
        self.dfm_profile = dfm_profile or get_profile("jlcpcb_standard")

    def assess(self, board: Board, routing_done: bool = False) -> ConfidenceReport:
        """
        Generate confidence report for a board.

        Args:
            board: Board to assess
            routing_done: Whether routing has been completed

        Returns:
            ConfidenceReport with scores and flags
        """
        report = ConfidenceReport(
            component_count=len(board.components),
            net_count=len(board.nets),
            layer_count=board.layer_count,
        )

        # Run all checks
        placement_flags, placement_score = self._check_placement(board)
        report.flags.extend(placement_flags)
        report.placement_score = placement_score

        if routing_done:
            routing_flags, routing_score = self._check_routing(board)
            report.flags.extend(routing_flags)
            report.routing_score = routing_score
        else:
            report.routing_score = 0.5  # Unknown

        dfm_flags, dfm_score = self._check_dfm(board)
        report.flags.extend(dfm_flags)
        report.dfm_score = dfm_score

        electrical_flags, electrical_score = self._check_electrical(board)
        report.flags.extend(electrical_flags)
        report.electrical_score = electrical_score

        # Calculate overall score
        weights = {
            "placement": 0.3,
            "routing": 0.3 if routing_done else 0.0,
            "dfm": 0.2,
            "electrical": 0.2 if routing_done else 0.4,
        }

        total_weight = sum(weights.values())
        report.overall_score = (
            weights["placement"] * report.placement_score +
            weights["routing"] * report.routing_score +
            weights["dfm"] * report.dfm_score +
            weights["electrical"] * report.electrical_score
        ) / total_weight

        # Reduce score for critical/error flags
        critical_count = len(report.get_flags_by_severity(Severity.CRITICAL))
        error_count = len(report.get_flags_by_severity(Severity.ERROR))

        report.overall_score *= (0.5 ** critical_count) * (0.9 ** error_count)
        report.overall_score = max(0.0, min(1.0, report.overall_score))

        return report

    def _check_placement(self, board: Board) -> Tuple[List[DesignFlag], float]:
        """Check placement quality."""
        flags = []
        score = 1.0

        # Check for overlapping components
        # - check_layers: ensure Top/Bottom components don't falsely conflict
        # - include_pads: ensure pad-inclusive bounding boxes are used for accuracy
        overlaps = board.find_overlaps(self.dfm_profile.min_spacing, check_layers=True, include_pads=True)
        for ref1, ref2, dist in overlaps:
            # Skip if either component is DNP
            c1 = board.get_component(ref1)
            c2 = board.get_component(ref2)
            if (c1 and c1.dnp) or (c2 and c2.dnp):
                continue
            flags.append(DesignFlag(
                severity=Severity.CRITICAL,
                category=FlagCategory.PLACEMENT,
                location=f"{ref1}/{ref2}",
                message=f"Components {ref1} and {ref2} overlap or too close ({dist:.2f}mm)",
                suggested_action="Increase spacing or relocate components",
                confidence=1.0,
                rule_source="DFM: Minimum component clearance",
            ))
            score = 0.0  # Critical failure

        # Check decoupling capacitor placement
        decoupling_flags, decoupling_score = self._check_decoupling(board)
        flags.extend(decoupling_flags)
        score *= decoupling_score

        # Check if components are within board boundaries
        boundary_flags = self._check_boundaries(board)
        flags.extend(boundary_flags)
        if any(f.severity == Severity.ERROR for f in boundary_flags):
            score *= 0.5

        # Check for high-density areas
        density_flags = self._check_density(board)
        flags.extend(density_flags)

        return flags, max(score, 0.0)

    def _check_decoupling(self, board: Board) -> Tuple[List[DesignFlag], float]:
        """Check decoupling capacitor placement.

        Uses adaptive distance limits based on IC type:
        - High-speed ICs (USB, Ethernet, RF, high-freq clocks): <2mm recommended
        - Standard digital ICs: <5mm recommended
        - Low-speed ICs: <10mm acceptable

        The distance requirements are based on inductance considerations:
        - 1mm of trace ≈ 1nH inductance
        - At high frequencies, even small inductance can cause significant voltage droop
        """
        flags = []
        score = 1.0

        # High-speed IC patterns (require closer decoupling)
        high_speed_patterns = [
            'USB', 'ETH', 'ENET', 'PHY', 'HDMI', 'LVDS', 'PCIE',
            'DDR', 'SDRAM', 'FPGA', 'CPLD', 'RF', 'WIFI', 'BLE', 'BT',
            'GHZ', 'MHZ',  # Frequency indicators
        ]

        # Medium-speed digital patterns
        medium_speed_patterns = [
            'STM32', 'ESP32', 'NRF', 'ATMEGA', 'PIC', 'RP2040',  # MCUs
            'SPI', 'QSPI', 'I2C', 'UART',  # Communication interfaces
        ]

        # Find ICs (U* components)
        ics = board.get_components_by_prefix('U')

        for ic in ics:
            if ic.dnp:  # Skip Do Not Populate components
                continue
            # Determine IC speed class and appropriate distance limits
            ic_value = ic.value.upper() if ic.value else ""
            ic_footprint = ic.footprint.upper() if ic.footprint else ""
            ic_str = ic_value + " " + ic_footprint

            is_high_speed = any(p in ic_str for p in high_speed_patterns)
            is_medium_speed = any(p in ic_str for p in medium_speed_patterns)

            # Set distance thresholds based on IC type
            if is_high_speed:
                critical_distance = 2.0   # mm - high-speed needs very close decoupling
                warning_distance = 3.0
                info_distance = 5.0
                speed_class = "high-speed"
            elif is_medium_speed:
                critical_distance = 5.0   # mm - standard digital
                warning_distance = 7.0
                info_distance = 10.0
                speed_class = "digital"
            else:
                critical_distance = 10.0  # mm - low-speed/unknown
                warning_distance = 15.0
                info_distance = 20.0
                speed_class = "standard"

            # Find connected capacitors
            ic_nets = ic.get_connected_nets()
            power_nets = [n for n in ic_nets if board.nets.get(n) and
                          board.nets[n].is_power]

            if not power_nets:
                continue

            # Find closest decoupling cap (must be connected to power AND ground)
            # Get ground net names for verification
            ground_net_names = {n.name for n in board.get_ground_nets()}

            min_cap_distance = float('inf')
            closest_cap = None

            for net_name in power_nets:
                net = board.nets.get(net_name)
                if not net:
                    continue

                for ref in net.get_component_refs():
                    if ref.startswith('C'):
                        cap = board.get_component(ref)
                        if cap:
                            # Verify this is a decoupling cap (connected to ground)
                            # True decoupling caps are connected to both power and ground
                            cap_nets = cap.get_connected_nets()
                            is_decoupling = any(n in ground_net_names for n in cap_nets)

                            if not is_decoupling:
                                # Skip - this capacitor is not connected to ground
                                # (e.g., bulk caps, filter caps, coupling caps)
                                continue

                            dist = ic.distance_to(cap)
                            if dist < min_cap_distance:
                                min_cap_distance = dist
                                closest_cap = cap

            if closest_cap:
                if min_cap_distance > critical_distance:
                    severity = Severity.WARNING if is_high_speed else Severity.INFO
                    flags.append(DesignFlag(
                        severity=severity,
                        category=FlagCategory.PLACEMENT,
                        location=ic.reference,
                        message=f"Decoupling cap {closest_cap.reference} is {min_cap_distance:.1f}mm from {speed_class} IC {ic.reference} (recommended <{critical_distance}mm)",
                        suggested_action=f"Move {closest_cap.reference} closer to {ic.reference} power pins",
                        confidence=0.9 if is_high_speed else 0.7,
                        rule_source="layout_rules: PDN/Decoupling Capacitors",
                    ))
                    score *= 0.95 if is_high_speed else 0.98
                elif min_cap_distance > warning_distance:
                    flags.append(DesignFlag(
                        severity=Severity.INFO,
                        category=FlagCategory.PLACEMENT,
                        location=ic.reference,
                        message=f"Decoupling cap {closest_cap.reference} could be closer to {ic.reference} ({min_cap_distance:.1f}mm)",
                        suggested_action="Consider moving closer for better high-frequency decoupling",
                        confidence=0.6,
                        rule_source="layout_rules: PDN/Decoupling Capacitors",
                    ))

        return flags, score

    def _check_boundaries(self, board: Board) -> List[DesignFlag]:
        """Check if all components are within board boundaries.

        Uses BoardOutline.contains_point() which properly handles:
        - Polygon outlines (non-rectangular boards)
        - Board cutouts/holes
        - Margin enforcement for edge clearance

        Skipped when board has no explicit outline defined.
        """
        flags = []
        outline = board.outline

        # Skip boundary checks when no explicit outline is defined
        if not outline.has_outline:
            return flags

        margin = self.dfm_profile.min_trace_to_edge

        for ref, comp in board.components.items():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            # Use pad-inclusive bounding box to catch pads that protrude beyond body
            bbox = comp.get_bounding_box_with_pads()

            # Check all 4 corners of bounding box against board outline
            corners = [
                (bbox[0], bbox[1]),  # min_x, min_y (bottom-left)
                (bbox[2], bbox[1]),  # max_x, min_y (bottom-right)
                (bbox[0], bbox[3]),  # min_x, max_y (top-left)
                (bbox[2], bbox[3]),  # max_x, max_y (top-right)
            ]

            outside_corners = []
            for cx, cy in corners:
                if not outline.contains_point(cx, cy, margin):
                    outside_corners.append((cx, cy))

            if outside_corners:
                flags.append(DesignFlag(
                    severity=Severity.ERROR,
                    category=FlagCategory.PLACEMENT,
                    location=ref,
                    message=f"Component {ref} extends outside board boundary or too close to edge ({len(outside_corners)} corners violate {margin:.2f}mm clearance)",
                    suggested_action="Move component within board outline",
                    confidence=1.0,
                    rule_source="DFM: Board edge clearance",
                ))

        return flags

    def _check_density(self, board: Board) -> List[DesignFlag]:
        """Check for problematic component density."""
        flags = []

        # Calculate board utilization using proper area (handles polygon outlines)
        board_area = board._calculate_board_area()
        component_area = 0.0

        for comp in board.components.values():
            if comp.dnp:  # Skip Do Not Populate components
                continue
            component_area += comp.width * comp.height

        utilization = component_area / board_area if board_area > 0 else 0

        if utilization > 0.7:
            flags.append(DesignFlag(
                severity=Severity.WARNING,
                category=FlagCategory.PLACEMENT,
                location="board",
                message=f"High component density ({utilization:.0%}) may cause routing difficulties",
                suggested_action="Consider larger board or component reorganization",
                confidence=0.8,
                rule_source="DFM: Routability",
            ))
        elif utilization > 0.5:
            flags.append(DesignFlag(
                severity=Severity.INFO,
                category=FlagCategory.PLACEMENT,
                location="board",
                message=f"Moderate component density ({utilization:.0%})",
                confidence=0.6,
            ))

        return flags

    def _check_routing(self, board: Board) -> Tuple[List[DesignFlag], float]:
        """Check routing quality (placeholder for post-routing checks)."""
        # TODO: Implement routing checks when routing is available
        return [], 1.0

    def _check_dfm(self, board: Board) -> Tuple[List[DesignFlag], float]:
        """Check DFM compliance."""
        flags = []
        score = 1.0

        # Check layer count compatibility
        if board.layer_count not in self.dfm_profile.supported_layers:
            flags.append(DesignFlag(
                severity=Severity.ERROR,
                category=FlagCategory.DFM,
                location="board",
                message=f"{board.layer_count}-layer board not supported by {self.dfm_profile.name}",
                suggested_action=f"Use supported layer count: {self.dfm_profile.supported_layers}",
                confidence=1.0,
                rule_source=f"DFM: {self.dfm_profile.name}",
            ))
            score *= 0.5

        # Check board size - maximum
        if board.outline.width > self.dfm_profile.max_board_size:
            flags.append(DesignFlag(
                severity=Severity.ERROR,
                category=FlagCategory.DFM,
                location="board",
                message=f"Board width ({board.outline.width}mm) exceeds maximum ({self.dfm_profile.max_board_size}mm)",
                confidence=1.0,
            ))
            score *= 0.5

        if board.outline.height > self.dfm_profile.max_board_size:
            flags.append(DesignFlag(
                severity=Severity.ERROR,
                category=FlagCategory.DFM,
                location="board",
                message=f"Board height ({board.outline.height}mm) exceeds maximum ({self.dfm_profile.max_board_size}mm)",
                confidence=1.0,
            ))
            score *= 0.5

        # Check board size - minimum
        min_size = self.dfm_profile.min_board_size
        if board.outline.width < min_size or board.outline.height < min_size:
            flags.append(DesignFlag(
                severity=Severity.WARNING,
                category=FlagCategory.DFM,
                location="board",
                message=f"Board dimensions ({board.outline.width:.1f}x{board.outline.height:.1f}mm) below minimum ({min_size}mm)",
                suggested_action=f"Increase board dimensions or verify with fab house",
                confidence=0.9,
                rule_source=f"DFM: {self.dfm_profile.name}",
            ))
            score *= 0.9

        return flags, score

    def _check_electrical(self, board: Board) -> Tuple[List[DesignFlag], float]:
        """Check electrical design quality."""
        flags = []
        score = 1.0

        # Identify high-speed signals
        hs_nets = self._identify_high_speed_nets(board)
        for net_name in hs_nets:
            flags.append(DesignFlag(
                severity=Severity.INFO,
                category=FlagCategory.ELECTRICAL,
                location=net_name,
                message=f"High-speed signal '{net_name}' detected - verify impedance matching",
                suggested_action="Review trace length matching and impedance control",
                confidence=0.8,
                rule_source="layout_rules: Differential Pairs",
            ))

        # Check for unconnected power nets
        for net_name, net in board.nets.items():
            if net.is_power and len(net.connections) == 0:
                flags.append(DesignFlag(
                    severity=Severity.ERROR,
                    category=FlagCategory.ELECTRICAL,
                    location=net_name,
                    message=f"Power net '{net_name}' has no connections",
                    confidence=1.0,
                ))
                score *= 0.8

        return flags, score

    def _identify_high_speed_nets(self, board: Board) -> List[str]:
        """Identify high-speed signal nets.

        Uses specific patterns to avoid false positives. For differential pairs,
        requires suffix matching (_P/_N at end of name) rather than substring
        matching to avoid flagging nets like TEMP, EN_P, etc.
        """
        hs_nets = []

        # Patterns that can match anywhere in net name
        hs_patterns = [
            'USB', 'HDMI', 'LVDS', 'PCIE', 'ETH', 'SDIO', 'QSPI',
            '_D+', '_D-', 'CLK', 'MCLK', 'SCLK',
        ]

        # Differential pair suffixes (must be at end of net name)
        # NOTE: Avoid single-letter suffixes like 'P', 'N' which cause false positives
        # (e.g., TEMP, EN, LOOP would incorrectly match). Use '_P', '_N' or '+', '-' instead.
        diff_pair_suffixes = ['_P', '_N', '+', '-']

        for net_name in board.nets:
            name_upper = net_name.upper()

            # Check standard high-speed patterns
            for pattern in hs_patterns:
                if pattern in name_upper:
                    hs_nets.append(net_name)
                    break
            else:
                # Check for differential pair suffixes at end of name
                # Only flag if there's a base signal name (at least 2 chars)
                if len(name_upper) >= 3:
                    for suffix in diff_pair_suffixes:
                        if name_upper.endswith(suffix) and len(suffix) < len(name_upper):
                            # Additional check: avoid single-char base names
                            base = name_upper[:-len(suffix)]
                            if len(base) >= 2 and base[-1] != '_':
                                hs_nets.append(net_name)
                                break

        return hs_nets
