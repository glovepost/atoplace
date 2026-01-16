"""
DRC (Design Rule Check) Runner for MCP Server.

Provides DRC capabilities via two backends:
1. kicad-cli: Native KiCad DRC (comprehensive, requires kicad-cli in PATH)
2. atoplace: Python-based DRC checks (always available, placement-focused)

The runner can combine both for comprehensive coverage.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..board.abstraction import Board
from ..dfm.profiles import DFMProfile, get_profile
from ..validation.drc import DRCChecker, DRCViolation

logger = logging.getLogger(__name__)


# =============================================================================
# DRC Result Data Structures
# =============================================================================

@dataclass
class DRCViolationInfo:
    """Unified DRC violation format for MCP responses."""
    id: str
    rule: str
    severity: str  # "error", "warning"
    message: str
    location: Dict[str, float]  # {"x": float, "y": float}
    items: List[str]  # Affected refs/nets
    actionable: bool
    suggested_action: Optional[str]
    source: str  # "kicad-cli" or "atoplace"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DRCResult:
    """Complete DRC result."""
    passed: bool
    error_count: int
    warning_count: int
    violations: List[DRCViolationInfo]
    source: str  # "kicad-cli", "atoplace", or "combined"
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "violations": [v.to_dict() for v in self.violations],
            "source": self.source,
            "summary": self.summary,
        }


# =============================================================================
# Violation Classification
# =============================================================================

# Rules that can be fixed by moving components
ACTIONABLE_RULES = {
    "clearance",
    "component_clearance",
    "edge_clearance",
    "courtyard_overlap",
}

# Rules with limited fix options
PARTIALLY_ACTIONABLE_RULES = {
    "hole_to_hole",
    "hole_to_edge",
}

# Suggested actions by rule type
SUGGESTED_ACTIONS = {
    "clearance": "Move one of the components to increase spacing",
    "component_clearance": "Move one of the components to increase spacing",
    "edge_clearance": "Move component away from board edge",
    "courtyard_overlap": "Move components apart to eliminate courtyard overlap",
    "hole_to_hole": "Consider adjusting component placement (pad positions are fixed)",
    "hole_to_edge": "Move component away from board edge",
    "min_drill_size": "Check footprint - drill size is a footprint property",
    "min_annular_ring": "Check footprint - annular ring is a footprint property",
    "track_width": "Adjust track width in routing (not a placement issue)",
    "unconnected": "Route the connection (not a placement issue)",
}


def classify_violation(rule: str) -> Tuple[bool, Optional[str]]:
    """
    Classify a violation rule as actionable or not.

    Returns:
        (actionable, suggested_action)
    """
    rule_lower = rule.lower()

    # Check actionable rules
    for actionable_rule in ACTIONABLE_RULES:
        if actionable_rule in rule_lower:
            return True, SUGGESTED_ACTIONS.get(actionable_rule)

    # Check partially actionable rules
    for partial_rule in PARTIALLY_ACTIONABLE_RULES:
        if partial_rule in rule_lower:
            return True, SUGGESTED_ACTIONS.get(partial_rule)

    # Look up by exact match or prefix
    for key, action in SUGGESTED_ACTIONS.items():
        if rule_lower.startswith(key):
            return key in ACTIONABLE_RULES or key in PARTIALLY_ACTIONABLE_RULES, action

    return False, None


# =============================================================================
# DRC Runner
# =============================================================================

class DRCRunner:
    """
    Runs DRC checks using available backends.

    Supports:
    - kicad-cli: Native KiCad DRC via command line
    - atoplace: Python-based DRC checks
    """

    def __init__(self):
        self._violation_counter = 0
        self._cached_violations: Dict[str, DRCViolationInfo] = {}
        self._kicad_cli_path: Optional[str] = None

    def _next_violation_id(self) -> str:
        """Generate unique violation ID."""
        self._violation_counter += 1
        return f"v{self._violation_counter:03d}"

    def clear_cache(self):
        """Clear violation cache (call before new DRC run)."""
        self._violation_counter = 0
        self._cached_violations.clear()

    def get_violation(self, violation_id: str) -> Optional[DRCViolationInfo]:
        """Get cached violation by ID."""
        return self._cached_violations.get(violation_id)

    # =========================================================================
    # kicad-cli Backend
    # =========================================================================

    def find_kicad_cli(self) -> Optional[str]:
        """Find kicad-cli executable."""
        if self._kicad_cli_path:
            return self._kicad_cli_path

        # Check common locations
        candidates = [
            "kicad-cli",  # In PATH
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",  # macOS
            "/usr/bin/kicad-cli",  # Linux
            "/usr/local/bin/kicad-cli",  # Linux alt
            r"C:\Program Files\KiCad\8.0\bin\kicad-cli.exe",  # Windows
            r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe",  # Windows 9.0
        ]

        # Check environment variable
        env_path = os.environ.get("KICAD_CLI")
        if env_path:
            candidates.insert(0, env_path)

        for candidate in candidates:
            if shutil.which(candidate):
                self._kicad_cli_path = candidate
                logger.info("Found kicad-cli at: %s", candidate)
                return candidate

        logger.debug("kicad-cli not found in common locations")
        return None

    def run_kicad_cli_drc(self, board_path: Path) -> Optional[DRCResult]:
        """
        Run DRC using kicad-cli.

        Args:
            board_path: Path to .kicad_pcb file

        Returns:
            DRCResult or None if kicad-cli unavailable
        """
        kicad_cli = self.find_kicad_cli()
        if not kicad_cli:
            logger.warning("kicad-cli not found, skipping native DRC")
            return None

        if not board_path.exists():
            logger.error("Board file not found: %s", board_path)
            return None

        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            # Run kicad-cli DRC
            cmd = [
                kicad_cli, "pcb", "drc",
                "--format", "json",
                "--severity-all",
                "--units", "mm",
                "-o", str(output_path),
                str(board_path)
            ]

            logger.info("Running kicad-cli DRC: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Exit code 5 means violations found (not an error)
            if result.returncode not in (0, 5):
                logger.error("kicad-cli DRC failed: %s", result.stderr)
                return None

            # Parse JSON output
            if not output_path.exists():
                logger.error("DRC output file not created")
                return None

            with open(output_path, 'r') as f:
                drc_data = json.load(f)

            return self._parse_kicad_cli_output(drc_data)

        except subprocess.TimeoutExpired:
            logger.error("kicad-cli DRC timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error("Failed to parse DRC JSON: %s", e)
            return None
        except Exception as e:
            logger.error("kicad-cli DRC error: %s", e)
            return None
        finally:
            # Clean up temp file
            try:
                if output_path.exists():
                    output_path.unlink()
            except OSError as e:
                logger.debug("Failed to cleanup temp DRC file: %s", e)

    def _parse_kicad_cli_output(self, data: Dict) -> DRCResult:
        """Parse kicad-cli JSON output into DRCResult."""
        violations = []
        error_count = 0
        warning_count = 0

        # kicad-cli output structure varies by version
        # Common keys: "violations", "errors", "warnings"
        raw_violations = data.get("violations", [])

        # Also check for "drc" key in some versions
        if "drc" in data:
            raw_violations = data["drc"].get("violations", raw_violations)

        for v in raw_violations:
            severity = v.get("severity", "error").lower()
            if severity == "error":
                error_count += 1
            else:
                warning_count += 1

            # Extract rule type
            rule = v.get("type", v.get("rule", "unknown"))

            # Extract position
            pos = v.get("pos", {})
            if isinstance(pos, dict):
                x = pos.get("x", 0)
                y = pos.get("y", 0)
            else:
                x, y = 0, 0

            # Extract affected items
            items = v.get("items", [])
            if isinstance(items, list):
                # Items might be dicts with "uuid" and "description"
                item_refs = []
                for item in items:
                    if isinstance(item, dict):
                        desc = item.get("description", "")
                        # Try to extract ref from description
                        if desc:
                            item_refs.append(desc.split()[0] if desc else "")
                    else:
                        item_refs.append(str(item))
                items = [i for i in item_refs if i]

            # Classify
            actionable, suggested = classify_violation(rule)

            vid = self._next_violation_id()
            violation = DRCViolationInfo(
                id=vid,
                rule=rule,
                severity=severity,
                message=v.get("description", v.get("message", str(v))),
                location={"x": x, "y": y},
                items=items,
                actionable=actionable,
                suggested_action=suggested,
                source="kicad-cli"
            )
            violations.append(violation)
            self._cached_violations[vid] = violation

        passed = error_count == 0

        return DRCResult(
            passed=passed,
            error_count=error_count,
            warning_count=warning_count,
            violations=violations,
            source="kicad-cli",
            summary=f"KiCad DRC: {error_count} errors, {warning_count} warnings"
        )

    # =========================================================================
    # Atoplace Backend
    # =========================================================================

    def run_atoplace_drc(
        self,
        board: Board,
        dfm_profile: str = "jlcpcb_standard"
    ) -> DRCResult:
        """
        Run DRC using atoplace's built-in checker.

        Args:
            board: Board object to check
            dfm_profile: Name of DFM profile to use

        Returns:
            DRCResult
        """
        # Get DFM profile
        try:
            profile = get_profile(dfm_profile)
        except KeyError:
            logger.warning("Unknown DFM profile: %s, using jlcpcb_standard", dfm_profile)
            profile = get_profile("jlcpcb_standard")

        # Run checks
        checker = DRCChecker(board, profile)
        passed, violations = checker.run_checks()

        # Convert to unified format
        result_violations = []
        error_count = 0
        warning_count = 0

        for v in violations:
            if v.severity == "error":
                error_count += 1
            else:
                warning_count += 1

            actionable, suggested = classify_violation(v.rule)

            vid = self._next_violation_id()
            violation = DRCViolationInfo(
                id=vid,
                rule=v.rule,
                severity=v.severity,
                message=v.message,
                location={"x": v.location[0], "y": v.location[1]},
                items=v.items,
                actionable=actionable,
                suggested_action=suggested,
                source="atoplace"
            )
            result_violations.append(violation)
            self._cached_violations[vid] = violation

        return DRCResult(
            passed=passed,
            error_count=error_count,
            warning_count=warning_count,
            violations=result_violations,
            source="atoplace",
            summary=f"Atoplace DRC: {error_count} errors, {warning_count} warnings"
        )

    # =========================================================================
    # Combined DRC
    # =========================================================================

    def run_combined_drc(
        self,
        board: Board,
        board_path: Optional[Path] = None,
        dfm_profile: str = "jlcpcb_standard",
        use_kicad: bool = True
    ) -> DRCResult:
        """
        Run DRC using both backends and combine results.

        Args:
            board: Board object for atoplace checks
            board_path: Path to .kicad_pcb file for kicad-cli (optional)
            dfm_profile: DFM profile name
            use_kicad: Whether to try kicad-cli

        Returns:
            Combined DRCResult
        """
        self.clear_cache()

        all_violations = []
        total_errors = 0
        total_warnings = 0
        sources = []

        # Run atoplace DRC (always available)
        atoplace_result = self.run_atoplace_drc(board, dfm_profile)
        all_violations.extend(atoplace_result.violations)
        total_errors += atoplace_result.error_count
        total_warnings += atoplace_result.warning_count
        sources.append("atoplace")

        # Run kicad-cli DRC if available and requested
        if use_kicad and board_path and board_path.exists():
            kicad_result = self.run_kicad_cli_drc(board_path)
            if kicad_result:
                all_violations.extend(kicad_result.violations)
                total_errors += kicad_result.error_count
                total_warnings += kicad_result.warning_count
                sources.append("kicad-cli")

        # Determine source label
        if len(sources) > 1:
            source = "combined"
        else:
            source = sources[0]

        passed = total_errors == 0

        return DRCResult(
            passed=passed,
            error_count=total_errors,
            warning_count=total_warnings,
            violations=all_violations,
            source=source,
            summary=f"DRC ({source}): {total_errors} errors, {total_warnings} warnings"
        )


# =============================================================================
# Auto-Fix Strategies
# =============================================================================

class DRCFixer:
    """Automatic DRC violation fixer."""

    def __init__(self, board: Board):
        self.board = board

    def fix_clearance_violation(
        self,
        violation: DRCViolationInfo,
        strategy: str = "auto"
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Attempt to fix a clearance violation.

        Args:
            violation: The violation to fix
            strategy: "auto", "move_first", "move_second", "spread"

        Returns:
            (success, message, updates) - updates is list of component moves
        """
        if len(violation.items) < 2:
            return False, "Need two components to fix clearance", []

        ref1, ref2 = violation.items[0], violation.items[1]

        comp1 = self.board.components.get(ref1)
        comp2 = self.board.components.get(ref2)

        if not comp1 or not comp2:
            return False, f"Component not found: {ref1 if not comp1 else ref2}", []

        # Calculate direction to move apart
        dx = comp2.x - comp1.x
        dy = comp2.y - comp1.y
        distance = (dx**2 + dy**2)**0.5

        if distance < 0.001:
            # Components at same position, pick arbitrary direction
            dx, dy = 1.0, 0.0
            distance = 1.0

        # Normalize direction
        nx, ny = dx / distance, dy / distance

        # Determine move amount (add clearance + margin)
        # Use board's default clearance or 0.3mm
        clearance = self.board.default_clearance or 0.3
        move_amount = clearance + 0.5  # Add margin

        updates = []

        if strategy == "auto":
            # Check if either component is locked
            if comp1.locked and not comp2.locked:
                strategy = "move_second"
            elif comp2.locked and not comp1.locked:
                strategy = "move_first"
            else:
                strategy = "spread"

        if strategy == "move_first":
            new_x = comp1.x - nx * move_amount
            new_y = comp1.y - ny * move_amount
            updates.append({"ref": ref1, "x": new_x, "y": new_y})
            message = f"Moved {ref1} away from {ref2}"

        elif strategy == "move_second":
            new_x = comp2.x + nx * move_amount
            new_y = comp2.y + ny * move_amount
            updates.append({"ref": ref2, "x": new_x, "y": new_y})
            message = f"Moved {ref2} away from {ref1}"

        else:  # spread
            half_move = move_amount / 2
            updates.append({
                "ref": ref1,
                "x": comp1.x - nx * half_move,
                "y": comp1.y - ny * half_move
            })
            updates.append({
                "ref": ref2,
                "x": comp2.x + nx * half_move,
                "y": comp2.y + ny * half_move
            })
            message = f"Spread {ref1} and {ref2} apart"

        return True, message, updates

    def fix_edge_clearance_violation(
        self,
        violation: DRCViolationInfo
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Attempt to fix an edge clearance violation.

        Returns:
            (success, message, updates)
        """
        if not violation.items:
            return False, "No component specified in violation", []

        ref = violation.items[0]
        comp = self.board.components.get(ref)

        if not comp:
            return False, f"Component not found: {ref}", []

        if comp.locked:
            return False, f"Component {ref} is locked", []

        # Get board bounds
        outline = self.board.outline
        if not outline.has_outline:
            return False, "No board outline defined", []

        # Calculate center of board
        center_x = outline.origin_x + outline.width / 2
        center_y = outline.origin_y + outline.height / 2

        # Move component toward center
        dx = center_x - comp.x
        dy = center_y - comp.y
        distance = (dx**2 + dy**2)**0.5

        if distance < 0.001:
            return False, f"Component {ref} is already at board center", []

        # Move by edge clearance amount toward center
        edge_clearance = 0.5  # Conservative
        move_amount = edge_clearance + 0.5

        nx, ny = dx / distance, dy / distance
        new_x = comp.x + nx * move_amount
        new_y = comp.y + ny * move_amount

        return True, f"Moved {ref} toward board center", [
            {"ref": ref, "x": new_x, "y": new_y}
        ]

    def fix_violation(
        self,
        violation: DRCViolationInfo,
        strategy: str = "auto"
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Attempt to fix a violation using appropriate strategy.

        Returns:
            (success, message, updates)
        """
        rule = violation.rule.lower()

        if "clearance" in rule and "edge" not in rule:
            return self.fix_clearance_violation(violation, strategy)
        elif "edge" in rule:
            return self.fix_edge_clearance_violation(violation)
        else:
            return False, f"No auto-fix available for rule: {violation.rule}", []


# Global DRC runner instance
_drc_runner = DRCRunner()


def get_drc_runner() -> DRCRunner:
    """Get the global DRC runner instance."""
    return _drc_runner
