"""
Natural Language Constraint Parser

Extracts structured placement constraints from natural language input.
Uses a combination of regex patterns for common requests and LLM
interpretation for complex/ambiguous inputs.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from enum import Enum

from ..board.abstraction import Board
from ..placement.constraints import (
    PlacementConstraint,
    ProximityConstraint,
    EdgeConstraint,
    ZoneConstraint,
    GroupingConstraint,
    SeparationConstraint,
    FixedConstraint,
    ConstraintType,
)
from ..patterns import get_patterns


class ParseConfidence(Enum):
    """Confidence levels for parsed constraints."""
    HIGH = "high"       # Regex match, unambiguous
    MEDIUM = "medium"   # LLM parsed, likely correct
    LOW = "low"         # LLM parsed, uncertain


@dataclass
class ParsedConstraint:
    """A constraint parsed from natural language."""
    constraint: PlacementConstraint
    confidence: ParseConfidence
    source_text: str
    alternatives: List[PlacementConstraint] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing natural language input."""
    constraints: List[ParsedConstraint]
    unrecognized_text: str = ""
    warnings: List[str] = field(default_factory=list)


class ConstraintParser:
    """
    Parse natural language into placement constraints.

    Supports patterns like:
    - "Keep C1 close to U1"
    - "USB connector on left edge"
    - "Rotate J1 90 degrees"
    - "Keep analog and digital sections separate"
    - "Group all decoupling capacitors together"
    """

    # Pattern definitions: (regex, constraint_type, extractor_function)
    PATTERNS: List[Tuple[str, ConstraintType, Callable]] = []

    def __init__(self, board: Optional[Board] = None, patterns_config: Optional[str] = None):
        """
        Initialize parser.

        Args:
            board: Optional board for validating component references
            patterns_config: Optional path to custom component patterns YAML file
        """
        self.board = board
        self.patterns = get_patterns(patterns_config)
        self._setup_patterns()

    def _setup_patterns(self):
        """Set up regex patterns for constraint extraction."""
        self.PATTERNS = [
            # Proximity constraints
            (
                r"keep\s+(\w+)\s+(?:close\s+to|near|next\s+to)\s+(\w+)",
                ConstraintType.PROXIMITY,
                self._extract_proximity,
            ),
            (
                r"(\w+)\s+should\s+be\s+(?:close\s+to|near|next\s+to)\s+(\w+)",
                ConstraintType.PROXIMITY,
                self._extract_proximity,
            ),
            (
                r"place\s+(\w+)\s+(?:close\s+to|near|next\s+to|within\s+(\d+(?:\.\d+)?)\s*mm\s+of)\s+(\w+)",
                ConstraintType.PROXIMITY,
                self._extract_proximity_with_distance,
            ),

            # Edge placement constraints
            (
                r"(\w+)\s+(?:on|at)\s+(?:the\s+)?(left|right|top|bottom)\s+edge",
                ConstraintType.EDGE_PLACEMENT,
                self._extract_edge,
            ),
            (
                r"place\s+(\w+)\s+(?:on|at)\s+(?:the\s+)?(left|right|top|bottom)(?:\s+edge)?",
                ConstraintType.EDGE_PLACEMENT,
                self._extract_edge,
            ),
            (
                r"(left|right|top|bottom)\s+edge\s+(?:for\s+)?(\w+)",
                ConstraintType.EDGE_PLACEMENT,
                self._extract_edge_reverse,
            ),

            # Rotation constraints (captured but handled differently)
            (
                r"rotate\s+(\w+)\s+(\d+)\s*(?:degrees?|deg|°)?",
                ConstraintType.ORIENTATION,
                self._extract_rotation,
            ),

            # Grouping constraints
            (
                r"group\s+(.+?)\s+together",
                ConstraintType.GROUPING,
                self._extract_grouping,
            ),
            (
                r"keep\s+(.+?)\s+together",
                ConstraintType.GROUPING,
                self._extract_grouping,
            ),

            # Separation constraints
            (
                r"(?:keep|separate)\s+(.+?)\s+(?:and|from)\s+(.+?)\s+(?:separate|apart)",
                ConstraintType.SEPARATION,
                self._extract_separation,
            ),
            (
                r"separate\s+(.+?)\s+(?:and|from)\s+(.+)",
                ConstraintType.SEPARATION,
                self._extract_separation,
            ),

            # Fixed position constraints
            (
                r"fix\s+(\w+)\s+at\s+\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?",
                ConstraintType.FIXED,
                self._extract_fixed,
            ),
            (
                r"(\w+)\s+at\s+\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?",
                ConstraintType.FIXED,
                self._extract_fixed,
            ),
        ]

    def parse(self, text: str) -> ParseResult:
        """
        Parse natural language text into constraints.

        Args:
            text: Natural language constraint description

        Returns:
            ParseResult with extracted constraints
        """
        constraints: List[ParsedConstraint] = []
        warnings: List[str] = []

        # Normalize text
        text = text.lower().strip()

        # Track which parts of text we've processed
        processed_spans: List[Tuple[int, int]] = []

        # Try each pattern
        for pattern, ctype, extractor in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Skip if this span overlaps with already processed text
                # Check all overlap cases:
                # 1. start inside existing span: s <= start < e
                # 2. end inside existing span: s < end <= e
                # 3. new match fully contains existing: start <= s and e <= end
                # 4. new match fully contained by existing: s <= start and end <= e
                start, end = match.span()
                if any(
                    (s <= start < e) or  # start inside existing
                    (s < end <= e) or    # end inside existing
                    (start <= s and e <= end) or  # new contains existing
                    (s <= start and end <= e)     # existing contains new
                    for s, e in processed_spans
                ):
                    continue

                try:
                    constraint = extractor(match)
                    if constraint:
                        # Validate references if board is available
                        if self.board:
                            valid, warning = self._validate_constraint(constraint)
                            if warning:
                                warnings.append(warning)
                            if not valid:
                                continue

                        constraints.append(ParsedConstraint(
                            constraint=constraint,
                            confidence=ParseConfidence.HIGH,
                            source_text=match.group(0),
                        ))
                        processed_spans.append((start, end))
                except Exception as e:
                    warnings.append(f"Error parsing '{match.group(0)}': {e}")

        # Find unrecognized text
        unrecognized_parts = []
        last_end = 0
        for start, end in sorted(processed_spans):
            if start > last_end:
                part = text[last_end:start].strip()
                if part and len(part) > 2:
                    unrecognized_parts.append(part)
            last_end = end
        if last_end < len(text):
            part = text[last_end:].strip()
            if part and len(part) > 2:
                unrecognized_parts.append(part)

        unrecognized = " ... ".join(unrecognized_parts)

        return ParseResult(
            constraints=constraints,
            unrecognized_text=unrecognized,
            warnings=warnings,
        )

    def parse_interactive(self, text: str) -> Tuple[List[PlacementConstraint], str]:
        """
        Parse text and return constraints plus a summary message.

        Returns:
            (constraints, summary_message)
        """
        result = self.parse(text)

        constraints = [pc.constraint for pc in result.constraints]

        # Build summary
        lines = []
        if constraints:
            lines.append(f"Extracted {len(constraints)} constraint(s):")
            for pc in result.constraints:
                conf_str = {"high": "", "medium": " (medium confidence)",
                           "low": " (low confidence)"}
                lines.append(f"  - {pc.constraint.description}{conf_str[pc.confidence.value]}")

        if result.unrecognized_text:
            lines.append(f"\nUnrecognized text: \"{result.unrecognized_text}\"")
            lines.append("Try rephrasing or use more specific component references.")

        if result.warnings:
            lines.append("\nWarnings:")
            for w in result.warnings:
                lines.append(f"  - {w}")

        return constraints, "\n".join(lines) if lines else "No constraints found."

    # --- Extractor functions ---

    def _extract_proximity(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract proximity constraint from regex match."""
        target = match.group(1).upper()
        anchor = match.group(2).upper()

        return ProximityConstraint(
            target_ref=target,
            anchor_ref=anchor,
            max_distance=5.0,
            ideal_distance=2.0,
            description=f"Keep {target} close to {anchor}",
            source_text=match.group(0),
        )

    def _extract_proximity_with_distance(self, match: re.Match
                                          ) -> Optional[PlacementConstraint]:
        """Extract proximity constraint with specific distance."""
        target = match.group(1).upper()
        distance = float(match.group(2)) if match.group(2) else 5.0
        anchor = match.group(3).upper()

        return ProximityConstraint(
            target_ref=target,
            anchor_ref=anchor,
            max_distance=distance,
            ideal_distance=distance * 0.5,
            description=f"Keep {target} within {distance}mm of {anchor}",
            source_text=match.group(0),
        )

    def _extract_edge(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract edge placement constraint."""
        component = match.group(1).upper()
        edge = match.group(2).lower()

        return EdgeConstraint(
            component_ref=component,
            edge=edge,
            offset=2.0,
            description=f"Place {component} on {edge} edge",
            source_text=match.group(0),
        )

    def _extract_edge_reverse(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract edge placement from reversed pattern."""
        edge = match.group(1).lower()
        component = match.group(2).upper()

        return EdgeConstraint(
            component_ref=component,
            edge=edge,
            offset=2.0,
            description=f"Place {component} on {edge} edge",
            source_text=match.group(0),
        )

    def _extract_rotation(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract rotation constraint (as fixed constraint with rotation only)."""
        component = match.group(1).upper()
        angle = float(match.group(2))

        return FixedConstraint(
            component_ref=component,
            rotation=angle,
            rotation_only=True,  # Only constrain rotation, not position
            description=f"Rotate {component} to {angle}°",
            source_text=match.group(0),
        )

    def _extract_grouping(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract grouping constraint."""
        components_text = match.group(1)

        # Parse component list
        components = self._parse_component_list(components_text)

        if len(components) < 2:
            return None

        return GroupingConstraint(
            components=components,
            max_spread=15.0,
            description=f"Group {', '.join(components)} together",
            source_text=match.group(0),
        )

    def _extract_separation(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract separation constraint."""
        group_a_text = match.group(1)
        group_b_text = match.group(2)

        group_a = self._parse_component_list(group_a_text)
        group_b = self._parse_component_list(group_b_text)

        if not group_a or not group_b:
            return None

        return SeparationConstraint(
            group_a=group_a,
            group_b=group_b,
            min_separation=10.0,
            description=f"Separate {group_a_text} from {group_b_text}",
            source_text=match.group(0),
        )

    def _extract_fixed(self, match: re.Match) -> Optional[PlacementConstraint]:
        """Extract fixed position constraint."""
        component = match.group(1).upper()
        x = float(match.group(2))
        y = float(match.group(3))

        return FixedConstraint(
            component_ref=component,
            x=x,
            y=y,
            description=f"Fix {component} at ({x}, {y})",
            source_text=match.group(0),
        )

    # --- Helper functions ---

    def _parse_component_list(self, text: str) -> List[str]:
        """Parse a list of components from text."""
        components = []

        # Handle common patterns
        # "all capacitors" -> find all C* components
        if "all" in text.lower():
            if "capacitor" in text.lower() and self.board:
                components = [c.reference for c in self.board.get_components_by_prefix('C')]
            elif "resistor" in text.lower() and self.board:
                components = [c.reference for c in self.board.get_components_by_prefix('R')]
            elif "inductor" in text.lower() and self.board:
                components = [c.reference for c in self.board.get_components_by_prefix('L')]
            elif "decoupling" in text.lower() and self.board:
                # Decoupling caps are typically small value caps near ICs
                components = [c.reference for c in self.board.get_components_by_prefix('C')]

        # Handle "analog" / "digital" zones - resolve to actual components
        elif "analog" in text.lower():
            components = self._get_analog_components()
        elif "digital" in text.lower():
            components = self._get_digital_components()

        # Handle explicit list like "C1, C2, C3" or "C1 and C2"
        else:
            # Look for component references
            refs = re.findall(r'\b([A-Z]+\d+)\b', text.upper())
            components = refs

        return components

    def _get_analog_components(self) -> List[str]:
        """Get components that are likely analog (op-amps, analog sensors, etc.)."""
        if not self.board:
            return []

        analog_refs = []
        analog_patterns = self.patterns.analog_patterns

        for ref, comp in self.board.components.items():
            fp = comp.footprint.upper()
            value = comp.value.upper()

            # Check for analog patterns
            for pattern in analog_patterns:
                if re.search(pattern, value) or re.search(pattern, fp):
                    analog_refs.append(ref)
                    break

        # If no specific analog components found, return components near analog ICs
        if not analog_refs:
            # Fall back to all 'U' components that might be analog
            analog_refs = [ref for ref in self.board.components if ref.startswith('U')]

        return analog_refs

    def _get_digital_components(self) -> List[str]:
        """Get components that are likely digital (MCUs, logic ICs, etc.)."""
        if not self.board:
            return []

        digital_refs = []
        digital_patterns = self.patterns.digital_patterns

        for ref, comp in self.board.components.items():
            fp = comp.footprint.upper()
            value = comp.value.upper()

            # Check for digital patterns
            for pattern in digital_patterns:
                if re.search(pattern, value) or re.search(pattern, fp):
                    digital_refs.append(ref)
                    break

        # If no specific digital components found, fall back
        if not digital_refs:
            digital_refs = [ref for ref in self.board.components if ref.startswith('U')]

        return digital_refs

    def _validate_constraint(self, constraint: PlacementConstraint
                              ) -> Tuple[bool, Optional[str]]:
        """Validate that constraint references exist in board."""
        if not self.board:
            return (True, None)

        # Check component references based on constraint type
        refs_to_check = []

        if isinstance(constraint, ProximityConstraint):
            refs_to_check = [constraint.target_ref, constraint.anchor_ref]
        elif isinstance(constraint, EdgeConstraint):
            refs_to_check = [constraint.component_ref]
        elif isinstance(constraint, FixedConstraint):
            refs_to_check = [constraint.component_ref]
        elif isinstance(constraint, GroupingConstraint):
            refs_to_check = constraint.components
        elif isinstance(constraint, SeparationConstraint):
            refs_to_check = constraint.group_a + constraint.group_b

        # Check each reference
        missing = []
        for ref in refs_to_check:
            if ref not in self.board.components:
                missing.append(ref)

        if missing:
            return (False, f"Unknown component(s): {', '.join(missing)}")

        return (True, None)


class ModificationHandler:
    """Handle modification requests for existing placements."""

    MODIFICATION_PATTERNS = [
        # Move with target: "move C1 closer to U1" or "move C1 away from U1"
        (r"move\s+(\w+)\s+(closer\s+to|away\s+from)\s+(\w+)",
         "move_relative"),
        # Move directional: "move C1 left/right/up/down"
        (r"move\s+(\w+)\s+(left|right|up|down)",
         "move"),
        (r"rotate\s+(\w+)\s+(\d+)\s*(?:degrees?)?",
         "rotate"),
        (r"swap\s+(\w+)\s+(?:and|with)\s+(\w+)",
         "swap"),
        (r"flip\s+(\w+)",
         "flip"),
    ]

    def __init__(self, board: Board):
        self.board = board

    def parse_modification(self, text: str) -> Optional[Dict]:
        """Parse a modification request."""
        text = text.lower().strip()

        for pattern, mod_type in self.MODIFICATION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._extract_modification(mod_type, match, text)

        return None

    def _extract_modification(self, mod_type: str, match: re.Match,
                               full_text: str) -> Dict:
        """Extract modification details from match."""
        if mod_type == "move":
            # Simple directional move: left/right/up/down
            return {
                "type": "move",
                "component": match.group(1).upper(),
                "direction": match.group(2),
            }
        elif mod_type == "move_relative":
            # Move relative to another component: closer to / away from
            direction = match.group(2).lower()
            return {
                "type": "move",
                "component": match.group(1).upper(),
                "direction": "closer" if "closer" in direction else "away",
                "target": match.group(3).upper(),
            }
        elif mod_type == "rotate":
            return {
                "type": "rotate",
                "component": match.group(1).upper(),
                "angle": float(match.group(2)),
            }
        elif mod_type == "swap":
            return {
                "type": "swap",
                "component1": match.group(1).upper(),
                "component2": match.group(2).upper(),
            }
        elif mod_type == "flip":
            return {
                "type": "flip",
                "component": match.group(1).upper(),
            }

        return {"type": mod_type}

    def apply_modification(self, mod: Dict) -> bool:
        """Apply a modification to the board."""
        mod_type = mod.get("type")

        if mod_type == "rotate":
            comp = self.board.get_component(mod["component"])
            if comp:
                comp.rotation = (comp.rotation + mod["angle"]) % 360
                return True

        elif mod_type == "swap":
            comp1 = self.board.get_component(mod["component1"])
            comp2 = self.board.get_component(mod["component2"])
            if comp1 and comp2:
                comp1.x, comp2.x = comp2.x, comp1.x
                comp1.y, comp2.y = comp2.y, comp1.y
                return True

        elif mod_type == "flip":
            comp = self.board.get_component(mod["component"])
            if comp:
                # Flip layer
                from ..board.abstraction import Layer
                if comp.layer == Layer.TOP_COPPER:
                    comp.layer = Layer.BOTTOM_COPPER
                else:
                    comp.layer = Layer.TOP_COPPER
                return True

        elif mod_type == "move":
            comp = self.board.get_component(mod["component"])
            if not comp:
                return False

            direction = mod.get("direction", "")
            target_ref = mod.get("target")  # Target component for relative moves
            move_amount = 5.0  # Default move amount in mm

            if direction == "left":
                comp.x -= move_amount
            elif direction == "right":
                comp.x += move_amount
            elif direction == "up":
                comp.y -= move_amount  # Y decreases going up in KiCad
            elif direction == "down":
                comp.y += move_amount
            elif direction == "closer" and target_ref:
                # Move closer to target component
                target = self.board.get_component(target_ref)
                if target:
                    # Move 20% closer to target
                    dx = target.x - comp.x
                    dy = target.y - comp.y
                    comp.x += dx * 0.2
                    comp.y += dy * 0.2
                else:
                    return False
            elif direction == "away" and target_ref:
                # Move away from target component
                target = self.board.get_component(target_ref)
                if target:
                    # Move 20% of move_amount away from target
                    dx = comp.x - target.x
                    dy = comp.y - target.y
                    dist = (dx*dx + dy*dy) ** 0.5
                    if dist > 0.1:
                        comp.x += (dx / dist) * move_amount
                        comp.y += (dy / dist) * move_amount
                else:
                    return False
            return True

        return False
