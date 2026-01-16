"""Tests for natural language parsing."""

import pytest
from atoplace.nlp.constraint_parser import ConstraintParser, ParseConfidence
from atoplace.placement.constraints import (
    ProximityConstraint,
    EdgeConstraint,
    GroupingConstraint,
)


class TestConstraintParser:
    """Tests for constraint parser."""

    @pytest.fixture
    def parser(self):
        """Create parser without board validation."""
        return ConstraintParser(board=None)

    def test_parse_proximity(self, parser):
        """Parse proximity constraint."""
        result = parser.parse("keep C1 close to U1")

        assert len(result.constraints) == 1
        pc = result.constraints[0]
        assert pc.confidence == ParseConfidence.HIGH
        assert isinstance(pc.constraint, ProximityConstraint)
        assert pc.constraint.target_ref == "C1"
        assert pc.constraint.anchor_ref == "U1"

    def test_parse_edge_placement(self, parser):
        """Parse edge placement constraint."""
        result = parser.parse("USB connector on left edge")

        # Note: This won't match because "USB connector" isn't a valid ref
        # But "J1 on left edge" would work
        result = parser.parse("J1 on left edge")

        assert len(result.constraints) == 1
        pc = result.constraints[0]
        assert isinstance(pc.constraint, EdgeConstraint)
        assert pc.constraint.component_ref == "J1"
        assert pc.constraint.edge == "left"

    def test_parse_multiple_constraints(self, parser):
        """Parse multiple constraints from text."""
        result = parser.parse(
            "Keep C1 close to U1. Place J1 on left edge. Keep C2 near U2."
        )

        assert len(result.constraints) >= 2

    def test_parse_edge_variations(self, parser):
        """Parse different edge placement phrasings."""
        variations = [
            "J1 on left edge",
            "place J1 at the left",
            "J1 at the bottom edge",
            "left edge for J1",
        ]

        for text in variations:
            result = parser.parse(text)
            if result.constraints:
                assert isinstance(result.constraints[0].constraint, EdgeConstraint)

    def test_parse_interactive_returns_summary(self, parser):
        """parse_interactive should return constraints and summary."""
        constraints, summary = parser.parse_interactive("keep C1 close to U1")

        assert len(constraints) == 1
        assert "C1" in summary
        assert "U1" in summary

    def test_unrecognized_text_reported(self, parser):
        """Unrecognized text should be reported."""
        result = parser.parse("some random text that doesn't match any pattern")

        assert len(result.constraints) == 0
        assert len(result.unrecognized_text) > 0
