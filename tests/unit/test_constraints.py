"""Tests for placement constraints."""

import pytest
from atoplace.placement.constraints import (
    ProximityConstraint,
    EdgeConstraint,
    GroupingConstraint,
    ConstraintSolver,
)
from atoplace.board.abstraction import Board, Component, BoardOutline


@pytest.fixture
def simple_board():
    """Create a simple test board."""
    board = Board(
        name="test",
        outline=BoardOutline(width=100, height=100, origin_x=0, origin_y=0),
    )

    # Add some components
    board.add_component(Component(
        reference="U1",
        footprint="Package_QFP:LQFP-48",
        x=50, y=50,
        width=10, height=10,
    ))
    board.add_component(Component(
        reference="C1",
        footprint="Capacitor_SMD:C_0402",
        x=45, y=55,
        width=1, height=0.5,
    ))
    board.add_component(Component(
        reference="C2",
        footprint="Capacitor_SMD:C_0402",
        x=80, y=80,
        width=1, height=0.5,
    ))
    board.add_component(Component(
        reference="J1",
        footprint="Connector_USB:USB_C",
        x=50, y=10,
        width=9, height=7,
    ))

    return board


class TestProximityConstraint:
    """Tests for proximity constraints."""

    def test_satisfied_when_close(self, simple_board):
        """Constraint is satisfied when components are within max_distance."""
        constraint = ProximityConstraint(
            target_ref="C1",
            anchor_ref="U1",
            max_distance=10.0,
        )

        satisfied, violation = constraint.is_satisfied(simple_board)
        assert satisfied
        assert violation == 0.0

    def test_violated_when_far(self, simple_board):
        """Constraint is violated when components exceed max_distance."""
        constraint = ProximityConstraint(
            target_ref="C2",
            anchor_ref="U1",
            max_distance=10.0,
        )

        satisfied, violation = constraint.is_satisfied(simple_board)
        assert not satisfied
        assert violation > 0


class TestEdgeConstraint:
    """Tests for edge placement constraints."""

    def test_force_pulls_to_edge(self, simple_board):
        """Force should pull component toward specified edge."""
        constraint = EdgeConstraint(
            component_ref="J1",
            edge="left",
            offset=5.0,
        )

        fx, fy = constraint.calculate_force(simple_board, "J1", strength=1.0)

        # Should pull left (negative x)
        assert fx < 0
        assert abs(fy) < 0.1  # Minimal vertical force


class TestGroupingConstraint:
    """Tests for grouping constraints."""

    def test_satisfied_when_grouped(self, simple_board):
        """Constraint satisfied when components are within spread."""
        # Move C2 closer
        simple_board.move_component("C2", 55, 55)

        constraint = GroupingConstraint(
            components=["C1", "C2"],
            max_spread=20.0,
        )

        satisfied, _ = constraint.is_satisfied(simple_board)
        assert satisfied

    def test_violated_when_spread(self, simple_board):
        """Constraint violated when components are too spread out."""
        constraint = GroupingConstraint(
            components=["C1", "C2"],
            max_spread=10.0,
        )

        satisfied, violation = constraint.is_satisfied(simple_board)
        assert not satisfied
        assert violation > 0


class TestConstraintSolver:
    """Tests for constraint solver."""

    def test_evaluate_all(self, simple_board):
        """Solver should evaluate all constraints."""
        solver = ConstraintSolver(simple_board)

        solver.add_constraint(ProximityConstraint(
            target_ref="C1",
            anchor_ref="U1",
            max_distance=10.0,
        ))
        solver.add_constraint(EdgeConstraint(
            component_ref="J1",
            edge="top",
            offset=5.0,
        ))

        results = solver.evaluate_all()
        assert len(results) == 2

    def test_constraint_score(self, simple_board):
        """Score should reflect constraint satisfaction."""
        solver = ConstraintSolver(simple_board)

        # Add a satisfied constraint
        solver.add_constraint(ProximityConstraint(
            target_ref="C1",
            anchor_ref="U1",
            max_distance=20.0,
        ))

        score = solver.get_constraint_score()
        assert score == 1.0  # All satisfied
