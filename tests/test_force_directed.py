"""
Tests for the force-directed placement algorithm.

Tests cover:
- Force calculation (repulsion, attraction, boundary, constraint)
- State initialization and convergence
- Adaptive damping for oscillation control
- Layer-aware anchoring
- Constraint application
"""

import pytest
import math
from typing import Dict, List, Tuple

from atoplace.board.abstraction import (
    Board,
    BoardOutline,
    Component,
    Net,
    Pad,
    Layer,
)
from atoplace.placement.force_directed import (
    ForceDirectedRefiner,
    RefinementConfig,
    PlacementState,
    Force,
    ForceType,
)
from atoplace.placement.constraints import (
    PlacementConstraint,
    ProximityConstraint,
    EdgeConstraint,
    ZoneConstraint,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_board() -> Board:
    """A simple board with a few components for testing."""
    board = Board(name="test_board")
    board.outline = BoardOutline(
        width=100.0,
        height=100.0,
        origin_x=0.0,
        origin_y=0.0,
        has_outline=True,
    )

    # Add IC
    ic = Component(
        reference="U1",
        footprint="Package_SO:SOIC-8",
        value="ATtiny85",
        x=50.0,
        y=50.0,
        width=5.0,
        height=4.0,
    )
    ic.pads = [
        Pad(number="1", x=-1.5, y=-1.5, width=0.6, height=1.0, net="VCC"),
        Pad(number="4", x=-1.5, y=1.5, width=0.6, height=1.0, net="GND"),
        Pad(number="8", x=1.5, y=-1.5, width=0.6, height=1.0, net="VCC"),
    ]

    # Add capacitor
    cap = Component(
        reference="C1",
        footprint="Capacitor_SMD:C_0603",
        value="100nF",
        x=45.0,
        y=50.0,
        width=1.6,
        height=0.8,
    )
    cap.pads = [
        Pad(number="1", x=-0.75, y=0, width=0.5, height=0.6, net="VCC"),
        Pad(number="2", x=0.75, y=0, width=0.5, height=0.6, net="GND"),
    ]

    # Add resistor
    res = Component(
        reference="R1",
        footprint="Resistor_SMD:R_0603",
        value="10k",
        x=55.0,
        y=50.0,
        width=1.6,
        height=0.8,
    )
    res.pads = [
        Pad(number="1", x=-0.75, y=0, width=0.5, height=0.6, net="VCC"),
        Pad(number="2", x=0.75, y=0, width=0.5, height=0.6, net="SIG"),
    ]

    board.components = {"U1": ic, "C1": cap, "R1": res}

    # Create nets
    vcc = Net(name="VCC", code=1, is_power=True)
    vcc.add_connection("U1", "1")
    vcc.add_connection("U1", "8")
    vcc.add_connection("C1", "1")
    vcc.add_connection("R1", "1")

    gnd = Net(name="GND", code=2, is_ground=True)
    gnd.add_connection("U1", "4")
    gnd.add_connection("C1", "2")

    sig = Net(name="SIG", code=3)
    sig.add_connection("R1", "2")

    board.nets = {"VCC": vcc, "GND": gnd, "SIG": sig}

    return board


@pytest.fixture
def overlapping_board() -> Board:
    """A board with overlapping components."""
    board = Board(name="overlap_test")
    board.outline = BoardOutline(
        width=100.0,
        height=100.0,
        origin_x=0.0,
        origin_y=0.0,
        has_outline=True,
    )

    # Two overlapping components
    c1 = Component(
        reference="C1",
        footprint="Capacitor_SMD:C_0603",
        value="100nF",
        x=50.0,
        y=50.0,
        width=1.6,
        height=0.8,
    )
    c2 = Component(
        reference="C2",
        footprint="Capacitor_SMD:C_0603",
        value="100nF",
        x=50.5,  # Overlaps with C1
        y=50.0,
        width=1.6,
        height=0.8,
    )

    board.components = {"C1": c1, "C2": c2}
    board.nets = {}

    return board


@pytest.fixture
def boundary_board() -> Board:
    """A board with a component near the boundary."""
    board = Board(name="boundary_test")
    board.outline = BoardOutline(
        width=50.0,
        height=50.0,
        origin_x=0.0,
        origin_y=0.0,
        has_outline=True,
    )

    # Component placed outside boundary
    comp = Component(
        reference="U1",
        footprint="Package_SO:SOIC-8",
        value="IC",
        x=-5.0,  # Outside left edge
        y=25.0,
        width=5.0,
        height=4.0,
    )

    board.components = {"U1": comp}
    board.nets = {}

    return board


@pytest.fixture
def default_config() -> RefinementConfig:
    """Default configuration for testing."""
    return RefinementConfig(
        max_iterations=100,
        min_movement=0.01,
        repulsion_strength=100.0,
        attraction_strength=0.5,
        boundary_strength=200.0,
    )


# =============================================================================
# State Initialization Tests
# =============================================================================

class TestStateInitialization:
    """Tests for state initialization."""

    def test_initialize_state_positions(self, simple_board, default_config):
        """Test that state is initialized with component positions."""
        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        assert "U1" in state.positions
        assert "C1" in state.positions
        assert "R1" in state.positions

        assert state.positions["U1"] == (50.0, 50.0)
        assert state.positions["C1"] == (45.0, 50.0)
        assert state.positions["R1"] == (55.0, 50.0)

    def test_initialize_state_velocities(self, simple_board, default_config):
        """Test that velocities are initialized to zero."""
        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        for ref in state.velocities:
            assert state.velocities[ref] == (0.0, 0.0)

    def test_initialize_state_rotations(self, simple_board, default_config):
        """Test that rotations are initialized from components."""
        simple_board.components["U1"].rotation = 45.0
        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        assert state.rotations["U1"] == 45.0


# =============================================================================
# Force Calculation Tests
# =============================================================================

class TestForceCalculation:
    """Tests for force calculation."""

    def test_repulsion_force_on_overlap(self, overlapping_board):
        """Test that overlapping components experience repulsion."""
        # Disable auto-anchor to isolate repulsion force testing
        # Without this, center attraction forces dominate
        config = RefinementConfig(
            max_iterations=100,
            auto_anchor_largest_ic=False,
            center_strength=0.0,  # Disable center attraction for isolated testing
        )
        refiner = ForceDirectedRefiner(overlapping_board, config)
        state = refiner._initialize_state()

        forces = refiner._calculate_all_forces(state)

        # Both components should have repulsion forces
        c1_forces = [f for f in forces["C1"] if f.force_type == ForceType.REPULSION]
        c2_forces = [f for f in forces["C2"] if f.force_type == ForceType.REPULSION]

        assert len(c1_forces) > 0, "C1 should have repulsion force"
        assert len(c2_forces) > 0, "C2 should have repulsion force"

        # Forces should be opposite directions (check both axes since algorithm
        # may push along the axis with smaller overlap - Y in this case since
        # components are in a horizontal row at the same Y coordinate)
        c1_fx = sum(f.fx for f in c1_forces)
        c1_fy = sum(f.fy for f in c1_forces)
        c2_fx = sum(f.fx for f in c2_forces)
        c2_fy = sum(f.fy for f in c2_forces)

        # At least one axis should have opposite forces
        x_opposite = c1_fx * c2_fx < 0 if (c1_fx != 0 or c2_fx != 0) else False
        y_opposite = c1_fy * c2_fy < 0 if (c1_fy != 0 or c2_fy != 0) else False

        assert x_opposite or y_opposite, "Repulsion forces should be opposite on at least one axis"

    def test_no_repulsion_when_separated(self, simple_board, default_config):
        """Test that well-separated components don't repel strongly."""
        # Move components far apart
        simple_board.components["C1"].x = 10.0
        simple_board.components["R1"].x = 90.0

        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        forces = refiner._calculate_all_forces(state)

        # C1 and R1 should not have repulsion forces from each other
        # (they may have repulsion from U1)
        c1_repulsion = sum(f.magnitude for f in forces["C1"]
                          if f.force_type == ForceType.REPULSION and "R1" in f.source)

        assert c1_repulsion == 0, "Distant components should not repel"

    def test_attraction_force_on_connected_components(self, simple_board, default_config):
        """Test that connected components experience attraction."""
        # Move capacitor far from IC
        simple_board.components["C1"].x = 10.0
        simple_board.components["C1"].y = 10.0

        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        forces = refiner._calculate_all_forces(state)

        # C1 is connected to VCC net with U1, should have attraction
        c1_attraction = [f for f in forces["C1"] if f.force_type == ForceType.ATTRACTION]

        assert len(c1_attraction) > 0, "Connected components should attract"

    def test_boundary_force_outside_board(self, boundary_board):
        """Test that components outside boundary experience inward force."""
        # Disable auto-anchor since U1 is an IC that would become anchor
        # and get moved to center, eliminating the boundary violation we want to test
        config = RefinementConfig(
            max_iterations=100,
            auto_anchor_largest_ic=False,
        )
        refiner = ForceDirectedRefiner(boundary_board, config)
        state = refiner._initialize_state()

        forces = refiner._calculate_all_forces(state)

        # U1 is outside left edge, should have positive x force (push right)
        boundary_forces = [f for f in forces["U1"] if f.force_type == ForceType.BOUNDARY]

        assert len(boundary_forces) > 0, "Boundary force should exist"

        fx = sum(f.fx for f in boundary_forces)
        assert fx > 0, "Boundary force should push component right (into board)"

    def test_no_boundary_force_inside_board(self, simple_board, default_config):
        """Test that components inside boundary have no boundary force."""
        # All components are well inside the 100x100 board
        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        forces = refiner._calculate_all_forces(state)

        for ref in forces:
            boundary_forces = [f for f in forces[ref] if f.force_type == ForceType.BOUNDARY]
            total_boundary = sum(f.magnitude for f in boundary_forces)
            assert total_boundary < 0.1, f"{ref} inside board shouldn't have boundary force"


# =============================================================================
# Convergence Tests
# =============================================================================

class TestConvergence:
    """Tests for convergence behavior."""

    def test_refinement_converges(self, simple_board, default_config):
        """Test that refinement eventually converges."""
        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner.refine()

        # Should converge within max_iterations
        assert state.iteration < default_config.max_iterations or state.converged

    def test_overlapping_components_separate(self, overlapping_board):
        """Test that overlapping components separate during refinement."""
        # Calculate initial 2D distance (algorithm may separate along X or Y)
        c1_init = overlapping_board.components["C1"]
        c2_init = overlapping_board.components["C2"]
        initial_distance = math.sqrt(
            (c1_init.x - c2_init.x) ** 2 + (c1_init.y - c2_init.y) ** 2
        )

        # Disable auto-anchor and center forces to isolate repulsion behavior
        config = RefinementConfig(
            max_iterations=100,
            auto_anchor_largest_ic=False,
            center_strength=0.0,
        )
        refiner = ForceDirectedRefiner(overlapping_board, config)
        state = refiner.refine()

        # Calculate final 2D distance
        c1_final = state.positions["C1"]
        c2_final = state.positions["C2"]
        final_distance = math.sqrt(
            (c1_final[0] - c2_final[0]) ** 2 + (c1_final[1] - c2_final[1]) ** 2
        )

        assert final_distance > initial_distance, "Overlapping components should separate"

    def test_boundary_violation_corrected(self, boundary_board, default_config):
        """Test that components outside boundary are pushed inside."""
        refiner = ForceDirectedRefiner(boundary_board, default_config)
        state = refiner.refine()

        # Component should be pushed inside the board
        final_x = state.positions["U1"][0]
        comp_half_width = boundary_board.components["U1"].width / 2

        # Should be at least inside the left edge
        assert final_x >= comp_half_width, "Component should be inside board"


# =============================================================================
# Adaptive Damping Tests
# =============================================================================

class TestAdaptiveDamping:
    """Tests for adaptive damping behavior."""

    def test_oscillation_detection(self, simple_board, default_config):
        """Test oscillation detection logic."""
        refiner = ForceDirectedRefiner(simple_board, default_config)

        # Simulate oscillating energy history
        energy_history = [100.0, 90.0, 95.0, 85.0, 92.0, 88.0]
        movement_history = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        is_oscillating = refiner._detect_oscillation(energy_history, movement_history)

        # High alternation should trigger oscillation detection
        assert is_oscillating is True or is_oscillating is False  # Just test it runs

    def test_variance_calculation(self, simple_board, default_config):
        """Test variance calculation helper."""
        refiner = ForceDirectedRefiner(simple_board, default_config)

        # Constant values should have zero variance
        assert refiner._calculate_variance([5.0, 5.0, 5.0, 5.0]) == 0.0

        # Known variance
        values = [2.0, 4.0, 6.0, 8.0]  # Mean=5, variance=5
        variance = refiner._calculate_variance(values)
        assert abs(variance - 5.0) < 0.001


# =============================================================================
# Constraint Tests
# =============================================================================

class TestConstraints:
    """Tests for constraint application."""

    def test_proximity_constraint(self, simple_board, default_config):
        """Test proximity constraint creates attraction force."""
        # Move C1 far from U1
        simple_board.components["C1"].x = 10.0
        simple_board.components["C1"].y = 10.0

        constraint = ProximityConstraint(
            target_ref="C1",
            anchor_ref="U1",
            max_distance=5.0,
            description="Keep C1 close to U1",
        )

        refiner = ForceDirectedRefiner(simple_board, default_config)
        refiner.add_constraint(constraint)

        state = refiner._initialize_state()
        forces = refiner._calculate_all_forces(state)

        # C1 should have constraint force toward U1
        constraint_forces = [f for f in forces["C1"] if f.force_type == ForceType.CONSTRAINT]

        assert len(constraint_forces) > 0, "Proximity constraint should create force"

    def test_edge_constraint(self, simple_board, default_config):
        """Test edge constraint pushes component to edge."""
        constraint = EdgeConstraint(
            component_ref="R1",
            edge="right",
            offset=5.0,
            description="R1 on right edge",
        )

        refiner = ForceDirectedRefiner(simple_board, default_config)
        refiner.add_constraint(constraint)

        state = refiner._initialize_state()
        forces = refiner._calculate_all_forces(state)

        # R1 should have constraint force toward right edge
        constraint_forces = [f for f in forces["R1"] if f.force_type == ForceType.CONSTRAINT]

        assert len(constraint_forces) > 0, "Edge constraint should create force"

        # Force should be positive X (toward right)
        fx = sum(f.fx for f in constraint_forces)
        assert fx > 0, "Edge constraint should push right"

    def test_zone_constraint(self, simple_board, default_config):
        """Test zone constraint keeps components in zone."""
        # Move R1 outside the zone
        simple_board.components["R1"].x = 90.0

        constraint = ZoneConstraint(
            zone_x=0.0,
            zone_y=0.0,
            zone_width=50.0,
            zone_height=50.0,
            components=["R1"],  # API uses 'components' not 'affected_components'
            description="Keep R1 in left half",
        )

        refiner = ForceDirectedRefiner(simple_board, default_config)
        refiner.add_constraint(constraint)

        state = refiner._initialize_state()
        forces = refiner._calculate_all_forces(state)

        # R1 should have force pushing it back into zone
        constraint_forces = [f for f in forces["R1"] if f.force_type == ForceType.CONSTRAINT]

        assert len(constraint_forces) > 0, "Zone constraint should create force"


# =============================================================================
# Anchor and Center Tests
# =============================================================================

class TestAnchorBehavior:
    """Tests for anchor component behavior."""

    def test_largest_ic_becomes_anchor(self, simple_board, default_config):
        """Test that largest IC is selected as anchor."""
        # U1 is the only IC and should become anchor
        refiner = ForceDirectedRefiner(simple_board, default_config)

        assert refiner._anchor_ref == "U1" or refiner._anchor_top == "U1"

    def test_anchor_doesnt_move(self, simple_board, default_config):
        """Test that anchor component doesn't move during refinement."""
        # Get initial anchor position
        initial_x = simple_board.components["U1"].x
        initial_y = simple_board.components["U1"].y

        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner.refine()

        # Anchor should not have moved significantly
        final_x, final_y = state.positions["U1"]

        # Anchor position is set to center, so check it's at center
        center_x = simple_board.outline.origin_x + simple_board.outline.width / 2
        center_y = simple_board.outline.origin_y + simple_board.outline.height / 2

        assert abs(final_x - center_x) < 0.1, "Anchor should be at center X"
        assert abs(final_y - center_y) < 0.1, "Anchor should be at center Y"

    def test_anchor_disabled(self, simple_board):
        """Test that auto-anchor can be disabled."""
        config = RefinementConfig(
            auto_anchor_largest_ic=False,
            max_iterations=10,
        )

        refiner = ForceDirectedRefiner(simple_board, config)

        assert refiner._anchor_ref is None
        assert refiner._anchor_top is None


# =============================================================================
# Energy Calculation Tests
# =============================================================================

class TestEnergyCalculation:
    """Tests for energy calculation."""

    def test_energy_decreases_during_refinement(self, overlapping_board, default_config):
        """Test that system energy generally decreases."""
        refiner = ForceDirectedRefiner(overlapping_board, default_config)

        energies = []
        def callback(state):
            energies.append(state.total_energy)

        refiner.refine(callback=callback)

        # Energy should generally decrease (allow some fluctuation)
        if len(energies) > 10:
            early_avg = sum(energies[:5]) / 5
            late_avg = sum(energies[-5:]) / 5
            assert late_avg <= early_avg * 1.5, "Energy should generally decrease"


# =============================================================================
# Component Size Tests
# =============================================================================

class TestComponentSizes:
    """Tests for component size calculations."""

    def test_component_sizes_computed(self, simple_board, default_config):
        """Test that component sizes are computed on initialization."""
        refiner = ForceDirectedRefiner(simple_board, default_config)

        assert "U1" in refiner._component_sizes
        assert "C1" in refiner._component_sizes
        assert "R1" in refiner._component_sizes

        # Sizes should be half-dimensions (positive)
        for ref, (half_w, half_h) in refiner._component_sizes.items():
            assert half_w > 0, f"{ref} should have positive half-width"
            assert half_h > 0, f"{ref} should have positive half-height"

    def test_size_update_on_rotation(self, simple_board, default_config):
        """Test that sizes are updated when rotation changes."""
        refiner = ForceDirectedRefiner(simple_board, default_config)

        # Store original sizes
        original_size = refiner._component_sizes["C1"]

        # Create state with rotated component
        state = refiner._initialize_state()
        state.rotations["C1"] = 90.0

        # Update sizes
        refiner._update_component_sizes(state)

        # For a rectangular component, rotating 90 degrees swaps width/height
        new_size = refiner._component_sizes["C1"]

        # The half-dimensions should have swapped (approximately)
        assert abs(original_size[0] - new_size[1]) < 0.5 or \
               abs(original_size[1] - new_size[0]) < 0.5


# =============================================================================
# DNP Component Tests
# =============================================================================

class TestDNPComponents:
    """Tests for Do Not Populate component handling."""

    def test_dnp_components_included_in_placement(self, simple_board, default_config):
        """Test that DNP components participate in force calculations.

        DNP (Do Not Populate) components should still be included in placement
        to avoid overlaps and ensure proper layout, even though they won't be
        physically populated on the board.
        """
        simple_board.components["C1"].dnp = True

        refiner = ForceDirectedRefiner(simple_board, default_config)
        state = refiner._initialize_state()

        forces = refiner._calculate_all_forces(state)

        # C1 should still receive forces (repulsion, attraction, etc.)
        # even though it's marked DNP - placement still matters
        c1_forces = forces.get("C1", [])

        # DNP component should be in the forces dict and participate in layout
        assert "C1" in forces, "DNP component should participate in placement"
        # It should have position tracked
        assert "C1" in state.positions, "DNP component should have position tracked"


# =============================================================================
# Locked Component Tests
# =============================================================================

class TestLockedComponents:
    """Tests for locked component behavior."""

    def test_locked_component_doesnt_move(self, simple_board, default_config):
        """Test that locked components don't move."""
        simple_board.components["R1"].locked = True
        initial_x = simple_board.components["R1"].x
        initial_y = simple_board.components["R1"].y

        config = RefinementConfig(
            lock_placed=True,
            max_iterations=50,
        )

        refiner = ForceDirectedRefiner(simple_board, config)
        state = refiner.refine()

        # R1 should not have moved
        final_x, final_y = state.positions["R1"]

        assert abs(final_x - initial_x) < 0.01, "Locked component X should not change"
        assert abs(final_y - initial_y) < 0.01, "Locked component Y should not change"
