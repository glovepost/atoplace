"""
Force-Directed Placement Refinement

Uses a physics-based simulation to optimize component placement by
balancing repulsion (prevent overlap) with attraction (minimize wire length)
and constraint forces (user requirements).

Based on research in layout_rules_research.md:
- Minimize loop areas for power and signal paths
- Keep decoupling capacitors close to IC power pins
- Maintain modularity (analog/digital/RF separation)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable
from enum import Enum

from ..board.abstraction import Board, Component, Net


class ForceType(Enum):
    """Types of forces in the simulation."""
    REPULSION = "repulsion"       # Prevents overlap
    ATTRACTION = "attraction"     # Pulls connected components together
    BOUNDARY = "boundary"         # Keeps components on board
    CONSTRAINT = "constraint"     # User-specified constraints
    ALIGNMENT = "alignment"       # Grid/alignment snapping


@dataclass
class Force:
    """A force vector with metadata."""
    fx: float
    fy: float
    force_type: ForceType
    source: str = ""  # Description of what generated this force
    magnitude: float = field(init=False)

    def __post_init__(self):
        self.magnitude = math.sqrt(self.fx * self.fx + self.fy * self.fy)


@dataclass
class PlacementState:
    """Current state of component placement."""
    positions: Dict[str, Tuple[float, float]]  # ref -> (x, y)
    rotations: Dict[str, float]  # ref -> rotation
    velocities: Dict[str, Tuple[float, float]]  # ref -> (vx, vy)
    iteration: int = 0
    total_energy: float = 0.0
    converged: bool = False


@dataclass
class RefinementConfig:
    """Configuration for force-directed refinement."""
    # Force strengths
    repulsion_strength: float = 100.0
    attraction_strength: float = 0.5
    boundary_strength: float = 200.0
    constraint_strength: float = 50.0
    alignment_strength: float = 10.0

    # Physics parameters
    damping: float = 0.85
    time_step: float = 0.1
    min_movement: float = 0.01  # mm - convergence threshold
    max_iterations: int = 500
    max_velocity: float = 5.0  # mm per iteration

    # Spacing parameters
    min_clearance: float = 0.25  # mm between components
    preferred_clearance: float = 0.5  # mm preferred spacing

    # Grid alignment
    grid_size: float = 0.0  # 0 = no grid snapping
    snap_to_grid: bool = False

    # Component-specific
    lock_placed: bool = False  # Don't move already-placed components


class ForceDirectedRefiner:
    """
    Refine component placement using force-directed algorithm.

    Forces applied:
    1. Repulsion - prevents overlapping/close components
    2. Attraction - pulls connected components together (minimize wire length)
    3. Boundary - keeps components within board outline
    4. Constraint - enforces user-specified constraints
    5. Alignment - optional grid snapping
    """

    def __init__(self, board: Board, config: Optional[RefinementConfig] = None):
        self.board = board
        self.config = config or RefinementConfig()
        self.constraints: List['PlacementConstraint'] = []

        # Precompute connectivity for attraction forces
        self._connectivity_matrix = self._build_connectivity_matrix()

        # Track component sizes for collision detection
        self._component_sizes = self._compute_component_sizes()

    def add_constraint(self, constraint: 'PlacementConstraint'):
        """Add a placement constraint."""
        self.constraints.append(constraint)

    def refine(self, callback: Optional[Callable[[PlacementState], None]] = None
               ) -> PlacementState:
        """
        Run force-directed refinement.

        Args:
            callback: Optional function called each iteration with current state

        Returns:
            Final PlacementState with optimized positions
        """
        # Initialize state
        state = self._initialize_state()

        for iteration in range(self.config.max_iterations):
            state.iteration = iteration

            # Calculate forces on each component
            forces = self._calculate_all_forces(state)

            # Update velocities and positions
            max_movement = self._apply_forces(state, forces)

            # Apply rotation constraints
            self._apply_rotation_constraints(state)

            # Calculate total system energy
            state.total_energy = self._calculate_energy(state, forces)

            # Optional callback for visualization/logging
            if callback:
                callback(state)

            # Check convergence
            if max_movement < self.config.min_movement:
                state.converged = True
                break

        # Apply final positions to board
        self._apply_to_board(state)

        return state

    def _initialize_state(self) -> PlacementState:
        """Initialize placement state from current board."""
        positions = {}
        rotations = {}
        velocities = {}

        for ref, comp in self.board.components.items():
            positions[ref] = (comp.x, comp.y)
            rotations[ref] = comp.rotation
            velocities[ref] = (0.0, 0.0)

        return PlacementState(
            positions=positions,
            rotations=rotations,
            velocities=velocities,
        )

    def _is_component_locked(self, ref: str) -> bool:
        """Check if a component should be treated as locked."""
        if not self.config.lock_placed:
            return False
        comp = self.board.components.get(ref)
        return comp.locked if comp else False

    def _calculate_all_forces(self, state: PlacementState
                              ) -> Dict[str, List[Force]]:
        """Calculate all forces on each component."""
        forces: Dict[str, List[Force]] = {ref: [] for ref in state.positions}

        # 1. Repulsion forces
        self._add_repulsion_forces(state, forces)

        # 2. Attraction forces (net connectivity)
        self._add_attraction_forces(state, forces)

        # 3. Boundary forces
        self._add_boundary_forces(state, forces)

        # 4. Constraint forces
        self._add_constraint_forces(state, forces)

        # 5. Alignment forces (optional)
        if self.config.snap_to_grid and self.config.grid_size > 0:
            self._add_alignment_forces(state, forces)

        return forces

    def _add_repulsion_forces(self, state: PlacementState,
                              forces: Dict[str, List[Force]]):
        """Add repulsion forces between components.

        Uses preferred_clearance to add spacing force between components
        that are between min_clearance and preferred_clearance distance.
        Also adds cutoff distance to avoid long-range forces dominating.
        """
        refs = list(state.positions.keys())
        # Cutoff distance beyond which repulsion is negligible (reduces noise)
        cutoff_distance = 50.0  # mm

        for i, ref1 in enumerate(refs):
            x1, y1 = state.positions[ref1]
            size1 = self._component_sizes[ref1]

            for ref2 in refs[i+1:]:
                x2, y2 = state.positions[ref2]
                size2 = self._component_sizes[ref2]

                # Calculate distance and direction
                dx = x1 - x2
                dy = y1 - y2
                distance = max(math.sqrt(dx*dx + dy*dy), 0.01)

                # Skip if beyond cutoff (avoids long-range force accumulation)
                if distance > cutoff_distance:
                    continue

                # Minimum distance to avoid overlap
                min_dist = (size1 + size2) / 2 + self.config.min_clearance
                # Preferred distance for comfortable spacing
                preferred_dist = (size1 + size2) / 2 + self.config.preferred_clearance

                if distance < min_dist:
                    # Strong repulsion when overlapping or below min clearance
                    overlap = min_dist - distance
                    force_mag = self.config.repulsion_strength * overlap

                    # Direction from ref2 to ref1
                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    forces[ref1].append(Force(fx, fy, ForceType.REPULSION,
                                              f"repel_{ref2}"))
                    forces[ref2].append(Force(-fx, -fy, ForceType.REPULSION,
                                              f"repel_{ref1}"))
                elif distance < preferred_dist:
                    # Medium repulsion when between min and preferred clearance
                    # Linear falloff as distance approaches preferred
                    shortfall = preferred_dist - distance
                    max_shortfall = preferred_dist - min_dist
                    ratio = shortfall / max_shortfall if max_shortfall > 0 else 0
                    force_mag = self.config.repulsion_strength * 0.3 * ratio

                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    forces[ref1].append(Force(fx, fy, ForceType.REPULSION,
                                              f"space_{ref2}"))
                    forces[ref2].append(Force(-fx, -fy, ForceType.REPULSION,
                                              f"space_{ref1}"))
                else:
                    # Weak inverse-distance (not inverse-square) for distant components
                    # Using inverse distance instead of inverse-square reduces
                    # oscillation in dense boards
                    force_mag = self.config.repulsion_strength * 0.1 / distance

                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    forces[ref1].append(Force(fx, fy, ForceType.REPULSION,
                                              f"space_{ref2}"))
                    forces[ref2].append(Force(-fx, -fy, ForceType.REPULSION,
                                              f"space_{ref1}"))

    def _add_attraction_forces(self, state: PlacementState,
                               forces: Dict[str, List[Force]]):
        """Add attraction forces for connected components."""
        for net_name, net in self.board.nets.items():
            if len(net.connections) < 2:
                continue

            # Get component references for this net
            refs = list(net.get_component_refs())

            # Attraction between all pairs (could optimize with MST)
            for i, ref1 in enumerate(refs):
                if ref1 not in state.positions:
                    continue
                x1, y1 = state.positions[ref1]

                for ref2 in refs[i+1:]:
                    if ref2 not in state.positions:
                        continue
                    x2, y2 = state.positions[ref2]

                    # Calculate distance
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance < 0.1:
                        continue

                    # Linear attraction (Hooke's law)
                    # Stronger for power/ground nets
                    strength = self.config.attraction_strength
                    if net.is_power or net.is_ground:
                        strength *= 2.0

                    force_mag = strength * distance

                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    forces[ref1].append(Force(fx, fy, ForceType.ATTRACTION,
                                              f"net_{net_name}"))
                    forces[ref2].append(Force(-fx, -fy, ForceType.ATTRACTION,
                                              f"net_{net_name}"))

    def _add_boundary_forces(self, state: PlacementState,
                             forces: Dict[str, List[Force]]):
        """Add forces to keep components within board boundaries."""
        outline = self.board.outline

        for ref, (x, y) in state.positions.items():
            size = self._component_sizes[ref]
            margin = size / 2 + self.config.min_clearance

            fx, fy = 0.0, 0.0

            # Left edge
            if x - margin < outline.origin_x:
                fx = self.config.boundary_strength * (outline.origin_x + margin - x)

            # Right edge
            if x + margin > outline.origin_x + outline.width:
                fx = self.config.boundary_strength * (
                    outline.origin_x + outline.width - margin - x)

            # Top edge
            if y - margin < outline.origin_y:
                fy = self.config.boundary_strength * (outline.origin_y + margin - y)

            # Bottom edge
            if y + margin > outline.origin_y + outline.height:
                fy = self.config.boundary_strength * (
                    outline.origin_y + outline.height - margin - y)

            if fx != 0 or fy != 0:
                forces[ref].append(Force(fx, fy, ForceType.BOUNDARY, "boundary"))

    def _add_constraint_forces(self, state: PlacementState,
                               forces: Dict[str, List[Force]]):
        """Add forces for user-defined constraints."""
        for constraint in self.constraints:
            # Support both interfaces:
            # - calculate_forces(state, board, strength) - force_directed.py constraints
            # - calculate_force(board, ref, strength) - constraints.py constraints
            if hasattr(constraint, 'calculate_forces'):
                constraint_forces = constraint.calculate_forces(
                    state, self.board, self.config.constraint_strength)
                for ref, (fx, fy) in constraint_forces.items():
                    if ref in forces:
                        forces[ref].append(Force(fx, fy, ForceType.CONSTRAINT,
                                                 getattr(constraint, 'description', '')))
            elif hasattr(constraint, 'calculate_force'):
                # Iterate over all components for constraints.py style
                for ref in state.positions:
                    fx, fy = constraint.calculate_force(
                        self.board, ref, self.config.constraint_strength)
                    if fx != 0 or fy != 0:
                        forces[ref].append(Force(fx, fy, ForceType.CONSTRAINT,
                                                 getattr(constraint, 'description', '')))

    def _add_alignment_forces(self, state: PlacementState,
                              forces: Dict[str, List[Force]]):
        """Add forces for grid alignment."""
        grid = self.config.grid_size

        for ref, (x, y) in state.positions.items():
            # Find nearest grid points
            nearest_x = round(x / grid) * grid
            nearest_y = round(y / grid) * grid

            dx = nearest_x - x
            dy = nearest_y - y

            if abs(dx) > 0.01 or abs(dy) > 0.01:
                fx = self.config.alignment_strength * dx
                fy = self.config.alignment_strength * dy
                forces[ref].append(Force(fx, fy, ForceType.ALIGNMENT, "grid"))

    def _apply_forces(self, state: PlacementState,
                      forces: Dict[str, List[Force]]) -> float:
        """Apply forces to update velocities and positions.

        Respects lock_placed config: locked components don't move.
        """
        max_movement = 0.0

        for ref in state.positions:
            # Skip locked components
            if self._is_component_locked(ref):
                state.velocities[ref] = (0.0, 0.0)
                continue

            # Sum all forces
            total_fx = sum(f.fx for f in forces[ref])
            total_fy = sum(f.fy for f in forces[ref])

            # Update velocity with damping
            vx, vy = state.velocities[ref]
            vx = (vx + total_fx * self.config.time_step) * self.config.damping
            vy = (vy + total_fy * self.config.time_step) * self.config.damping

            # Clamp velocity
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > self.config.max_velocity:
                scale = self.config.max_velocity / speed
                vx *= scale
                vy *= scale

            state.velocities[ref] = (vx, vy)

            # Update position
            x, y = state.positions[ref]
            new_x = x + vx
            new_y = y + vy

            movement = math.sqrt((new_x - x)**2 + (new_y - y)**2)
            max_movement = max(max_movement, movement)

            state.positions[ref] = (new_x, new_y)

        return max_movement

    def _calculate_energy(self, state: PlacementState,
                          forces: Dict[str, List[Force]]) -> float:
        """Calculate total system energy for convergence tracking."""
        total_energy = 0.0

        # Kinetic energy from velocities
        for vx, vy in state.velocities.values():
            total_energy += 0.5 * (vx*vx + vy*vy)

        # Potential energy from forces (simplified)
        for ref_forces in forces.values():
            for f in ref_forces:
                total_energy += f.magnitude

        return total_energy

    def _apply_rotation_constraints(self, state: PlacementState):
        """Apply rotation constraints to component rotations."""
        for constraint in self.constraints:
            # Check for FixedConstraint with rotation (from constraints.py)
            if hasattr(constraint, 'get_target_rotation'):
                target_rot = constraint.get_target_rotation()
                if target_rot is not None:
                    ref = getattr(constraint, 'component_ref', None)
                    if ref and ref in state.rotations:
                        # Gradually rotate toward target
                        current = state.rotations[ref]
                        diff = target_rot - current

                        # Handle wraparound
                        if diff > 180:
                            diff -= 360
                        elif diff < -180:
                            diff += 360

                        # Apply rotation with damping
                        state.rotations[ref] = (current + diff * 0.3) % 360

            # Also handle rotation field directly for force_directed.py constraints
            elif hasattr(constraint, 'rotation') and constraint.rotation is not None:
                ref = getattr(constraint, 'component_ref', None)
                if ref and ref in state.rotations:
                    target_rot = constraint.rotation
                    current = state.rotations[ref]
                    diff = target_rot - current

                    if diff > 180:
                        diff -= 360
                    elif diff < -180:
                        diff += 360

                    state.rotations[ref] = (current + diff * 0.3) % 360

    def _apply_to_board(self, state: PlacementState):
        """Apply final positions to the board."""
        for ref, (x, y) in state.positions.items():
            if ref in self.board.components:
                self.board.components[ref].x = x
                self.board.components[ref].y = y
                self.board.components[ref].rotation = state.rotations[ref]

    def _build_connectivity_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build connectivity weights between components."""
        connectivity = {}

        for net in self.board.nets.values():
            refs = list(net.get_component_refs())
            weight = 1.0
            if net.is_power or net.is_ground:
                weight = 2.0

            for i, ref1 in enumerate(refs):
                for ref2 in refs[i+1:]:
                    key = tuple(sorted([ref1, ref2]))
                    connectivity[key] = connectivity.get(key, 0) + weight

        return connectivity

    def _compute_component_sizes(self) -> Dict[str, float]:
        """Compute effective size (diagonal) for each component."""
        sizes = {}
        for ref, comp in self.board.components.items():
            # Use diagonal as effective size for collision detection
            sizes[ref] = math.sqrt(comp.width**2 + comp.height**2)
        return sizes


@dataclass
class PlacementConstraint:
    """Base class for placement constraints."""
    description: str = ""
    priority: str = "preferred"  # "required", "preferred", "optional"
    affected_components: List[str] = field(default_factory=list)

    def calculate_forces(self, state: PlacementState, board: Board,
                         strength: float) -> Dict[str, Tuple[float, float]]:
        """Calculate forces to satisfy this constraint."""
        raise NotImplementedError


@dataclass
class ProximityConstraint(PlacementConstraint):
    """Keep components close together."""
    target_ref: str = ""
    anchor_ref: str = ""
    max_distance: float = 5.0  # mm

    def calculate_forces(self, state: PlacementState, board: Board,
                         strength: float) -> Dict[str, Tuple[float, float]]:
        forces = {}

        if self.target_ref not in state.positions or self.anchor_ref not in state.positions:
            return forces

        x1, y1 = state.positions[self.target_ref]
        x2, y2 = state.positions[self.anchor_ref]

        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > self.max_distance:
            force_mag = strength * (distance - self.max_distance) / distance
            forces[self.target_ref] = (force_mag * dx, force_mag * dy)

        return forces


@dataclass
class EdgeConstraint(PlacementConstraint):
    """Place component on board edge."""
    component_ref: str = ""
    edge: str = "left"  # "left", "right", "top", "bottom"
    offset: float = 2.0  # mm from edge

    def calculate_forces(self, state: PlacementState, board: Board,
                         strength: float) -> Dict[str, Tuple[float, float]]:
        forces = {}

        if self.component_ref not in state.positions:
            return forces

        x, y = state.positions[self.component_ref]
        outline = board.outline

        fx, fy = 0.0, 0.0

        if self.edge == "left":
            target_x = outline.origin_x + self.offset
            fx = strength * (target_x - x)
        elif self.edge == "right":
            target_x = outline.origin_x + outline.width - self.offset
            fx = strength * (target_x - x)
        elif self.edge == "top":
            target_y = outline.origin_y + self.offset
            fy = strength * (target_y - y)
        elif self.edge == "bottom":
            target_y = outline.origin_y + outline.height - self.offset
            fy = strength * (target_y - y)

        forces[self.component_ref] = (fx, fy)
        return forces


@dataclass
class ZoneConstraint(PlacementConstraint):
    """Keep components within a specific zone."""
    zone_x: float = 0.0
    zone_y: float = 0.0
    zone_width: float = 50.0
    zone_height: float = 50.0

    def calculate_forces(self, state: PlacementState, board: Board,
                         strength: float) -> Dict[str, Tuple[float, float]]:
        forces = {}

        for ref in self.affected_components:
            if ref not in state.positions:
                continue

            x, y = state.positions[ref]
            fx, fy = 0.0, 0.0

            # Push toward zone if outside
            if x < self.zone_x:
                fx = strength * (self.zone_x - x)
            elif x > self.zone_x + self.zone_width:
                fx = strength * (self.zone_x + self.zone_width - x)

            if y < self.zone_y:
                fy = strength * (self.zone_y - y)
            elif y > self.zone_y + self.zone_height:
                fy = strength * (self.zone_y + self.zone_height - y)

            if fx != 0 or fy != 0:
                forces[ref] = (fx, fy)

        return forces
