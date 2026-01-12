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

    # Convergence parameters
    energy_window: int = 10  # Number of frames for rolling average
    energy_variance_threshold: float = 0.01  # Converge when variance < threshold

    # Spacing parameters
    min_clearance: float = 0.25  # mm between components
    preferred_clearance: float = 0.5  # mm preferred spacing
    max_attraction_distance: float = 50.0  # mm - cap attraction beyond this
    repulsion_cutoff: float = 50.0  # mm - no repulsion beyond this distance

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
        # Note: Connectivity matrix built but reserved for future enhancement
        # (pin-count weighted attraction). Currently attraction uses per-net model.
        # Uncomment when implementing multi-net pair attraction boost:
        # self._connectivity_matrix = self._build_connectivity_matrix()

        # Track component sizes for collision detection
        self._component_sizes = self._compute_component_sizes()

    def add_constraint(self, constraint: 'PlacementConstraint'):
        """Add a placement constraint."""
        self.constraints.append(constraint)

    def refine(self, callback: Optional[Callable[[PlacementState], None]] = None
               ) -> PlacementState:
        """
        Run force-directed refinement.

        Uses rolling average energy convergence to detect:
        - Stable convergence (energy variance below threshold)
        - Oscillation (high energy but low variance = stuck in local minimum)
        - Stall (no significant movement but hasn't converged)

        Args:
            callback: Optional function called each iteration with current state

        Returns:
            Final PlacementState with optimized positions
        """
        # Initialize state
        state = self._initialize_state()

        # Energy history for rolling average convergence detection
        energy_history: List[float] = []

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

            # Track energy history for convergence detection
            energy_history.append(state.total_energy)
            if len(energy_history) > self.config.energy_window:
                energy_history.pop(0)

            # Optional callback for visualization/logging
            if callback:
                callback(state)

            # Check convergence - requires BOTH low movement AND low energy variance
            # to prevent freezing high-energy states when damping limits movement
            converged = False
            low_movement = max_movement < self.config.min_movement

            # Check energy variance when we have enough history
            if len(energy_history) >= self.config.energy_window:
                energy_variance = self._calculate_variance(energy_history)
                avg_energy = sum(energy_history) / len(energy_history)
                num_components = len(state.positions)

                # Normalize variance by average energy to make threshold meaningful
                normalized_variance = energy_variance / (avg_energy + 1e-6)
                low_variance = normalized_variance < self.config.energy_variance_threshold

                # Scale energy threshold by component count for board-size independence
                # More components = higher baseline energy, so scale accordingly
                energy_threshold = self.config.min_movement * 10 * max(1, num_components / 10)
                low_energy = avg_energy < energy_threshold

                # Converge when movement is low AND (variance is low OR energy is low)
                # This prevents false convergence when high forces exist but movement is damped
                converged = low_movement and (low_variance or low_energy)
            else:
                # Before we have energy history, use movement + very low energy as fallback
                # Scale by component count for consistency
                num_components = len(state.positions)
                energy_threshold = self.config.min_movement * max(1, num_components / 10)
                converged = low_movement and state.total_energy < energy_threshold

            if converged:
                state.converged = True
                break

        # Apply final positions to board
        self._apply_to_board(state)

        return state

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return float('inf')
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

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

        Uses AABB overlap detection for more accurate collision detection
        with rectangular components. Also uses preferred_clearance for spacing
        and cutoff distance to avoid long-range force accumulation.

        When one component is locked, the non-locked component receives double
        the repulsion force to fully resolve the overlap.
        """
        refs = list(state.positions.keys())
        # Cutoff distance beyond which repulsion is negligible (reduces noise)
        cutoff_distance = self.config.repulsion_cutoff

        for i, ref1 in enumerate(refs):
            x1, y1 = state.positions[ref1]
            half_w1, half_h1 = self._component_sizes[ref1]
            locked1 = self._is_component_locked(ref1)

            for ref2 in refs[i+1:]:
                x2, y2 = state.positions[ref2]
                half_w2, half_h2 = self._component_sizes[ref2]
                locked2 = self._is_component_locked(ref2)

                # Skip if both locked (neither can move)
                if locked1 and locked2:
                    continue

                # Calculate center-to-center distance and direction
                dx = x1 - x2
                dy = y1 - y2
                distance = max(math.sqrt(dx*dx + dy*dy), 0.01)

                # Skip if beyond cutoff (avoids long-range force accumulation)
                if distance > cutoff_distance:
                    continue

                # AABB overlap detection - check if bounding boxes overlap
                # along each axis independently
                overlap_x = (half_w1 + half_w2 + self.config.min_clearance) - abs(dx)
                overlap_y = (half_h1 + half_h2 + self.config.min_clearance) - abs(dy)

                # Preferred separation for comfortable spacing
                preferred_sep_x = half_w1 + half_w2 + self.config.preferred_clearance
                preferred_sep_y = half_h1 + half_h2 + self.config.preferred_clearance

                if overlap_x > 0 and overlap_y > 0:
                    # Components are overlapping or too close - strong repulsion
                    # Push along the axis with smaller overlap (easier to separate)
                    if overlap_x < overlap_y:
                        force_mag = self.config.repulsion_strength * overlap_x
                        fx = force_mag * (1.0 if dx >= 0 else -1.0)
                        fy = 0.0
                    else:
                        force_mag = self.config.repulsion_strength * overlap_y
                        fx = 0.0
                        fy = force_mag * (1.0 if dy >= 0 else -1.0)

                    # Double force for non-locked component when other is locked
                    mult1 = 2.0 if locked2 else 1.0
                    mult2 = 2.0 if locked1 else 1.0
                    if not locked1:
                        forces[ref1].append(Force(fx * mult1, fy * mult1, ForceType.REPULSION,
                                                  f"repel_{ref2}"))
                    if not locked2:
                        forces[ref2].append(Force(-fx * mult2, -fy * mult2, ForceType.REPULSION,
                                                  f"repel_{ref1}"))

                elif abs(dx) < preferred_sep_x and abs(dy) < preferred_sep_y:
                    # Between min and preferred clearance - medium repulsion
                    # Use max shortfall to ensure proper spacing on the tighter axis
                    shortfall_x = preferred_sep_x - abs(dx)
                    shortfall_y = preferred_sep_y - abs(dy)
                    max_shortfall = self.config.preferred_clearance - self.config.min_clearance

                    if max_shortfall > 0:
                        # Use max to respond to worst-case axis (prevents under-repulsion)
                        ratio = max(shortfall_x, shortfall_y) / max_shortfall
                        ratio = min(ratio, 1.0)
                        force_mag = self.config.repulsion_strength * 0.3 * ratio

                        fx = force_mag * dx / distance
                        fy = force_mag * dy / distance

                        mult1 = 2.0 if locked2 else 1.0
                        mult2 = 2.0 if locked1 else 1.0
                        if not locked1:
                            forces[ref1].append(Force(fx * mult1, fy * mult1, ForceType.REPULSION,
                                                      f"space_{ref2}"))
                        if not locked2:
                            forces[ref2].append(Force(-fx * mult2, -fy * mult2, ForceType.REPULSION,
                                                      f"space_{ref1}"))
                else:
                    # Weak inverse-distance for distant components
                    force_mag = self.config.repulsion_strength * 0.1 / distance

                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    mult1 = 2.0 if locked2 else 1.0
                    mult2 = 2.0 if locked1 else 1.0
                    if not locked1:
                        forces[ref1].append(Force(fx * mult1, fy * mult1, ForceType.REPULSION,
                                                  f"space_{ref2}"))
                    if not locked2:
                        forces[ref2].append(Force(-fx * mult2, -fy * mult2, ForceType.REPULSION,
                                                  f"space_{ref1}"))

    def _add_attraction_forces(self, state: PlacementState,
                               forces: Dict[str, List[Force]]):
        """Add attraction forces for connected components.

        Uses a Hybrid Net Model:
        - For small nets (<=3 pins): pairwise attraction between all components
        - For large nets (>3 pins): Star Model - attract all pins to virtual centroid

        The star model reduces O(N²) to O(N) for high-degree nets (GND, VCC)
        and scales attraction by 1/k (where k is pin count) to prevent collapse.
        """
        for net_name, net in self.board.nets.items():
            if len(net.connections) < 2:
                continue

            # Get component references for this net (filter to valid positions)
            refs = [ref for ref in net.get_component_refs() if ref in state.positions]
            num_refs = len(refs)

            if num_refs < 2:
                continue

            # Base strength, boosted for power/ground nets
            base_strength = self.config.attraction_strength
            if net.is_power or net.is_ground:
                base_strength *= 2.0

            # Cap effective distance to prevent unbounded attraction on long nets
            max_dist = self.config.max_attraction_distance

            if num_refs <= 3:
                # Small nets: use pairwise attraction (original behavior)
                for i, ref1 in enumerate(refs):
                    x1, y1 = state.positions[ref1]

                    for ref2 in refs[i+1:]:
                        x2, y2 = state.positions[ref2]

                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)

                        if distance < 0.1:
                            continue

                        # Cap effective distance to limit attraction force
                        effective_dist = min(distance, max_dist)
                        force_mag = base_strength * effective_dist

                        fx = force_mag * dx / distance
                        fy = force_mag * dy / distance

                        forces[ref1].append(Force(fx, fy, ForceType.ATTRACTION,
                                                  f"net_{net_name}"))
                        forces[ref2].append(Force(-fx, -fy, ForceType.ATTRACTION,
                                                  f"net_{net_name}"))
            else:
                # Large nets: use Star Model with virtual centroid
                # Calculate centroid of all connected components
                centroid_x = sum(state.positions[ref][0] for ref in refs) / num_refs
                centroid_y = sum(state.positions[ref][1] for ref in refs) / num_refs

                # Scale strength by 1/k to prevent collapse on large nets
                scaled_strength = base_strength / num_refs

                for ref in refs:
                    x, y = state.positions[ref]

                    dx = centroid_x - x
                    dy = centroid_y - y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance < 0.1:
                        continue

                    # Cap effective distance to limit attraction force
                    effective_dist = min(distance, max_dist)
                    force_mag = scaled_strength * effective_dist

                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    forces[ref].append(Force(fx, fy, ForceType.ATTRACTION,
                                             f"net_{net_name}_star"))

    def _add_boundary_forces(self, state: PlacementState,
                             forces: Dict[str, List[Force]]):
        """Add forces to keep components within board boundaries.

        Uses AABB (Axis-Aligned Bounding Box) instead of diagonal radius
        for more accurate boundary detection with rectangular components.
        This allows long thin components to get closer to edges without
        false boundary violations.

        Forces are accumulated (+=) rather than overwritten to properly
        handle corner violations where both X and Y boundaries are exceeded.
        """
        outline = self.board.outline

        for ref, (x, y) in state.positions.items():
            # Get component dimensions (accounting for rotation)
            half_w, half_h = self._component_sizes[ref]
            clearance = self.config.min_clearance

            fx, fy = 0.0, 0.0

            # Left edge - check component's left boundary
            left_bound = x - half_w - clearance
            if left_bound < outline.origin_x:
                fx += self.config.boundary_strength * (outline.origin_x - left_bound)

            # Right edge - check component's right boundary
            right_bound = x + half_w + clearance
            if right_bound > outline.origin_x + outline.width:
                fx += self.config.boundary_strength * (
                    outline.origin_x + outline.width - right_bound)

            # Top edge - check component's top boundary
            top_bound = y - half_h - clearance
            if top_bound < outline.origin_y:
                fy += self.config.boundary_strength * (outline.origin_y - top_bound)

            # Bottom edge - check component's bottom boundary
            bottom_bound = y + half_h + clearance
            if bottom_bound > outline.origin_y + outline.height:
                fy += self.config.boundary_strength * (
                    outline.origin_y + outline.height - bottom_bound)

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

            # Compute mass based on component area (larger = heavier = moves less)
            # Normalized so a 1mm x 1mm component has mass ~1
            half_w, half_h = self._component_sizes[ref]
            area = 4 * half_w * half_h  # Full component area
            mass = max(1.0, area)  # Minimum mass of 1 to avoid division issues

            # Update velocity with damping and mass scaling (F = ma -> a = F/m)
            vx, vy = state.velocities[ref]
            vx = (vx + (total_fx / mass) * self.config.time_step) * self.config.damping
            vy = (vy + (total_fy / mass) * self.config.time_step) * self.config.damping

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
        """Apply rotation constraints to component rotations.

        Respects lock_placed config: locked components don't rotate.
        Updates component sizes after rotation changes to keep AABB accurate.
        """
        rotations_changed = False

        for constraint in self.constraints:
            # Check for FixedConstraint with rotation (from constraints.py)
            if hasattr(constraint, 'get_target_rotation'):
                target_rot = constraint.get_target_rotation()
                if target_rot is not None:
                    ref = getattr(constraint, 'component_ref', None)
                    if ref and ref in state.rotations:
                        # Skip locked components
                        if self._is_component_locked(ref):
                            continue

                        # Gradually rotate toward target
                        current = state.rotations[ref]
                        diff = target_rot - current

                        # Handle wraparound
                        if diff > 180:
                            diff -= 360
                        elif diff < -180:
                            diff += 360

                        # Apply rotation with damping
                        new_rot = (current + diff * 0.3) % 360
                        if abs(new_rot - current) > 0.01:
                            state.rotations[ref] = new_rot
                            rotations_changed = True

            # Also handle rotation field directly for force_directed.py constraints
            elif hasattr(constraint, 'rotation') and constraint.rotation is not None:
                ref = getattr(constraint, 'component_ref', None)
                if ref and ref in state.rotations:
                    # Skip locked components
                    if self._is_component_locked(ref):
                        continue

                    target_rot = constraint.rotation
                    current = state.rotations[ref]
                    diff = target_rot - current

                    if diff > 180:
                        diff -= 360
                    elif diff < -180:
                        diff += 360

                    new_rot = (current + diff * 0.3) % 360
                    if abs(new_rot - current) > 0.01:
                        state.rotations[ref] = new_rot
                        rotations_changed = True

        # Update component sizes if any rotations changed (keeps AABB accurate)
        if rotations_changed:
            self._update_component_sizes(state)

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

    def _compute_component_sizes(self) -> Dict[str, Tuple[float, float]]:
        """Compute effective AABB half-dimensions for each component.

        Returns (half_width, half_height) tuples accounting for rotation.
        For rotated components, we compute the axis-aligned bounding box
        of the rotated rectangle.
        """
        sizes = {}
        for ref, comp in self.board.components.items():
            w, h = comp.width, comp.height
            rot_rad = math.radians(comp.rotation)

            # For axis-aligned bounding box of rotated rectangle:
            # half_w = |w/2 * cos(θ)| + |h/2 * sin(θ)|
            # half_h = |w/2 * sin(θ)| + |h/2 * cos(θ)|
            cos_r = abs(math.cos(rot_rad))
            sin_r = abs(math.sin(rot_rad))

            half_w = (w / 2) * cos_r + (h / 2) * sin_r
            half_h = (w / 2) * sin_r + (h / 2) * cos_r

            sizes[ref] = (half_w, half_h)
        return sizes

    def _update_component_sizes(self, state: PlacementState):
        """Update component sizes based on current rotation state.

        Called after rotation constraints are applied to keep AABB
        dimensions accurate for boundary and collision detection.
        """
        for ref, rotation in state.rotations.items():
            comp = self.board.components.get(ref)
            if not comp:
                continue

            w, h = comp.width, comp.height
            rot_rad = math.radians(rotation)

            # Compute axis-aligned bounding box of rotated rectangle
            cos_r = abs(math.cos(rot_rad))
            sin_r = abs(math.sin(rot_rad))

            half_w = (w / 2) * cos_r + (h / 2) * sin_r
            half_h = (w / 2) * sin_r + (h / 2) * cos_r

            self._component_sizes[ref] = (half_w, half_h)


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
