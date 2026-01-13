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

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable
from enum import Enum

from ..board.abstraction import Board, Component, Net

logger = logging.getLogger(__name__)


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
    center_strength: float = 5.0  # Attraction toward board center/anchor

    # Physics parameters
    damping: float = 0.85
    time_step: float = 0.1
    min_movement: float = 0.01  # mm - convergence threshold
    max_iterations: int = 500
    max_velocity: float = 5.0  # mm per iteration

    # Convergence parameters
    energy_window: int = 10  # Number of frames for rolling average
    energy_variance_threshold: float = 0.01  # Converge when variance < threshold

    # Adaptive damping for oscillation control
    adaptive_damping: bool = True  # Enable adaptive damping
    damping_increase_rate: float = 0.02  # Rate to increase damping on oscillation
    max_damping: float = 0.98  # Maximum damping value
    velocity_decay_rate: float = 0.95  # Decay velocity clamp on oscillation

    # Spacing parameters
    min_clearance: float = 0.25  # mm between components
    edge_clearance: float = 0.3  # mm from board edge (should match DFM min_trace_to_edge)
    preferred_clearance: float = 0.5  # mm preferred spacing
    max_attraction_distance: float = 50.0  # mm - cap attraction beyond this
    repulsion_cutoff: float = 20.0  # mm - no repulsion beyond this distance (reduced from 50)

    # Grid alignment
    grid_size: float = 0.0  # 0 = no grid snapping
    snap_to_grid: bool = False

    # Component-specific
    lock_placed: bool = False  # Don't move already-placed components

    # Anchor/center mode
    auto_anchor_largest_ic: bool = True  # Auto-lock largest IC at center as anchor
    initial_radius: float = 30.0  # mm - move distant components within this radius at start


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

    def __init__(
        self,
        board: Board,
        config: Optional[RefinementConfig] = None,
        visualizer: Optional['PlacementVisualizer'] = None,
        modules: Optional[Dict[str, str]] = None,
    ):
        self.board = board
        self.config = config or RefinementConfig()
        self.constraints: List['PlacementConstraint'] = []
        self.visualizer = visualizer
        self.modules = modules or {}  # ref -> module_type mapping
        self._viz_interval = 2  # Capture frame every N iterations (lower = smoother playback)

        # Anchor components per layer (largest IC/component on each layer, locked at center)
        self._anchor_ref: Optional[str] = None  # Top layer anchor (legacy compat)
        self._anchor_top: Optional[str] = None  # Top layer anchor
        self._anchor_bottom: Optional[str] = None  # Bottom layer anchor
        self._center_x: float = 0.0
        self._center_y: float = 0.0

        # Precompute connectivity for attraction forces
        # Note: Connectivity matrix built but reserved for future enhancement
        # (pin-count weighted attraction). Currently attraction uses per-net model.
        # Uncomment when implementing multi-net pair attraction boost:
        # self._connectivity_matrix = self._build_connectivity_matrix()

        # Track component sizes for collision detection
        self._component_sizes = self._compute_component_sizes()

        # Find and setup anchor if enabled
        if self.config.auto_anchor_largest_ic:
            self._setup_anchor()

    def add_constraint(self, constraint: 'PlacementConstraint'):
        """Add a placement constraint."""
        self.constraints.append(constraint)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Added constraint: %s", getattr(constraint, "description", constraint))

    def _capture_viz_frame(
        self,
        state: PlacementState,
        forces: Dict[str, List[Force]],
        label: str,
        phase: str = "refinement",
    ):
        """Capture a visualization frame if visualizer is enabled."""
        if not self.visualizer:
            return

        # Build component data
        components = {}
        pads = {}

        for ref, (x, y) in state.positions.items():
            comp = self.board.components.get(ref)
            if not comp or comp.dnp:
                continue

            rot = state.rotations.get(ref, comp.rotation)
            # Component x,y is now the centroid (as of the origin offset fix)
            components[ref] = (x, y, rot, comp.width, comp.height)

            # Extract pads - need to compute pad positions relative to centroid
            # Pad.x,y is relative to KiCad origin, so we need to offset by -origin_offset
            pad_list = []
            for pad in comp.pads:
                # Transform pad from KiCad-origin-relative to centroid-relative
                # then apply rotation and translate to board coordinates
                px = pad.x - comp.origin_offset_x
                py = pad.y - comp.origin_offset_y
                if rot != 0:
                    rad = math.radians(rot)
                    cos_r = math.cos(rad)
                    sin_r = math.sin(rad)
                    px, py = px * cos_r - py * sin_r, px * sin_r + py * cos_r
                pad_list.append((
                    x + px,
                    y + py,
                    pad.width,
                    pad.height,
                    pad.net or "",
                ))
            pads[ref] = pad_list

        # Build force data
        force_data = {}
        for ref, force_list in forces.items():
            force_tuples = []
            for f in force_list:
                force_tuples.append((f.fx, f.fy, f.force_type.value))
            force_data[ref] = force_tuples

        # Find overlaps
        overlap_pairs = self.board.find_overlaps(clearance=0.1)

        self.visualizer.capture_frame(
            label=label,
            iteration=state.iteration,
            phase=phase,
            components=components,
            pads=pads,
            modules=self.modules,
            forces=force_data,
            overlaps=list(overlap_pairs),
            energy=state.total_energy,
            max_move=max(
                (abs(vx) + abs(vy) for vx, vy in state.velocities.values()),
                default=0.0
            ),
            overlap_count=len(overlap_pairs),
        )

    def refine(self, callback: Optional[Callable[[PlacementState], None]] = None
               ) -> PlacementState:
        """
        Run force-directed refinement.

        Uses rolling average energy convergence to detect:
        - Stable convergence (energy variance below threshold)
        - Oscillation (high energy but low variance = stuck in local minimum)
        - Stall (no significant movement but hasn't converged)

        Implements adaptive damping to handle oscillation:
        - Detects oscillation via energy pattern analysis
        - Increases damping when oscillation detected
        - Decays velocity clamp to encourage settling

        Args:
            callback: Optional function called each iteration with current state

        Returns:
            Final PlacementState with optimized positions
        """
        # Initialize state
        state = self._initialize_state()
        if logger.isEnabledFor(logging.DEBUG):
            locked = sum(1 for ref in state.positions if self._is_component_locked(ref))
            logger.debug(
                "Force-directed refinement start: components=%d locked=%d grid=%s snap=%s",
                len(state.positions),
                locked,
                f"{self.config.grid_size:.3f}mm" if self.config.grid_size else "none",
                self.config.snap_to_grid,
            )
            logger.debug(
                "Refinement config: repulsion=%.2f attraction=%.2f boundary=%.2f constraint=%.2f alignment=%.2f",
                self.config.repulsion_strength,
                self.config.attraction_strength,
                self.config.boundary_strength,
                self.config.constraint_strength,
                self.config.alignment_strength,
            )

            # Log active force types for clarity
            active_forces = ["repulsion", "attraction"]
            if self.board.outline.has_outline:
                active_forces.append("boundary")
            else:
                logger.debug("  Note: No board outline - boundary forces disabled")
            if self.constraints:
                active_forces.append(f"constraint ({len(self.constraints)} constraints)")
            else:
                logger.debug("  Note: No constraints specified - constraint forces disabled (this is normal)")
            if self.config.snap_to_grid and self.config.grid_size > 0:
                active_forces.append(f"alignment (grid={self.config.grid_size}mm)")
            else:
                logger.debug("  Note: No grid snapping - alignment forces disabled (use --grid to enable)")
            logger.debug("  Active force types: %s", ", ".join(active_forces))

        # Energy history for rolling average convergence detection
        energy_history: List[float] = []
        movement_history: List[float] = []

        # Adaptive damping state
        current_damping = self.config.damping
        current_max_velocity = self.config.max_velocity
        oscillation_count = 0

        # Stability tracking for early convergence
        stable_iterations = 0
        stable_threshold = 30  # Exit after N consecutive stable iterations
        movement_stable_threshold = 0.1  # mm - consider stable if max_movement below this

        log_every = 10

        # Capture initial state
        if self.visualizer:
            initial_forces = self._calculate_all_forces(state)
            self._capture_viz_frame(state, initial_forces, "Initial", phase="initial")

        for iteration in range(self.config.max_iterations):
            state.iteration = iteration

            # Calculate forces on each component
            forces = self._calculate_all_forces(state)

            # Update velocities and positions with current damping
            max_movement = self._apply_forces(
                state, forces,
                damping_override=current_damping,
                max_velocity_override=current_max_velocity
            )

            # Apply hard boundary clamping to prevent components going off-board
            # This is a safety net - boundary forces should normally keep components in
            self._clamp_to_boundary(state)

            # Apply rotation constraints
            self._apply_rotation_constraints(state)

            # Calculate total system energy
            state.total_energy = self._calculate_energy(state, forces)

            # Track energy and movement history for convergence/oscillation detection
            energy_history.append(state.total_energy)
            movement_history.append(max_movement)
            if len(energy_history) > self.config.energy_window:
                energy_history.pop(0)
            if len(movement_history) > self.config.energy_window:
                movement_history.pop(0)

            # Adaptive damping: detect and respond to oscillation
            if self.config.adaptive_damping and len(energy_history) >= 4:
                is_oscillating = self._detect_oscillation(energy_history, movement_history)
                if is_oscillating:
                    oscillation_count += 1
                    # Increase damping to slow things down
                    current_damping = min(
                        current_damping + self.config.damping_increase_rate,
                        self.config.max_damping
                    )
                    # Decay velocity clamp to encourage settling
                    current_max_velocity *= self.config.velocity_decay_rate

                    if logger.isEnabledFor(logging.DEBUG) and oscillation_count % 5 == 1:
                        logger.debug(
                            "Oscillation detected at iteration %d: damping=%.3f max_vel=%.3f",
                            iteration, current_damping, current_max_velocity
                        )

            # Optional callback for visualization/logging
            if callback:
                callback(state)

            # Capture visualization frame periodically
            if self.visualizer and iteration % self._viz_interval == 0:
                self._capture_viz_frame(
                    state, forces, f"Iteration {iteration}", phase="refinement"
                )

            if logger.isEnabledFor(logging.DEBUG) and iteration % log_every == 0:
                summary = self._summarize_forces(forces)
                logger.debug(
                    "Iteration %d: energy=%.3f max_move=%.4f total_forces=%d avg_force=%.3f max_force=%.3f max_force_ref=%s",
                    iteration,
                    state.total_energy,
                    max_movement,
                    summary["total_count"],
                    summary["avg_magnitude"],
                    summary["max_magnitude"],
                    summary["max_ref"],
                )
                logger.debug(
                    "  Force counts: repulsion=%d attraction=%d boundary=%d constraint=%d alignment=%d",
                    summary["counts"].get(ForceType.REPULSION, 0),
                    summary["counts"].get(ForceType.ATTRACTION, 0),
                    summary["counts"].get(ForceType.BOUNDARY, 0),
                    summary["counts"].get(ForceType.CONSTRAINT, 0),
                    summary["counts"].get(ForceType.ALIGNMENT, 0),
                )

            # Check convergence - requires BOTH low movement AND low energy variance
            # to prevent freezing high-energy states when damping limits movement
            converged = False
            low_movement = max_movement < self.config.min_movement
            low_variance = False

            # Check energy variance when we have enough history
            if len(energy_history) >= self.config.energy_window:
                energy_variance = self._calculate_variance(energy_history)
                avg_energy = sum(energy_history) / len(energy_history)
                num_components = len(state.positions)

                # Normalize variance by average energy to make threshold meaningful
                normalized_variance = energy_variance / (avg_energy + 1e-6)
                low_variance = normalized_variance < self.config.energy_variance_threshold

                if logger.isEnabledFor(logging.DEBUG) and iteration % log_every == 0:
                    logger.debug(
                        "  Convergence: max_move=%.4f stable_iters=%d/%d",
                        max_movement,
                        stable_iterations,
                        stable_threshold
                    )

                # Scale energy threshold by component count for board-size independence
                # More components = higher baseline energy, so scale accordingly
                energy_threshold = self.config.min_movement * 10 * max(1, num_components / 10)
                low_energy = avg_energy < energy_threshold

                # Converge when movement is low AND (variance is low OR energy is low)
                # This prevents false convergence when high forces exist but movement is damped
                converged = low_movement and (low_variance or low_energy)

                # Track consecutive stable iterations for early exit
                # Use movement stability as the primary criterion since energy can oscillate
                # while components are barely moving due to damping
                movement_stable = max_movement < movement_stable_threshold
                if movement_stable:
                    stable_iterations += 1
                else:
                    stable_iterations = 0

                # Early exit: if movement has been low for N iterations, we're converged
                # This prevents long dwell time when algorithm has effectively finished
                if stable_iterations >= stable_threshold:
                    converged = True
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Early convergence: movement stable for %d iterations (max_move=%.4f)",
                            stable_iterations,
                            max_movement
                        )
            else:
                # Before we have energy history, use movement + very low energy as fallback
                # Scale by component count for consistency
                num_components = len(state.positions)
                energy_threshold = self.config.min_movement * max(1, num_components / 10)
                converged = low_movement and state.total_energy < energy_threshold

            if converged:
                state.converged = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Converged at iteration %d: max_move=%.4f energy=%.3f",
                        iteration,
                        max_movement,
                        state.total_energy,
                    )
                break

        # Log warning if max iterations reached without convergence
        if not state.converged:
            logger.warning(
                "Refinement did not converge after %d iterations (energy=%.3f, oscillations=%d). "
                "Consider increasing max_iterations or adjusting force strengths.",
                self.config.max_iterations,
                state.total_energy,
                oscillation_count
            )

        # Apply final positions to board
        self._apply_to_board(state)

        # Capture final state
        if self.visualizer:
            final_forces = self._calculate_all_forces(state)
            self._capture_viz_frame(
                state, final_forces,
                f"Final (iter {state.iteration})",
                phase="final"
            )

        return state

    def _detect_oscillation(self, energy_history: List[float],
                           movement_history: List[float]) -> bool:
        """Detect if the system is oscillating rather than converging.

        Oscillation indicators:
        - Energy bouncing up and down (alternating increases/decreases)
        - Movement staying high but not decreasing
        - Low energy variance but high energy level

        Returns:
            True if oscillation is detected
        """
        if len(energy_history) < 4:
            return False

        # Check for alternating energy pattern (up-down-up or down-up-down)
        diffs = [energy_history[i+1] - energy_history[i]
                 for i in range(len(energy_history) - 1)]
        sign_changes = sum(1 for i in range(len(diffs) - 1)
                          if diffs[i] * diffs[i+1] < 0)

        # High sign changes relative to history length indicates oscillation
        oscillation_ratio = sign_changes / (len(diffs) - 1) if len(diffs) > 1 else 0

        # Check if movement is staying high (not decreasing)
        if len(movement_history) >= 4:
            recent_avg = sum(movement_history[-2:]) / 2
            older_avg = sum(movement_history[-4:-2]) / 2
            movement_stuck = recent_avg > older_avg * 0.9 and recent_avg > self.config.min_movement * 5

            if oscillation_ratio > 0.6 and movement_stuck:
                return True

        # Pure oscillation check: many direction changes
        return oscillation_ratio > 0.7

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return float('inf')
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _summarize_forces(self, forces: Dict[str, List[Force]]) -> Dict[str, object]:
        """Summarize forces for logging."""
        counts: Dict[ForceType, int] = {}
        total_magnitude = 0.0
        total_count = 0
        max_magnitude = 0.0
        max_ref = ""

        for ref, ref_forces in forces.items():
            for force in ref_forces:
                counts[force.force_type] = counts.get(force.force_type, 0) + 1
                total_magnitude += force.magnitude
                total_count += 1
                if force.magnitude > max_magnitude:
                    max_magnitude = force.magnitude
                    max_ref = ref

        avg_magnitude = total_magnitude / total_count if total_count else 0.0

        return {
            "counts": counts,
            "total_count": total_count,
            "avg_magnitude": avg_magnitude,
            "max_magnitude": max_magnitude,
            "max_ref": max_ref,
        }

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
        # Anchors are always locked (top and bottom layers)
        if ref == self._anchor_ref or ref == self._anchor_top or ref == self._anchor_bottom:
            return True

        if not self.config.lock_placed:
            return False
        comp = self.board.components.get(ref)
        return comp.locked if comp else False

    def _setup_anchor(self):
        """Find largest component on each layer and set as anchors at board center.

        The anchors act as fixed points that other components compact around.
        This prevents the "explosion" effect where repulsion pushes everything
        to the edges.

        For two-layer boards:
        - Top layer: largest IC (or component) anchored at center
        - Bottom layer: largest component anchored at center
        """
        from ..board.abstraction import Layer

        # Calculate board center
        outline = self.board.outline
        if outline.has_outline:
            self._center_x = outline.origin_x + outline.width / 2
            self._center_y = outline.origin_y + outline.height / 2
        else:
            # Fall back to centroid of all components
            if self.board.components:
                xs = [c.x for c in self.board.components.values() if not c.dnp]
                ys = [c.y for c in self.board.components.values() if not c.dnp]
                if xs and ys:
                    self._center_x = sum(xs) / len(xs)
                    self._center_y = sum(ys) / len(ys)

        # Separate components by layer
        # Skip locked components - they should not become anchors or be moved
        top_components = []
        bottom_components = []

        for ref, comp in self.board.components.items():
            if comp.dnp:
                continue
            # Skip locked components when selecting anchors
            if self.config.lock_placed and comp.locked:
                continue

            if comp.layer == Layer.BOTTOM_COPPER:
                bottom_components.append((ref, comp))
            else:
                # Default to top layer (TOP_COPPER or through-hole)
                top_components.append((ref, comp))

        # Find largest IC on top layer (prefer ICs, fall back to any largest)
        def find_largest_anchor(components, prefer_ic=True):
            largest_ic_area = 0.0
            largest_ic_ref = None
            largest_any_area = 0.0
            largest_any_ref = None

            for ref, comp in components:
                area = comp.width * comp.height

                # Check if it's an IC (U prefix)
                is_ic = ref.upper().startswith('U')

                if is_ic and area > largest_ic_area:
                    largest_ic_area = area
                    largest_ic_ref = ref

                if area > largest_any_area:
                    largest_any_area = area
                    largest_any_ref = ref

            # Prefer IC if found and prefer_ic is True
            if prefer_ic and largest_ic_ref:
                return largest_ic_ref
            return largest_any_ref

        # Setup top layer anchor (prefer ICs)
        self._anchor_top = find_largest_anchor(top_components, prefer_ic=True)
        if self._anchor_top:
            comp = self.board.components[self._anchor_top]
            comp.x = self._center_x
            comp.y = self._center_y
            logger.info(
                f"Top layer anchor: {self._anchor_top} ({comp.width:.1f}x{comp.height:.1f}mm) "
                f"at center ({self._center_x:.1f}, {self._center_y:.1f})"
            )

        # Setup bottom layer anchor (any largest component)
        self._anchor_bottom = find_largest_anchor(bottom_components, prefer_ic=False)
        if self._anchor_bottom:
            comp = self.board.components[self._anchor_bottom]
            comp.x = self._center_x
            comp.y = self._center_y
            logger.info(
                f"Bottom layer anchor: {self._anchor_bottom} ({comp.width:.1f}x{comp.height:.1f}mm) "
                f"at center ({self._center_x:.1f}, {self._center_y:.1f})"
            )

        # Legacy compatibility - set _anchor_ref to top layer anchor
        self._anchor_ref = self._anchor_top

        # Initial compaction: move distant components closer (per layer)
        self._compact_distant_components()

    def _compact_distant_components(self):
        """Move components outside initial_radius closer to the center.

        This gives the force-directed algorithm a better starting point
        by ensuring all components are within reach of the center attraction.
        Components are placed at the edge of the initial_radius, preserving
        their relative direction from the center.

        Works per-layer: components compact toward their respective layer's anchor.
        """
        from ..board.abstraction import Layer

        if self.config.initial_radius <= 0:
            return

        center_x = self._center_x
        center_y = self._center_y
        radius = self.config.initial_radius
        moved_top = 0
        moved_bottom = 0

        for ref, comp in self.board.components.items():
            # Skip anchors, DNP, and locked components
            if ref == self._anchor_top or ref == self._anchor_bottom or comp.dnp:
                continue
            # Skip locked components - they should not be moved during setup
            if self.config.lock_placed and comp.locked:
                continue

            # Determine which anchor this component relates to based on layer
            is_bottom = comp.layer == Layer.BOTTOM_COPPER
            anchor_ref = self._anchor_bottom if is_bottom else self._anchor_top

            # If no anchor for this layer, use the board center
            if anchor_ref and anchor_ref in self.board.components:
                anchor_comp = self.board.components[anchor_ref]
                target_x, target_y = anchor_comp.x, anchor_comp.y
            else:
                target_x, target_y = center_x, center_y

            # Calculate distance from the target (layer's anchor or center)
            dx = comp.x - target_x
            dy = comp.y - target_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > radius:
                # Move to edge of radius, preserving direction
                if distance > 0.01:
                    # Place at 80% of radius to leave room for spreading
                    new_distance = radius * 0.8
                    scale = new_distance / distance
                    comp.x = target_x + dx * scale
                    comp.y = target_y + dy * scale
                else:
                    # Component at same position as center - add random offset
                    import random
                    angle = random.uniform(0, 2 * math.pi)
                    comp.x = target_x + radius * 0.5 * math.cos(angle)
                    comp.y = target_y + radius * 0.5 * math.sin(angle)

                if is_bottom:
                    moved_bottom += 1
                else:
                    moved_top += 1

        if moved_top > 0 or moved_bottom > 0:
            logger.info(
                f"Compacted distant components within {radius}mm radius: "
                f"top={moved_top}, bottom={moved_bottom}"
            )

    def _calculate_all_forces(self, state: PlacementState
                              ) -> Dict[str, List[Force]]:
        """Calculate all forces on each component."""
        forces: Dict[str, List[Force]] = {ref: [] for ref in state.positions}

        # 1. Repulsion forces (prevent overlap)
        self._add_repulsion_forces(state, forces)

        # 2. Attraction forces (net connectivity)
        self._add_attraction_forces(state, forces)

        # 3. Center attraction (compaction toward anchor/center)
        self._add_center_forces(state, forces)

        # 4. Boundary forces
        self._add_boundary_forces(state, forces)

        # 5. Constraint forces
        self._add_constraint_forces(state, forces)

        # 6. Alignment forces (optional)
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
            comp1 = self.board.components[ref1]
            # Skip DNP (Do Not Populate) components
            if comp1.dnp:
                continue

            x1, y1 = state.positions[ref1]
            half_w1, half_h1 = self._component_sizes[ref1]
            locked1 = self._is_component_locked(ref1)

            for ref2 in refs[i+1:]:
                comp2 = self.board.components[ref2]
                # Skip DNP (Do Not Populate) components
                if comp2.dnp:
                    continue

                x2, y2 = state.positions[ref2]
                half_w2, half_h2 = self._component_sizes[ref2]
                locked2 = self._is_component_locked(ref2)

                # Skip if both locked (neither can move)
                if locked1 and locked2:
                    continue

                # Layer check to avoid unnecessary repulsion between Top/Bottom
                if not (comp1.is_through_hole or comp2.is_through_hole):
                    from ..board.abstraction import Layer
                    
                    is_top1 = comp1.layer == Layer.TOP_COPPER
                    is_bottom1 = comp1.layer == Layer.BOTTOM_COPPER
                    is_top2 = comp2.layer == Layer.TOP_COPPER
                    is_bottom2 = comp2.layer == Layer.BOTTOM_COPPER
                    
                    if (is_top1 and is_bottom2) or (is_bottom1 and is_top2):
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
                # NOTE: Removed weak inverse-distance repulsion for distant components
                # This was causing components to drift to board edges. Now components
                # only repel when overlapping or within preferred clearance.

    def _add_attraction_forces(self, state: PlacementState,
                               forces: Dict[str, List[Force]]):
        """Add attraction forces for connected components.

        Uses a Hybrid Net Model:
        - For small nets (<=3 pins): pairwise attraction between all components
        - For large nets (>3 pins): Star Model - attract all pins to virtual centroid

        The star model reduces O(N²) to O(N) for high-degree nets (GND, VCC)
        and scales attraction by 1/k (where k is pin count) to prevent collapse.

        Multi-pin weighting: Components with multiple pins on the same net
        receive proportionally stronger attraction. An IC with 4 GND pins
        gets 4x the attraction force of a resistor with 1 GND pin.
        """
        for net_name, net in self.board.nets.items():
            if len(net.connections) < 2:
                continue

            # Count pins per component for proper weighting
            # Components with multiple pins on a net should have stronger attraction
            # Skip DNP (Do Not Populate) components
            pin_counts: Dict[str, int] = {}
            for ref, _ in net.connections:
                if ref in state.positions:
                    comp = self.board.components.get(ref)
                    if comp and comp.dnp:
                        continue  # Skip unpopulated components
                    pin_counts[ref] = pin_counts.get(ref, 0) + 1

            refs = list(pin_counts.keys())
            num_refs = len(refs)

            if num_refs < 2:
                continue

            # Base strength, significantly boosted for power/ground nets
            # Decoupling capacitors need to stay close to ICs for good PDN
            base_strength = self.config.attraction_strength
            is_power_net = net.is_power or net.is_ground
            if is_power_net:
                # 10x boost for power nets to keep decoupling caps close
                base_strength *= 10.0

            # Cap effective distance to prevent unbounded attraction on long nets
            max_dist = self.config.max_attraction_distance

            # For power nets, identify capacitor-IC pairs for extra strong attraction
            # Decoupling caps should be within 10mm of their target IC
            cap_refs = []
            ic_refs = []
            if is_power_net:
                for ref in refs:
                    first_char = ref[0].upper() if ref else ''
                    if first_char == 'C':
                        cap_refs.append(ref)
                    elif first_char == 'U':
                        ic_refs.append(ref)

            if num_refs <= 3:
                # Small nets: use pairwise attraction with pin-count weighting
                for i, ref1 in enumerate(refs):
                    x1, y1 = state.positions[ref1]
                    weight1 = pin_counts[ref1]

                    for ref2 in refs[i+1:]:
                        x2, y2 = state.positions[ref2]
                        weight2 = pin_counts[ref2]

                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)

                        if distance < 0.001:
                            # Identical positions - apply random jitter to break symmetry
                            import random
                            dx = random.uniform(-0.5, 0.5)
                            dy = random.uniform(-0.5, 0.5)
                            distance = math.sqrt(dx*dx + dy*dy)

                        if distance < 0.1:
                            continue

                        # Cap effective distance to limit attraction force
                        effective_dist = min(distance, max_dist)

                        # Weight by pin counts: more pins = stronger connection
                        pair_weight = (weight1 + weight2) / 2.0
                        force_mag = base_strength * effective_dist * pair_weight

                        fx = force_mag * dx / distance
                        fy = force_mag * dy / distance

                        forces[ref1].append(Force(fx, fy, ForceType.ATTRACTION,
                                                  f"net_{net_name}"))
                        forces[ref2].append(Force(-fx, -fy, ForceType.ATTRACTION,
                                                  f"net_{net_name}"))
            else:
                # Large nets: use Star Model with virtual centroid
                # Use pin-weighted centroid so multi-pin ICs pull centroid toward them
                total_pins = sum(pin_counts.values())
                centroid_x = sum(state.positions[ref][0] * pin_counts[ref]
                                 for ref in refs) / total_pins
                centroid_y = sum(state.positions[ref][1] * pin_counts[ref]
                                 for ref in refs) / total_pins

                # Scale strength by 1/total_pins to prevent collapse on large nets
                scaled_strength = base_strength / total_pins

                for ref in refs:
                    x, y = state.positions[ref]
                    weight = pin_counts[ref]

                    dx = centroid_x - x
                    dy = centroid_y - y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance < 0.1:
                        continue

                    # Cap effective distance to limit attraction force
                    effective_dist = min(distance, max_dist)

                    # Scale by pin count: multi-pin components get stronger pull
                    force_mag = scaled_strength * effective_dist * weight

                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance

                    forces[ref].append(Force(fx, fy, ForceType.ATTRACTION,
                                             f"net_{net_name}_star"))

            # Extra strong attraction for decoupling capacitors to nearest IC
            # This ensures caps stay close to their target IC for good PDN
            # Best practice: decoupling caps should be within 5mm of IC power pins
            if cap_refs and ic_refs:
                decoupling_strength = self.config.attraction_strength * 100.0  # Very strong
                decoupling_target_dist = 5.0  # Target distance in mm (tighter than before)
                decoupling_max_dist = 10.0  # Maximum acceptable distance

                for cap_ref in cap_refs:
                    cap_x, cap_y = state.positions[cap_ref]

                    # Find nearest IC on this net
                    nearest_ic = None
                    nearest_dist = float('inf')
                    for ic_ref in ic_refs:
                        ic_x, ic_y = state.positions[ic_ref]
                        dx = ic_x - cap_x
                        dy = ic_y - cap_y
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_ic = ic_ref

                    if nearest_ic and nearest_dist > 0.1:
                        ic_x, ic_y = state.positions[nearest_ic]
                        dx = ic_x - cap_x
                        dy = ic_y - cap_y

                        # Apply attraction if cap is further than target distance
                        # Force increases quadratically as distance exceeds target
                        if nearest_dist > decoupling_target_dist:
                            # Quadratic force for stronger pull when far away
                            excess_dist = nearest_dist - decoupling_target_dist
                            # Extra urgency if exceeding max acceptable distance
                            if nearest_dist > decoupling_max_dist:
                                excess_dist *= 2.0  # Double the urgency
                            force_mag = decoupling_strength * excess_dist * (1 + excess_dist / decoupling_target_dist)

                            fx = force_mag * dx / nearest_dist
                            fy = force_mag * dy / nearest_dist

                            forces[cap_ref].append(Force(fx, fy, ForceType.ATTRACTION,
                                                         f"decoupling_{cap_ref}_to_{nearest_ic}"))

    def _add_center_forces(self, state: PlacementState,
                           forces: Dict[str, List[Force]]):
        """Add center attraction forces for compaction.

        Pulls all non-anchor components toward their layer's anchor (or board center).
        This counteracts repulsion and keeps the placement compact instead
        of spreading to the edges.

        The force is weak but constant, creating a gentle inward pressure
        that balances repulsion at equilibrium.

        Layer-aware: Components are attracted to their layer's anchor, ensuring
        top and bottom layer components compact around their respective anchors.
        """
        from ..board.abstraction import Layer

        if self.config.center_strength <= 0:
            return

        for ref, (x, y) in state.positions.items():
            # Skip anchors themselves
            if ref == self._anchor_top or ref == self._anchor_bottom:
                continue

            # Skip locked components
            if self._is_component_locked(ref):
                continue

            # Determine which anchor to attract toward based on layer
            comp = self.board.components.get(ref)
            if not comp:
                continue

            is_bottom = comp.layer == Layer.BOTTOM_COPPER
            anchor_ref = self._anchor_bottom if is_bottom else self._anchor_top

            # Get the target center point for this component
            if anchor_ref and anchor_ref in state.positions:
                center_x, center_y = state.positions[anchor_ref]
            else:
                # Fall back to board center
                center_x, center_y = self._center_x, self._center_y

            dx = center_x - x
            dy = center_y - y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < 0.1:
                continue

            # Linear attraction toward center - stronger when further away
            # This creates gentle compaction pressure
            force_mag = self.config.center_strength * min(distance, 50.0) / 10.0

            fx = force_mag * dx / distance
            fy = force_mag * dy / distance

            forces[ref].append(Force(fx, fy, ForceType.ATTRACTION, "center"))

    def _add_boundary_forces(self, state: PlacementState,
                             forces: Dict[str, List[Force]]):
        """Add forces to keep components within board boundaries.

        For rectangular outlines: Uses AABB (Axis-Aligned Bounding Box) instead
        of diagonal radius for more accurate boundary detection.

        For polygon outlines: Uses point-in-polygon test for each corner of
        the component's AABB. This properly handles complex board shapes,
        cutouts, and non-rectangular outlines.

        Forces are accumulated (+=) rather than overwritten to properly
        handle corner violations where both X and Y boundaries are exceeded.

        Skipped when board has no explicit outline defined.
        """
        outline = self.board.outline

        # Skip boundary forces when no explicit outline is defined
        if not outline.has_outline:
            return

        for ref, (x, y) in state.positions.items():
            # Skip DNP (Do Not Populate) components - they don't need boundary checking
            comp = self.board.components.get(ref)
            if comp and comp.dnp:
                continue

            # Get component dimensions (accounting for rotation)
            half_w, half_h = self._component_sizes[ref]
            # Use edge_clearance to match validator's min_trace_to_edge
            clearance = self.config.edge_clearance

            fx, fy = 0.0, 0.0

            # For polygon outlines, use contains_point for proper boundary check
            if outline.polygon:
                # Check each corner of the component's AABB
                corners = [
                    (x - half_w, y - half_h),  # bottom-left
                    (x + half_w, y - half_h),  # bottom-right
                    (x - half_w, y + half_h),  # top-left
                    (x + half_w, y + half_h),  # top-right
                ]

                # Also check center point
                if not outline.contains_point(x, y, margin=clearance):
                    # Component center is outside or too close to edge
                    # Push toward board center
                    bbox = outline.get_bounding_box()
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    dx = center_x - x
                    dy = center_y - y
                    dist = math.sqrt(dx * dx + dy * dy) + 0.001
                    fx += self.config.boundary_strength * dx / dist * 2
                    fy += self.config.boundary_strength * dy / dist * 2

                # Check corners
                for cx, cy in corners:
                    if not outline.contains_point(cx, cy, margin=clearance):
                        # This corner is outside - push component inward
                        # Calculate centroid direction
                        bbox = outline.get_bounding_box()
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        dx = center_x - cx
                        dy = center_y - cy
                        dist = math.sqrt(dx * dx + dy * dy) + 0.001
                        fx += self.config.boundary_strength * dx / dist
                        fy += self.config.boundary_strength * dy / dist
            else:
                # Rectangular outline - use simple AABB checks
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

    def _clamp_to_boundary(self, state: PlacementState):
        """Hard clamp components to stay within board boundaries.

        This is a safety net to ensure components never go off-board,
        even if repulsion forces are stronger than boundary forces.
        Boundary forces are the soft constraint; this is the hard constraint.

        For polygon outlines: Uses point-in-polygon test and iteratively
        moves component toward board centroid until all corners are inside.

        Uses edge_clearance to match DFM min_trace_to_edge requirement.

        Skipped when board has no explicit outline defined.
        """
        outline = self.board.outline

        if not outline.has_outline:
            return

        # Use edge_clearance to match validator's min_trace_to_edge
        clearance = self.config.edge_clearance

        for ref, (x, y) in state.positions.items():
            # Get component dimensions (accounting for rotation)
            half_w, half_h = self._component_sizes.get(ref, (1.0, 1.0))

            if outline.polygon:
                # For polygon outlines, iteratively move toward centroid
                bbox = outline.get_bounding_box()
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Check all corners with margin
                corners = [
                    (x - half_w, y - half_h),
                    (x + half_w, y - half_h),
                    (x - half_w, y + half_h),
                    (x + half_w, y + half_h),
                ]

                # Iteratively move toward center until all corners are inside
                for _ in range(20):  # Max iterations
                    all_inside = True
                    for cx, cy in corners:
                        if not outline.contains_point(cx, cy, margin=clearance):
                            all_inside = False
                            break

                    if all_inside:
                        break

                    # Move 10% toward center
                    dx = center_x - x
                    dy = center_y - y
                    x += dx * 0.1
                    y += dy * 0.1

                    # Update corners
                    corners = [
                        (x - half_w, y - half_h),
                        (x + half_w, y - half_h),
                        (x - half_w, y + half_h),
                        (x + half_w, y + half_h),
                    ]

                state.positions[ref] = (x, y)
            else:
                # Rectangular outline - use simple AABB clamp
                min_x = outline.origin_x + half_w + clearance
                max_x = outline.origin_x + outline.width - half_w - clearance
                min_y = outline.origin_y + half_h + clearance
                max_y = outline.origin_y + outline.height - half_h - clearance

                # Ensure there's valid space (board might be too small)
                if max_x < min_x:
                    # Board too narrow - center the component
                    x = outline.origin_x + outline.width / 2
                else:
                    x = max(min_x, min(max_x, x))

                if max_y < min_y:
                    # Board too short - center the component
                    y = outline.origin_y + outline.height / 2
                else:
                    y = max(min_y, min(max_y, y))

                state.positions[ref] = (x, y)

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
                      forces: Dict[str, List[Force]],
                      damping_override: Optional[float] = None,
                      max_velocity_override: Optional[float] = None) -> float:
        """Apply forces to update velocities and positions.

        Respects lock_placed config: locked components don't move.

        Args:
            state: Current placement state
            forces: Forces to apply
            damping_override: Override damping value (for adaptive damping)
            max_velocity_override: Override max velocity (for adaptive damping)

        Returns:
            Maximum movement of any component this iteration
        """
        max_movement = 0.0
        damping = damping_override if damping_override is not None else self.config.damping
        max_vel = max_velocity_override if max_velocity_override is not None else self.config.max_velocity

        for ref in state.positions:
            # Skip locked components
            if self._is_component_locked(ref):
                state.velocities[ref] = (0.0, 0.0)
                continue

            # Skip DNP (Do Not Populate) components - they don't need position updates
            comp = self.board.components.get(ref)
            if comp and comp.dnp:
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
            vx = (vx + (total_fx / mass) * self.config.time_step) * damping
            vy = (vy + (total_fy / mass) * self.config.time_step) * damping

            # Clamp velocity
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > max_vel:
                scale = max_vel / speed
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

        Returns (half_width, half_height) tuples accounting for rotation and
        pad extents. Uses get_bounding_box_with_pads() to include pads that
        extend beyond the component body (e.g., edge-mounted connectors).
        """
        sizes = {}
        for ref, comp in self.board.components.items():
            # Use pad-inclusive bounding box for accurate collision detection
            # This handles connectors, edge-launch parts, and irregular footprints
            bbox = comp.get_bounding_box_with_pads()
            min_x, min_y, max_x, max_y = bbox

            # Compute half-dimensions from bounding box
            # Account for asymmetric pad extents by taking max distance from center
            half_w = max(abs(max_x - comp.x), abs(comp.x - min_x))
            half_h = max(abs(max_y - comp.y), abs(comp.y - min_y))

            # Ensure minimum size to prevent division issues
            half_w = max(half_w, 0.1)
            half_h = max(half_h, 0.1)

            sizes[ref] = (half_w, half_h)
        return sizes

    def _update_component_sizes(self, state: PlacementState):
        """Update component sizes based on current rotation state.

        Called after rotation constraints are applied to keep AABB
        dimensions accurate for boundary and collision detection.

        Note: For rotation updates, we use the body dimensions plus a margin
        to approximate pad extents. The initial _compute_component_sizes uses
        get_bounding_box_with_pads() for accurate initial sizing. This is a
        reasonable approximation since rotations during placement are typically
        90-degree increments and the initial pad-inclusive size captures the
        worst case.
        """
        for ref, rotation in state.rotations.items():
            comp = self.board.components.get(ref)
            if not comp:
                continue

            # Use pad-inclusive bounding box for accurate sizing
            # Temporarily update component rotation to get correct pad positions
            original_rotation = comp.rotation
            comp.rotation = rotation

            bbox = comp.get_bounding_box_with_pads()
            min_x, min_y, max_x, max_y = bbox

            # Restore original rotation
            comp.rotation = original_rotation

            # Compute half-dimensions from bounding box
            half_w = max(abs(max_x - comp.x), abs(comp.x - min_x))
            half_h = max(abs(max_y - comp.y), abs(comp.y - min_y))

            # Ensure minimum size
            half_w = max(half_w, 0.1)
            half_h = max(half_h, 0.1)

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
