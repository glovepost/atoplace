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

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable, Iterable
from enum import Enum

from ..board.abstraction import Board, Component, Net

logger = logging.getLogger(__name__)


def _deterministic_jitter(key: str, scale: float = 0.5) -> Tuple[float, float]:
    """Generate deterministic jitter based on a string key.

    Uses MD5 hash to produce reproducible pseudo-random offsets,
    ensuring placement results are consistent across runs.

    Args:
        key: String to hash (e.g., "ref1_ref2" for component pairs)
        scale: Maximum offset magnitude (default 0.5mm)

    Returns:
        (dx, dy) offset tuple in range [-scale, scale]
    """
    h = hashlib.md5(key.encode()).hexdigest()
    # Use first 8 hex chars for x, next 8 for y
    x_val = int(h[:8], 16) / 0xFFFFFFFF  # Normalize to [0, 1]
    y_val = int(h[8:16], 16) / 0xFFFFFFFF
    # Convert to [-scale, scale]
    return (x_val * 2 * scale - scale, y_val * 2 * scale - scale)


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

    # Decoupling capacitor placement
    decoupling_strength_multiplier: float = 100.0  # Multiplier on attraction_strength for decoupling caps
    decoupling_target_distance: float = 5.0  # mm - target distance from IC power pins
    decoupling_max_distance: float = 10.0  # mm - maximum acceptable distance (triggers stronger pull)

    # Grid alignment
    grid_size: float = 0.0  # 0 = no grid snapping
    snap_to_grid: bool = False

    # Component-specific
    lock_placed: bool = False  # Don't move already-placed components

    # Anchor/center mode
    auto_anchor_largest_ic: bool = True  # Auto-lock largest IC at center as anchor
    initial_radius: float = 30.0  # mm - move distant components within this radius at start

    # Module cohesion
    module_cohesion_strength: float = 1.0  # Base cohesion force per module level
    module_cohesion_depth_multiplier: float = 1.4  # Stronger pull for deeper modules
    module_compact_clearance: float = 0.02  # Preferred clearance above min for same-module pairs
    module_bbox_strength: float = 1.1  # Extra bbox shrink for module grouping constraints

    # Discrete rotation search
    enable_discrete_rotation: bool = True  # Enable discrete rotation optimization
    rotation_search_interval: int = 10  # Iterations between rotation passes
    rotation_angles: Tuple[int, ...] = (0, 90, 180, 270)
    rotation_overlap_weight: float = 1.0  # Overlap penalty weight
    rotation_boundary_weight: float = 5.0  # Boundary penalty weight


class PlacementSpatialGrid:
    """Spatial grid for O(N) neighbor lookups in repulsion calculations.

    Instead of checking all N*(N-1)/2 component pairs (O(N²)), we bin components
    into grid cells and only check pairs in adjacent cells. For uniformly
    distributed components with cell_size = repulsion_cutoff, this reduces
    the average case to O(N).

    Cell size is set to repulsion_cutoff so that any two components that could
    potentially repel each other must be in the same cell or adjacent cells.
    """

    def __init__(self, cell_size: float):
        """Initialize spatial grid.

        Args:
            cell_size: Size of each grid cell (should be >= repulsion_cutoff)
        """
        self.cell_size = cell_size
        self._cells: Dict[Tuple[int, int], List[str]] = {}
        self._positions: Dict[str, Tuple[float, float]] = {}

    def _cell_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to cell key."""
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def clear(self):
        """Clear the grid for rebuilding."""
        self._cells.clear()
        self._positions.clear()

    def insert(self, ref: str, x: float, y: float):
        """Insert a component into the grid."""
        cell = self._cell_key(x, y)
        if cell not in self._cells:
            self._cells[cell] = []
        self._cells[cell].append(ref)
        self._positions[ref] = (x, y)

    def get_neighbors(self, ref: str) -> List[str]:
        """Get all components in adjacent cells (potential collision candidates).

        Returns components in 3x3 neighborhood around the component's cell.
        Only returns components with ref > input ref to avoid duplicate pairs.
        Used by repulsion force calculation which processes each pair once.
        """
        if ref not in self._positions:
            return []

        x, y = self._positions[ref]
        cx, cy = self._cell_key(x, y)

        neighbors = []
        # Check 3x3 neighborhood
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell in self._cells:
                    for other_ref in self._cells[cell]:
                        # Only return refs > current ref to avoid duplicates
                        if other_ref > ref:
                            neighbors.append(other_ref)

        return neighbors

    def get_all_neighbors(self, ref: str) -> List[str]:
        """Get ALL components in adjacent cells, regardless of ref ordering.

        Returns all components in 3x3 neighborhood except self.
        Used by rotation penalty calculation which needs to check all neighbors.
        """
        if ref not in self._positions:
            return []

        x, y = self._positions[ref]
        cx, cy = self._cell_key(x, y)

        neighbors = []
        # Check 3x3 neighborhood
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell in self._cells:
                    for other_ref in self._cells[cell]:
                        if other_ref != ref:
                            neighbors.append(other_ref)

        return neighbors


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

        # Cache for per-pair spacing derived from nets/DFM
        self._pair_clearance_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

        # Module grouping caches
        self._module_groups: Dict[str, Set[str]] = {}
        self._module_hierarchy: Dict[str, Set[str]] = {}
        self._module_depths: Dict[str, int] = {}
        self._module_spreads: Dict[str, float] = {}
        self._module_tree_built = False

        # Rotation helpers
        self._rotation_constraint_refs: Set[str] = set()
        self._rotation_extents_cache: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}

        # Spatial grid for O(N) repulsion neighbor lookups (Issue #23)
        # Cell size must be large enough that overlapping components are always
        # in adjacent cells. This means: cell_size >= cutoff + 2*max_half_diagonal
        max_half_diag = 0.0
        for half_w, half_h in self._component_sizes.values():
            diag = math.sqrt(half_w * half_w + half_h * half_h)
            max_half_diag = max(max_half_diag, diag)
        # Add small margin for floating-point safety
        self._repulsion_cell_size = self.config.repulsion_cutoff + 2 * max_half_diag + 0.1
        self._repulsion_grid = PlacementSpatialGrid(self._repulsion_cell_size)

    def add_constraint(self, constraint: 'PlacementConstraint'):
        """Add a placement constraint."""
        self.constraints.append(constraint)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Added constraint: %s", getattr(constraint, "description", constraint))

    def _compute_overlaps_from_state(
        self,
        state: PlacementState,
        clearance: float = 0.1
    ) -> List[Tuple[str, str, float]]:
        """Compute component overlaps using positions from simulation state.

        Unlike board.find_overlaps() which uses stale board component positions,
        this method uses the current simulated positions from state.positions.

        Args:
            state: Current placement state with simulated positions
            clearance: Minimum clearance between components (mm)

        Returns:
            List of (ref1, ref2, penetration_depth) tuples
        """
        overlaps = []
        refs = list(state.positions.keys())

        for i, ref1 in enumerate(refs):
            comp1 = self.board.components.get(ref1)
            if not comp1 or comp1.dnp:
                continue

            x1, y1 = state.positions[ref1]
            rot1 = state.rotations.get(ref1, comp1.rotation)
            hw1, hh1 = comp1.width / 2, comp1.height / 2

            # Compute rotated AABB for comp1
            rad1 = math.radians(rot1)
            cos1, sin1 = abs(math.cos(rad1)), abs(math.sin(rad1))
            half_w1 = hw1 * cos1 + hh1 * sin1
            half_h1 = hw1 * sin1 + hh1 * cos1

            # Bounding box with clearance
            bb1 = (x1 - half_w1 - clearance, y1 - half_h1 - clearance,
                   x1 + half_w1 + clearance, y1 + half_h1 + clearance)

            for ref2 in refs[i + 1:]:
                comp2 = self.board.components.get(ref2)
                if not comp2 or comp2.dnp:
                    continue

                x2, y2 = state.positions[ref2]
                rot2 = state.rotations.get(ref2, comp2.rotation)
                hw2, hh2 = comp2.width / 2, comp2.height / 2

                # Compute rotated AABB for comp2
                rad2 = math.radians(rot2)
                cos2, sin2 = abs(math.cos(rad2)), abs(math.sin(rad2))
                half_w2 = hw2 * cos2 + hh2 * sin2
                half_h2 = hw2 * sin2 + hh2 * cos2

                bb2 = (x2 - half_w2, y2 - half_h2,
                       x2 + half_w2, y2 + half_h2)

                # Check AABB intersection
                overlap_x = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
                overlap_y = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])

                if overlap_x > 0 and overlap_y > 0:
                    # Penetration depth is minimum separation needed
                    penetration = min(overlap_x, overlap_y)
                    overlaps.append((ref1, ref2, penetration))

        return overlaps

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
            # Store only mutable state (position, rotation)
            # Width/height are in visualizer.static_props
            components[ref] = (x, y, rot)

            # Extract pads - compute pad positions in board coordinates
            # Pad.x,y is already relative to component center (centroid)
            # so we only need to apply rotation and translate to board coordinates
            pad_list = []
            for pad in comp.pads:
                # Pad coordinates are centroid-relative, apply rotation then translate
                px = pad.x
                py = pad.y
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

        # Find overlaps using simulated positions (not stale board positions)
        overlap_pairs = self._compute_overlaps_from_state(state, clearance=0.1)

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
        # Ensure module constraints are available before starting
        self._add_module_constraints_if_missing()
        self._rotation_constraint_refs = self._collect_rotation_constraint_refs()

        # Initialize state
        state = self._initialize_state()
        # Reset caches per run
        self._pair_clearance_cache = {}
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
            if self.config.module_cohesion_strength > 0 and self.modules:
                active_forces.append("module-cohesion")
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

            # Optional discrete rotation search
            self._apply_discrete_rotation_search(state, iteration)

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
                # Guard against zero/near-zero energy which would make variance meaningless
                if avg_energy > 1e-6:
                    normalized_variance = energy_variance / avg_energy
                else:
                    normalized_variance = 0.0  # Zero energy means converged
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
        """Calculate variance of a list of values.

        Returns 0.0 for empty/single-element lists to avoid division issues
        in callers that normalize by variance.
        """
        if len(values) < 2:
            return 0.0  # Return 0 instead of inf to avoid division issues
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

    def _module_key(self, ref: str) -> Optional[str]:
        """Return the module name for a component (None if unassigned)."""
        module = self.modules.get(ref)
        if not module or module == "unassigned":
            return None
        return module

    def _ensure_module_groups(self):
        """Build module grouping and hierarchy caches from self.modules."""
        if self._module_tree_built:
            return

        groups: Dict[str, Set[str]] = {}
        for ref, module in self.modules.items():
            if not module or module == "unassigned":
                continue
            if ref not in self.board.components:
                continue
            groups.setdefault(module, set()).add(ref)

        hierarchy: Dict[str, Set[str]] = {}
        depths: Dict[str, int] = {}
        # Use sorted() for deterministic iteration order (Issue #22)
        for module_name, refs in sorted(groups.items()):
            parts = module_name.split(".")
            for depth in range(1, len(parts) + 1):
                parent = ".".join(parts[:depth])
                hierarchy.setdefault(parent, set()).update(refs)
                depths[parent] = parent.count(".")

        self._module_groups = groups
        self._module_hierarchy = hierarchy
        self._module_depths = depths
        self._module_tree_built = True

    def _estimate_module_spread(self, refs: Iterable[str]) -> float:
        """Estimate grouping radius based on component areas and clearance."""
        total_area = 0.0
        for ref in refs:
            comp = self.board.components.get(ref)
            if not comp:
                continue
            bbox = comp.get_bounding_box_with_pads()
            width = max(0.1, bbox[2] - bbox[0])
            height = max(0.1, bbox[3] - bbox[1])
            total_area += width * height + (self.config.min_clearance * 4)
        if total_area <= 0:
            return 15.0
        radius = math.sqrt(total_area / math.pi)
        return max(10.0, radius * 1.6)

    def _add_module_constraints_if_missing(self):
        """Auto-add grouping constraints for modules when not already present."""
        from .constraints import GroupingConstraint

        self._ensure_module_groups()
        if not self._module_groups:
            return

        existing_groups = []
        for constraint in self.constraints:
            if isinstance(constraint, GroupingConstraint):
                existing_groups.append(set(constraint.components))

        # Use sorted() for deterministic iteration order (Issue #22)
        for module_name, refs in sorted(self._module_groups.items()):
            if len(refs) < 2:
                continue
            ref_set = set(refs)
            if any(ref_set == existing for existing in existing_groups):
                continue

            spread = self._estimate_module_spread(refs)
            constraint = GroupingConstraint(
                components=sorted(refs),
                max_spread=spread,
                optimize_bbox=True,
                bbox_strength=self.config.module_bbox_strength,
                min_clearance=self.config.min_clearance,
                description=f"Group module: {module_name}",
            )
            self.constraints.append(constraint)
            existing_groups.append(ref_set)

    def _add_module_cohesion_forces(self, state: PlacementState,
                                    forces: Dict[str, List[Force]]):
        """Add hierarchical cohesion forces for module groups."""
        if self.config.module_cohesion_strength <= 0:
            return

        self._ensure_module_groups()
        if not self._module_hierarchy:
            return

        # Use sorted() for deterministic iteration order (Issue #22)
        for module_name, refs in sorted(self._module_hierarchy.items()):
            active_refs = []
            for ref in sorted(refs):
                comp = self.board.components.get(ref)
                if not comp or comp.dnp:
                    continue
                if ref not in state.positions:
                    continue
                active_refs.append(ref)

            if len(active_refs) < 2:
                continue

            spread = self._module_spreads.get(module_name)
            if spread is None:
                spread = self._estimate_module_spread(active_refs)
                self._module_spreads[module_name] = spread

            cx = sum(state.positions[ref][0] for ref in active_refs) / len(active_refs)
            cy = sum(state.positions[ref][1] for ref in active_refs) / len(active_refs)
            depth = self._module_depths.get(module_name, 0)
            strength = self.config.module_cohesion_strength * (
                self.config.module_cohesion_depth_multiplier ** depth
            )

            for ref in active_refs:
                if self._is_component_locked(ref):
                    continue
                x, y = state.positions[ref]
                dx = cx - x
                dy = cy - y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance <= spread / 2 or distance < 0.1:
                    continue
                force_mag = strength * (distance - spread / 2) / distance
                forces[ref].append(Force(force_mag * dx, force_mag * dy,
                                         ForceType.CONSTRAINT,
                                         f"module_{module_name}"))

    def _collect_rotation_constraint_refs(self) -> Set[str]:
        """Collect refs that have explicit rotation constraints."""
        constrained: Set[str] = set()
        for constraint in self.constraints:
            if hasattr(constraint, 'get_target_rotation'):
                target_rot = constraint.get_target_rotation()
                if target_rot is not None:
                    ref = getattr(constraint, 'component_ref', None)
                    if ref:
                        constrained.add(ref)
            elif hasattr(constraint, 'rotation') and constraint.rotation is not None:
                ref = getattr(constraint, 'component_ref', None)
                if ref:
                    constrained.add(ref)
        return constrained

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
                    # Component at same position as center - add deterministic offset
                    jitter_dx, jitter_dy = _deterministic_jitter(ref, scale=radius * 0.5)
                    comp.x = target_x + jitter_dx
                    comp.y = target_y + jitter_dy

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

        # 3. Module cohesion forces (hierarchical grouping)
        self._add_module_cohesion_forces(state, forces)

        # 4. Center attraction (compaction toward anchor/center)
        self._add_center_forces(state, forces)

        # 5. Boundary forces
        self._add_boundary_forces(state, forces)

        # 6. Constraint forces
        self._add_constraint_forces(state, forces)

        # 7. Alignment forces (optional)
        if self.config.snap_to_grid and self.config.grid_size > 0:
            self._add_alignment_forces(state, forces)

        return forces

    def _get_pair_clearance(self, ref1: str, ref2: str) -> Tuple[float, float]:
        """Compute min/preferred clearance between two components using net rules."""
        key = tuple(sorted((ref1, ref2)))
        if key in self._pair_clearance_cache:
            return self._pair_clearance_cache[key]

        min_clearance = self.config.min_clearance
        preferred = self.config.preferred_clearance

        nets_between = self.board.get_nets_between(ref1, ref2)
        for net in nets_between:
            if net.clearance is not None:
                min_clearance = max(min_clearance, net.clearance)
                preferred = max(preferred, net.clearance * 2)
            if net.trace_width is not None:
                preferred = max(preferred, net.trace_width + min_clearance)
            if net.is_differential:
                preferred = max(preferred, min_clearance * 1.2)

        # Ensure preferred >= min clearance by a small margin
        preferred = max(preferred, min_clearance + 0.05)

        # Allow tighter spacing within the same module
        module1 = self._module_key(ref1)
        module2 = self._module_key(ref2)
        if module1 and module1 == module2:
            preferred = max(
                min_clearance,
                min(preferred, min_clearance + self.config.module_compact_clearance)
            )

        self._pair_clearance_cache[key] = (min_clearance, preferred)
        return min_clearance, preferred

    def _add_repulsion_forces(self, state: PlacementState,
                              forces: Dict[str, List[Force]]):
        """Add repulsion forces between components.

        Uses AABB overlap detection for more accurate collision detection
        with rectangular components. Also uses preferred_clearance for spacing
        and cutoff distance to avoid long-range force accumulation.

        When one component is locked, the non-locked component receives double
        the repulsion force to fully resolve the overlap.

        Performance: Uses spatial grid binning for O(N) neighbor lookups (Issue #23).
        Components are binned by center position. The cell size is set large enough
        that any two components that could overlap or be within cutoff are guaranteed
        to be in adjacent cells.
        """
        # Rebuild spatial grid with current positions
        self._repulsion_grid.clear()
        for ref, (x, y) in state.positions.items():
            comp = self.board.components.get(ref)
            if comp and not comp.dnp:
                self._repulsion_grid.insert(ref, x, y)

        # Cutoff distance beyond which repulsion is negligible (reduces noise)
        cutoff_distance = self.config.repulsion_cutoff

        for ref1 in state.positions:
            comp1 = self.board.components.get(ref1)
            # Skip DNP (Do Not Populate) components
            if not comp1 or comp1.dnp:
                continue

            x1, y1 = state.positions[ref1]
            half_w1, half_h1 = self._component_sizes[ref1]
            locked1 = self._is_component_locked(ref1)

            # Use spatial grid for O(N) neighbor lookup instead of O(N²) all-pairs
            for ref2 in self._repulsion_grid.get_neighbors(ref1):
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

                pair_min_clearance, pair_preferred = self._get_pair_clearance(ref1, ref2)

                # AABB overlap detection - check if bounding boxes overlap
                # along each axis independently
                # NOTE: Check overlap BEFORE cutoff to handle large components (Issue #19)
                overlap_x = (half_w1 + half_w2 + pair_min_clearance) - abs(dx)
                overlap_y = (half_h1 + half_h2 + pair_min_clearance) - abs(dy)

                # Preferred separation for comfortable spacing
                preferred_sep_x = half_w1 + half_w2 + pair_preferred
                preferred_sep_y = half_h1 + half_h2 + pair_preferred

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
                    # Apply cutoff only to spacing forces, not overlap forces (Issue #19)
                    if distance > cutoff_distance:
                        continue

                    # Use max shortfall to ensure proper spacing on the tighter axis
                    shortfall_x = preferred_sep_x - abs(dx)
                    shortfall_y = preferred_sep_y - abs(dy)
                    max_shortfall = max(pair_preferred - pair_min_clearance, 0.01)

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

            # Count pins per component for proper weighting, grouped by module
            pin_counts_by_module: Dict[str, Dict[str, int]] = {}
            for ref, _ in net.connections:
                if ref not in state.positions:
                    continue
                comp = self.board.components.get(ref)
                if comp and comp.dnp:
                    continue
                module_key = self._module_key(ref) if self.modules else None
                if module_key is None:
                    module_key = "__unassigned__"
                pin_counts = pin_counts_by_module.setdefault(module_key, {})
                pin_counts[ref] = pin_counts.get(ref, 0) + 1

            for module_key, pin_counts in pin_counts_by_module.items():
                refs = list(pin_counts.keys())
                num_refs = len(refs)
                if num_refs < 2:
                    continue

                # Base strength, significantly boosted for power/ground nets
                base_strength = self.config.attraction_strength
                is_power_net = net.is_power or net.is_ground
                if is_power_net:
                    base_strength *= 10.0

                # Respect per-net rules: larger clearance/width -> stronger attraction
                if net.clearance is not None and self.config.min_clearance > 0:
                    clearance_factor = max(1.0, net.clearance / self.config.min_clearance)
                    base_strength *= clearance_factor
                if net.trace_width is not None and self.board.default_trace_width > 0:
                    width_factor = max(1.0, net.trace_width / self.board.default_trace_width)
                    base_strength *= min(width_factor, 3.0)

                # Cap effective distance to prevent unbounded attraction on long nets
                max_dist = self.config.max_attraction_distance

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
                                jitter_key = f"{ref1}_{ref2}_{net_name}"
                                dx, dy = _deterministic_jitter(jitter_key)
                                distance = math.sqrt(dx*dx + dy*dy)

                            if distance < 0.1:
                                continue

                            effective_dist = min(distance, max_dist)
                            pair_weight = (weight1 + weight2) / 2.0
                            force_mag = base_strength * effective_dist * pair_weight

                            fx = force_mag * dx / distance
                            fy = force_mag * dy / distance

                            forces[ref1].append(Force(fx, fy, ForceType.ATTRACTION,
                                                      f"net_{net_name}"))
                            forces[ref2].append(Force(-fx, -fy, ForceType.ATTRACTION,
                                                      f"net_{net_name}"))
                else:
                    total_pins = sum(pin_counts.values())
                    if total_pins == 0:
                        continue

                    centroid_x = sum(state.positions[ref][0] * pin_counts[ref]
                                     for ref in refs) / total_pins
                    centroid_y = sum(state.positions[ref][1] * pin_counts[ref]
                                     for ref in refs) / total_pins

                    scaled_strength = base_strength / total_pins

                    for ref in refs:
                        x, y = state.positions[ref]
                        weight = pin_counts[ref]

                        dx = centroid_x - x
                        dy = centroid_y - y
                        distance = math.sqrt(dx*dx + dy*dy)

                        if distance < 0.1:
                            continue

                        effective_dist = min(distance, max_dist)
                        force_mag = scaled_strength * effective_dist * weight

                        fx = force_mag * dx / distance
                        fy = force_mag * dy / distance

                        forces[ref].append(Force(fx, fy, ForceType.ATTRACTION,
                                                 f"net_{net_name}_star"))

                if cap_refs and ic_refs:
                    decoupling_strength = (self.config.attraction_strength *
                                          self.config.decoupling_strength_multiplier)
                    decoupling_target_dist = self.config.decoupling_target_distance
                    decoupling_max_dist = self.config.decoupling_max_distance

                    for cap_ref in cap_refs:
                        cap_x, cap_y = state.positions[cap_ref]

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

                        if nearest_ic and nearest_dist > 0.1 and nearest_ic in state.positions:
                            ic_x, ic_y = state.positions[nearest_ic]
                            dx = ic_x - cap_x
                            dy = ic_y - cap_y

                            if nearest_dist > decoupling_target_dist:
                                excess_dist = nearest_dist - decoupling_target_dist
                                if nearest_dist > decoupling_max_dist:
                                    excess_dist *= 2.0
                                force_mag = decoupling_strength * excess_dist * (
                                    1 + excess_dist / decoupling_target_dist
                                )

                                fx = force_mag * dx / nearest_dist
                                fy = force_mag * dy / nearest_dist

                                forces[cap_ref].append(Force(
                                    fx, fy, ForceType.ATTRACTION,
                                    f"decoupling_{cap_ref}_to_{nearest_ic}"
                                ))

        # Differential pairs: pull paired nets toward a shared midpoint to keep lengths tight
        self._add_diff_pair_forces(state, forces)

    def _add_diff_pair_forces(self, state: PlacementState,
                              forces: Dict[str, List[Force]]):
        """Add extra attraction for differential pair nets to keep them co-located."""
        processed: Set[Tuple[str, str]] = set()

        for net_name, net in self.board.nets.items():
            partner = net.diff_pair_net
            if not partner or partner not in self.board.nets:
                continue
            key = tuple(sorted((net_name, partner)))
            if key in processed:
                continue
            processed.add(key)

            net_a = self.board.nets[net_name]
            net_b = self.board.nets[partner]

            def _centroid(net_obj: Net, refs: Set[str]) -> Tuple[float, float]:
                pin_counts: Dict[str, int] = {}
                for ref, _ in net_obj.connections:
                    if ref in refs and ref in state.positions:
                        pin_counts[ref] = pin_counts.get(ref, 0) + 1
                if not pin_counts:
                    return (0.0, 0.0)
                total = sum(pin_counts.values())
                if total == 0:
                    return (0.0, 0.0)
                cx = sum(state.positions[ref][0] * cnt for ref, cnt in pin_counts.items()) / total
                cy = sum(state.positions[ref][1] * cnt for ref, cnt in pin_counts.items()) / total
                return (cx, cy)

            refs_a = {ref for ref, _ in net_a.connections if ref in state.positions}
            refs_b = {ref for ref, _ in net_b.connections if ref in state.positions}
            if not refs_a or not refs_b:
                continue

            module_groups: Dict[str, Tuple[Set[str], Set[str]]] = {}
            if self.modules:
                # Use sorted() for deterministic iteration (Issue #22)
                for ref in sorted(refs_a):
                    key = self._module_key(ref) or "__unassigned__"
                    module_groups.setdefault(key, (set(), set()))[0].add(ref)
                for ref in sorted(refs_b):
                    key = self._module_key(ref) or "__unassigned__"
                    module_groups.setdefault(key, (set(), set()))[1].add(ref)
            else:
                module_groups["__all__"] = (set(refs_a), set(refs_b))

            # Use sorted() for deterministic iteration (Issue #22)
            for module_key, (group_a, group_b) in sorted(module_groups.items()):
                if not group_a or not group_b:
                    continue

                cx_a, cy_a = _centroid(net_a, group_a)
                cx_b, cy_b = _centroid(net_b, group_b)
                midpoint = ((cx_a + cx_b) / 2, (cy_a + cy_b) / 2)
                centroid_distance = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                target_distance = max(self.config.min_clearance * 2, centroid_distance * 0.5)

                refs = group_a | group_b
                if len(refs) < 2:
                    continue

                # Use sorted() for deterministic force accumulation (Issue #22)
                for ref in sorted(refs):
                    x, y = state.positions[ref]
                    dx = midpoint[0] - x
                    dy = midpoint[1] - y
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance < 0.1:
                        continue

                    force_mag = self.config.attraction_strength * 0.5 * min(
                        self.config.max_attraction_distance, max(distance, target_distance)
                    )
                    fx = force_mag * dx / distance
                    fy = force_mag * dy / distance
                    forces[ref].append(Force(fx, fy, ForceType.ATTRACTION,
                                             f"diff_pair_{net_name}_{partner}"))

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

        # Cache bounding box and center for polygon outlines (avoids repeated calculation)
        polygon_bbox = None
        polygon_center_x = 0.0
        polygon_center_y = 0.0
        if outline.polygon:
            polygon_bbox = outline.get_bounding_box()
            polygon_center_x = (polygon_bbox[0] + polygon_bbox[2]) / 2
            polygon_center_y = (polygon_bbox[1] + polygon_bbox[3]) / 2

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
                # Check corners and edge midpoints to catch concave intrusions (Issue #20)
                # Corners alone can miss when component edges cross concave boundaries
                check_points = [
                    # Corners
                    (x - half_w, y - half_h),  # bottom-left
                    (x + half_w, y - half_h),  # bottom-right
                    (x - half_w, y + half_h),  # top-left
                    (x + half_w, y + half_h),  # top-right
                    # Edge midpoints
                    (x, y - half_h),           # bottom edge midpoint
                    (x, y + half_h),           # top edge midpoint
                    (x - half_w, y),           # left edge midpoint
                    (x + half_w, y),           # right edge midpoint
                ]

                # Also check center point
                if not outline.contains_point(x, y, margin=clearance):
                    # Component center is outside or too close to edge
                    # Push toward board center (use cached center)
                    dx = polygon_center_x - x
                    dy = polygon_center_y - y
                    dist = math.sqrt(dx * dx + dy * dy) + 0.001
                    fx += self.config.boundary_strength * dx / dist * 2
                    fy += self.config.boundary_strength * dy / dist * 2

                # Check corners and edge midpoints
                for cx, cy in check_points:
                    if not outline.contains_point(cx, cy, margin=clearance):
                        # This point is outside - push component inward
                        # Use cached center for direction calculation
                        dx = polygon_center_x - cx
                        dy = polygon_center_y - cy
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

        # Cache bounding box and center for polygon outlines (avoids repeated calculation)
        polygon_center_x = 0.0
        polygon_center_y = 0.0
        if outline.polygon:
            polygon_bbox = outline.get_bounding_box()
            polygon_center_x = (polygon_bbox[0] + polygon_bbox[2]) / 2
            polygon_center_y = (polygon_bbox[1] + polygon_bbox[3]) / 2

        for ref, (x, y) in state.positions.items():
            # Get component dimensions (accounting for rotation)
            half_w, half_h = self._component_sizes.get(ref, (1.0, 1.0))

            if outline.polygon:
                # For polygon outlines, iteratively move toward centroid (use cached center)
                center_x = polygon_center_x
                center_y = polygon_center_y

                # Helper to generate check points (corners + edge midpoints for Issue #20)
                def get_check_points(cx, cy, hw, hh):
                    return [
                        # Corners
                        (cx - hw, cy - hh),
                        (cx + hw, cy - hh),
                        (cx - hw, cy + hh),
                        (cx + hw, cy + hh),
                        # Edge midpoints (catch concave intrusions)
                        (cx, cy - hh),
                        (cx, cy + hh),
                        (cx - hw, cy),
                        (cx + hw, cy),
                    ]

                check_points = get_check_points(x, y, half_w, half_h)

                # Iteratively move toward center until all check points are inside
                converged = False
                for _ in range(20):  # Max iterations
                    all_inside = True
                    for px, py in check_points:
                        if not outline.contains_point(px, py, margin=clearance):
                            all_inside = False
                            break

                    if all_inside:
                        converged = True
                        break

                    # Move 10% toward center
                    dx = center_x - x
                    dy = center_y - y
                    x += dx * 0.1
                    y += dy * 0.1

                    # Update check points
                    check_points = get_check_points(x, y, half_w, half_h)

                # Fallback: if iterations exhausted and still outside, place at center
                if not converged:
                    x, y = center_x, center_y
                    logger.debug(
                        "Component %s could not be clamped to polygon boundary, "
                        "placing at board center (%.1f, %.1f)", ref, x, y
                    )

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
        """Add forces for user-defined constraints.

        IMPORTANT: For constraints using the board-only interface (calculate_force),
        we sync state positions to board components first. This ensures constraints
        see current positions, not stale initial positions. See Issue #18.
        """
        # Check if any constraints use the board-only interface
        has_board_only_constraints = any(
            hasattr(c, 'calculate_force') and not hasattr(c, 'calculate_forces')
            for c in self.constraints
        )

        # Sync state positions to board for board-only constraints
        # Store original positions to restore after (avoids side effects)
        original_positions: Dict[str, Tuple[float, float]] = {}
        if has_board_only_constraints:
            for ref, (x, y) in state.positions.items():
                comp = self.board.components.get(ref)
                if comp:
                    original_positions[ref] = (comp.x, comp.y)
                    comp.x = x
                    comp.y = y

        try:
            for constraint in self.constraints:
                # Support both interfaces:
                # - calculate_forces(state, board, strength) - state-aware interface
                # - calculate_force(board, ref, strength) - board-only interface
                if hasattr(constraint, 'calculate_forces'):
                    constraint_forces = constraint.calculate_forces(
                        state, self.board, self.config.constraint_strength)
                    for ref, (fx, fy) in constraint_forces.items():
                        if ref in forces:
                            forces[ref].append(Force(fx, fy, ForceType.CONSTRAINT,
                                                     getattr(constraint, 'description', '')))
                elif hasattr(constraint, 'calculate_force'):
                    # Iterate over all components for constraints.py style
                    # Board positions are now synced from state
                    for ref in state.positions:
                        fx, fy = constraint.calculate_force(
                            self.board, ref, self.config.constraint_strength)
                        if fx != 0 or fy != 0:
                            forces[ref].append(Force(fx, fy, ForceType.CONSTRAINT,
                                                     getattr(constraint, 'description', '')))
        finally:
            # Restore original board positions to avoid side effects
            # The board is updated once at the end via _apply_to_board()
            for ref, (orig_x, orig_y) in original_positions.items():
                comp = self.board.components.get(ref)
                if comp:
                    comp.x = orig_x
                    comp.y = orig_y

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

    def _get_rotation_extents(self, ref: str, rotation: int
                              ) -> Tuple[float, float, float, float]:
        """Get component AABB extents relative to its centroid for a rotation."""
        rot_key = int(rotation) % 360
        cache_key = (ref, rot_key)
        if cache_key in self._rotation_extents_cache:
            return self._rotation_extents_cache[cache_key]

        comp = self.board.components.get(ref)
        if not comp:
            extents = (-0.5, -0.5, 0.5, 0.5)
            self._rotation_extents_cache[cache_key] = extents
            return extents

        hw = comp.width / 2
        hh = comp.height / 2
        rad = math.radians(rot_key)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)

        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for cx, cy in corners:
            rx = cx * cos_r - cy * sin_r
            ry = cx * sin_r + cy * cos_r
            min_x = min(min_x, rx)
            min_y = min(min_y, ry)
            max_x = max(max_x, rx)
            max_y = max(max_y, ry)

        for pad in comp.pads:
            pad_min_x, pad_min_y, pad_max_x, pad_max_y = pad.get_bounding_box(0.0, 0.0, rot_key)
            min_x = min(min_x, pad_min_x)
            min_y = min(min_y, pad_min_y)
            max_x = max(max_x, pad_max_x)
            max_y = max(max_y, pad_max_y)

        extents = (min_x, min_y, max_x, max_y)
        self._rotation_extents_cache[cache_key] = extents
        return extents

    def _rotation_penalty(self, ref: str, rotation: int,
                          state: PlacementState) -> float:
        """Estimate overlap/boundary penalty for a discrete rotation.

        Performance: Uses spatial grid for O(N) neighbor lookups (Issue #24).
        Only checks components in adjacent grid cells instead of all components.
        """
        comp = self.board.components.get(ref)
        if not comp or comp.dnp:
            return float("inf")

        x, y = state.positions[ref]
        min_dx, min_dy, max_dx, max_dy = self._get_rotation_extents(ref, rotation)
        half_w = max(abs(min_dx), abs(max_dx))
        half_h = max(abs(min_dy), abs(max_dy))

        penalty = 0.0
        outline = self.board.outline
        clearance = self.config.edge_clearance

        if outline.has_outline:
            if outline.polygon:
                corners = [
                    (x + min_dx, y + min_dy),
                    (x + max_dx, y + min_dy),
                    (x + min_dx, y + max_dy),
                    (x + max_dx, y + max_dy),
                ]
                for cx, cy in corners:
                    if not outline.contains_point(cx, cy, margin=clearance):
                        penalty += self.config.rotation_boundary_weight
            else:
                left = x + min_dx - clearance
                right = x + max_dx + clearance
                bottom = y + min_dy - clearance
                top = y + max_dy + clearance
                if left < outline.origin_x:
                    penalty += (outline.origin_x - left) * self.config.rotation_boundary_weight
                if right > outline.origin_x + outline.width:
                    penalty += (right - (outline.origin_x + outline.width)) * self.config.rotation_boundary_weight
                if bottom < outline.origin_y:
                    penalty += (outline.origin_y - bottom) * self.config.rotation_boundary_weight
                if top > outline.origin_y + outline.height:
                    penalty += (top - (outline.origin_y + outline.height)) * self.config.rotation_boundary_weight

        # Use spatial grid for O(N) neighbor lookup (Issue #24)
        # Only check components in adjacent cells instead of all N components
        for ref2 in self._repulsion_grid.get_all_neighbors(ref):
            comp2 = self.board.components.get(ref2)
            if comp2 and comp2.dnp:
                continue

            if not (comp.is_through_hole or (comp2 and comp2.is_through_hole)):
                from ..board.abstraction import Layer

                is_top1 = comp.layer == Layer.TOP_COPPER
                is_bottom1 = comp.layer == Layer.BOTTOM_COPPER
                is_top2 = comp2.layer == Layer.TOP_COPPER if comp2 else False
                is_bottom2 = comp2.layer == Layer.BOTTOM_COPPER if comp2 else False
                if (is_top1 and is_bottom2) or (is_bottom1 and is_top2):
                    continue

            x2, y2 = state.positions[ref2]
            dx = x - x2
            dy = y - y2

            pair_min_clearance, _ = self._get_pair_clearance(ref, ref2)
            half_w2, half_h2 = self._component_sizes.get(ref2, (1.0, 1.0))
            overlap_x = (half_w + half_w2 + pair_min_clearance) - abs(dx)
            overlap_y = (half_h + half_h2 + pair_min_clearance) - abs(dy)
            if overlap_x > 0 and overlap_y > 0:
                penalty += overlap_x * overlap_y * self.config.rotation_overlap_weight

        return penalty

    def _apply_discrete_rotation_search(self, state: PlacementState, iteration: int):
        """Try discrete rotations to reduce overlap/boundary penalties.

        Performance: Uses spatial grid for O(N) neighbor lookups (Issue #24).
        Rebuilds grid with current positions before searching.
        """
        if not self.config.enable_discrete_rotation:
            return
        if self.config.rotation_search_interval <= 0:
            return
        if iteration % self.config.rotation_search_interval != 0:
            return
        if not self.config.rotation_angles:
            return

        # Rebuild spatial grid with current positions (Issue #24)
        # Positions may have changed since last force calculation
        self._repulsion_grid.clear()
        for ref, (x, y) in state.positions.items():
            comp = self.board.components.get(ref)
            if comp and not comp.dnp:
                self._repulsion_grid.insert(ref, x, y)

        for ref in state.positions:
            if self._is_component_locked(ref):
                continue
            comp = self.board.components.get(ref)
            if not comp or comp.dnp:
                continue
            if ref in self._rotation_constraint_refs:
                continue

            current_rot = int(state.rotations.get(ref, comp.rotation)) % 360
            best_rot = current_rot
            best_penalty = self._rotation_penalty(ref, current_rot, state)

            for candidate in self.config.rotation_angles:
                candidate_rot = int(candidate) % 360
                if candidate_rot == current_rot:
                    continue
                penalty = self._rotation_penalty(ref, candidate_rot, state)
                if penalty + 1e-6 < best_penalty:
                    best_penalty = penalty
                    best_rot = candidate_rot

            if best_rot != current_rot:
                state.rotations[ref] = best_rot
                min_dx, min_dy, max_dx, max_dy = self._get_rotation_extents(ref, best_rot)
                half_w = max(abs(min_dx), abs(max_dx), 0.1)
                half_h = max(abs(min_dy), abs(max_dy), 0.1)
                self._component_sizes[ref] = (half_w, half_h)

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


# Note: PlacementConstraint, ProximityConstraint, EdgeConstraint, and ZoneConstraint
# are defined in atoplace/placement/constraints.py - import from there.
# The ForceDirectedRefiner._add_constraint_forces() method supports both interfaces:
# - calculate_forces(state, board, strength) - batch interface
# - calculate_force(board, ref, strength) - per-component interface (used by constraints.py)
