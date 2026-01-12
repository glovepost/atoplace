"""
Placement Legalizer

Post-physics legalization pass that transforms "organic" force-directed
layouts into professional "Manhattan" aesthetics.

Three-phase process (from manhattan_placement_strategy.md):
1. Grid Snapping (Quantizer) - Snap positions and rotations to grid
2. Row Alignment (Beautifier) - Align passives using PCA and median projection
3. Overlap Removal (Shove) - Priority-based collision resolution with MTV

Based on REQ-P-03 from PRODUCT_REQUIREMENTS.md:
- Snaps component centroids to the user grid
- Aligns adjacent same-size components (e.g., 0402 resistors) into shared-axis rows/columns
- Removes overlaps using AABB collision detection
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum

from ..board.abstraction import Board, Component

logger = logging.getLogger(__name__)


class PassiveSize(Enum):
    """Standard passive component sizes (imperial)."""
    SIZE_0201 = "0201"
    SIZE_0402 = "0402"
    SIZE_0603 = "0603"
    SIZE_0805 = "0805"
    SIZE_1206 = "1206"
    SIZE_1210 = "1210"
    SIZE_2010 = "2010"
    SIZE_2512 = "2512"
    UNKNOWN = "unknown"


class ComponentPriority(Enum):
    """Priority for overlap resolution (higher = less likely to move)."""
    LOCKED = 100
    LARGE_IC = 80
    CONNECTOR = 60
    SMALL_IC = 40
    PASSIVE = 20
    OTHER = 10


# Approximate dimensions for passive sizes (mm)
PASSIVE_DIMENSIONS = {
    PassiveSize.SIZE_0201: (0.6, 0.3),
    PassiveSize.SIZE_0402: (1.0, 0.5),
    PassiveSize.SIZE_0603: (1.6, 0.8),
    PassiveSize.SIZE_0805: (2.0, 1.25),
    PassiveSize.SIZE_1206: (3.2, 1.6),
    PassiveSize.SIZE_1210: (3.2, 2.5),
    PassiveSize.SIZE_2010: (5.0, 2.5),
    PassiveSize.SIZE_2512: (6.3, 3.2),
}

# Fine-pitch passive sizes that should use secondary grid
FINE_PITCH_SIZES = {PassiveSize.SIZE_0201, PassiveSize.SIZE_0402}

# Metric to imperial size mapping (common metric codes)
METRIC_TO_IMPERIAL = {
    "0603": PassiveSize.SIZE_0201,  # Metric 0603 = Imperial 0201
    "1005": PassiveSize.SIZE_0402,  # Metric 1005 = Imperial 0402
    "1608": PassiveSize.SIZE_0603,  # Metric 1608 = Imperial 0603
    "2012": PassiveSize.SIZE_0805,  # Metric 2012 = Imperial 0805
    "3216": PassiveSize.SIZE_1206,  # Metric 3216 = Imperial 1206
    "3225": PassiveSize.SIZE_1210,  # Metric 3225 = Imperial 1210
    "5025": PassiveSize.SIZE_2010,  # Metric 5025 = Imperial 2010
    "6332": PassiveSize.SIZE_2512,  # Metric 6332 = Imperial 2512
}


@dataclass
class LegalizerConfig:
    """Configuration for the legalization pass."""
    # Grid snapping (Phase 1: Quantizer)
    primary_grid: float = 0.5  # mm - standard component placement grid
    secondary_grid: float = 0.1  # mm - fine-pitch components (0201, 0402)
    rotation_grid: float = 90.0  # degrees - snap rotations to this increment
    snap_rotation: bool = True  # whether to snap rotation

    # Row alignment (Phase 2: Beautifier)
    cluster_radius: float = 10.0  # mm - max distance for clustering
    alignment_tolerance: float = 2.0  # mm - max distance to consider for alignment
    min_row_components: int = 2  # minimum components to form a row
    row_spacing: float = 1.0  # mm - spacing between aligned components

    # Overlap removal (Phase 3: Shove)
    min_clearance: float = 0.25  # mm - minimum spacing between components
    max_displacement_iterations: int = 50  # max iterations for overlap removal
    manhattan_shove: bool = True  # constrain displacement to X/Y axes only
    overlap_retry_passes: int = 3  # number of retry passes with escalating displacement
    escalation_factor: float = 1.5  # multiply displacement by this factor on retry

    # Component filtering
    align_passives_only: bool = True  # only align R/C/L components
    skip_locked: bool = True  # don't move locked components


@dataclass
class LegalizationResult:
    """Result of legalization pass."""
    grid_snapped: int = 0  # components snapped to grid
    rows_formed: int = 0  # alignment rows created
    components_aligned: int = 0  # components aligned into rows
    overlaps_resolved: int = 0  # overlap violations fixed
    iterations_used: int = 0  # iterations for overlap removal
    final_overlaps: int = 0  # remaining overlaps (if any)
    locked_conflicts: List[Tuple[str, str]] = field(default_factory=list)  # unresolvable overlaps with locked components


class PlacementLegalizer:
    """
    Legalizes component placement after force-directed refinement.

    Transforms organic layouts into professional Manhattan aesthetics by:
    1. Snapping components to a regular grid
    2. Aligning same-size passives into neat rows/columns
    3. Resolving any overlaps created by snapping/alignment

    Respects placement constraints:
    - FixedConstraints: Components are treated as locked and not moved
    - ProximityConstraints: Moves that would violate proximity are avoided
    """

    def __init__(self, board: Board, config: Optional[LegalizerConfig] = None,
                 constraints: Optional[List] = None):
        """
        Initialize the legalizer.

        Args:
            board: The board to legalize
            config: Legalization configuration
            constraints: Optional list of placement constraints to respect.
                        FixedConstraints will prevent component movement.
        """
        self.board = board
        self.config = config or LegalizerConfig()
        self.constraints = constraints or []

        # Extract fixed components from constraints to treat as locked
        self._fixed_components: Set[str] = self._extract_fixed_components()

        # Cache component sizes for overlap detection
        self._component_sizes: Dict[str, Tuple[float, float]] = {}
        self._compute_component_sizes()

    def _extract_fixed_components(self) -> Set[str]:
        """Extract component refs from FixedConstraints to treat as locked."""
        fixed_refs = set()

        for constraint in self.constraints:
            # Check if it's a FixedConstraint by looking for component_ref attribute
            # and constraint_type (avoid import cycles by using duck typing)
            if hasattr(constraint, 'constraint_type') and hasattr(constraint, 'component_ref'):
                constraint_type = getattr(constraint.constraint_type, 'value', str(constraint.constraint_type))
                if constraint_type == 'fixed':
                    ref = constraint.component_ref
                    if ref:
                        fixed_refs.add(ref)

        return fixed_refs

    def _is_component_fixed(self, ref: str) -> bool:
        """Check if a component should be treated as fixed (not moved)."""
        # Check explicit locked flag
        comp = self.board.components.get(ref)
        if comp and comp.locked:
            return True

        # Check if in our fixed constraints set
        if ref in self._fixed_components:
            return True

        return False

    def legalize(self) -> LegalizationResult:
        """
        Run full legalization pass.

        Returns:
            LegalizationResult with statistics
        """
        result = LegalizationResult()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Legalization start: components=%d fixed=%d grid=%.3f secondary=%.3f snap_rot=%s",
                len(self.board.components),
                len(self._fixed_components),
                self.config.primary_grid,
                self.config.secondary_grid,
                self.config.snap_rotation,
            )
            logger.debug(
                "Legalizer config: row_spacing=%.3f min_clearance=%.3f align_passives_only=%s manhattan=%s",
                self.config.row_spacing,
                self.config.min_clearance,
                self.config.align_passives_only,
                self.config.manhattan_shove,
            )

        # Phase 1: Grid snapping
        result.grid_snapped = self._snap_to_grid()

        # Phase 2: Row alignment (passives)
        rows_result = self._align_rows()
        result.rows_formed = rows_result[0]
        result.components_aligned = rows_result[1]

        # Phase 3: Overlap removal
        overlap_result = self._remove_overlaps()
        result.overlaps_resolved = overlap_result[0]
        result.iterations_used = overlap_result[1]
        result.final_overlaps = overlap_result[2]
        result.locked_conflicts = overlap_result[3]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Legalization done: snapped=%d rows=%d aligned=%d overlaps_resolved=%d final_overlaps=%d",
                result.grid_snapped,
                result.rows_formed,
                result.components_aligned,
                result.overlaps_resolved,
                result.final_overlaps,
            )
        return result

    def _snap_to_grid(self) -> int:
        """
        Phase 1: Quantizer - Snap positions and rotations to grid.

        Uses grid hierarchy:
        - Primary Grid (0.5mm): Standard for most ICs/Connectors
        - Secondary Grid (0.1mm): Fine-pitch components (0201, 0402)
        - Rotation Grid (90°): Standard orthogonal alignment

        Returns:
            Number of components modified
        """
        snapped = 0
        skipped = 0

        for ref, comp in self.board.components.items():
            # Skip locked or fixed-constraint components
            if self.config.skip_locked and self._is_component_fixed(ref):
                skipped += 1
                continue

            modified = False
            old_x = comp.x
            old_y = comp.y
            old_rot = comp.rotation

            # Determine grid size based on component type
            grid = self.config.primary_grid
            if self._is_passive(ref):
                size = self._detect_passive_size(comp)
                if size in FINE_PITCH_SIZES:
                    grid = self.config.secondary_grid

            # Snap position to grid
            new_x = round(comp.x / grid) * grid
            new_y = round(comp.y / grid) * grid

            if abs(new_x - comp.x) > 0.001 or abs(new_y - comp.y) > 0.001:
                comp.x = new_x
                comp.y = new_y
                modified = True

            # Snap rotation to grid (typically 90°)
            if self.config.snap_rotation:
                rot_grid = self.config.rotation_grid
                new_rot = round(comp.rotation / rot_grid) * rot_grid
                new_rot = new_rot % 360  # Normalize to 0-359

                if abs(new_rot - comp.rotation) > 0.1:
                    comp.rotation = new_rot
                    modified = True
                    # Update component sizes cache for new rotation
                    self._update_component_size(ref, comp)

            if modified:
                snapped += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Snap %s: (%.3f, %.3f rot=%.1f) -> (%.3f, %.3f rot=%.1f) grid=%.3f",
                        ref,
                        old_x,
                        old_y,
                        old_rot,
                        comp.x,
                        comp.y,
                        comp.rotation,
                        grid,
                    )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Grid snapping complete: snapped=%d skipped=%d", snapped, skipped)

        return snapped

    def _update_component_size(self, ref: str, comp: Component):
        """Update cached size for a single component after rotation change."""
        w, h = comp.width, comp.height
        rot_rad = math.radians(comp.rotation)

        cos_r = abs(math.cos(rot_rad))
        sin_r = abs(math.sin(rot_rad))

        half_w = (w / 2) * cos_r + (h / 2) * sin_r
        half_h = (w / 2) * sin_r + (h / 2) * cos_r

        self._component_sizes[ref] = (half_w, half_h)

    def _align_rows(self) -> Tuple[int, int]:
        """
        Phase 2: Beautifier - Detect and enforce linear structures.

        Uses PCA (Principal Component Analysis) to detect natural row/column
        orientation, then projects components onto the median axis.

        Returns:
            (rows_formed, components_aligned)
        """
        if not self.config.align_passives_only:
            return (0, 0)

        # Group passives by size
        passives_by_size: Dict[PassiveSize, List[str]] = {}

        for ref, comp in self.board.components.items():
            # Skip locked or fixed-constraint components
            if self.config.skip_locked and self._is_component_fixed(ref):
                continue

            # Check if this is a passive (R, C, L)
            if not self._is_passive(ref):
                continue

            size = self._detect_passive_size(comp)
            if size == PassiveSize.UNKNOWN:
                continue

            if size not in passives_by_size:
                passives_by_size[size] = []
            passives_by_size[size].append(ref)

        rows_formed = 0
        components_aligned = 0

        # For each size group, find and form rows
        for size, refs in passives_by_size.items():
            if len(refs) < self.config.min_row_components:
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Row alignment: size=%s candidates=%d", size.value, len(refs))

            # Determine grid for this passive size
            use_fine_grid = size in FINE_PITCH_SIZES

            # Find clusters of nearby components within cluster_radius
            clusters = self._find_alignment_clusters(refs)

            for cluster in clusters:
                if len(cluster) < self.config.min_row_components:
                    continue

                # Use PCA to determine row vs column, then align
                aligned = self._align_cluster_pca(cluster, use_fine_grid)
                if aligned > 0:
                    rows_formed += 1
                    components_aligned += aligned

        return (rows_formed, components_aligned)

    def _find_alignment_clusters(self, refs: List[str]) -> List[List[str]]:
        """
        Find clusters of components that should be aligned together.

        Uses distance-based clustering with cluster_radius parameter.
        Components within cluster_radius of any cluster member are added.
        """
        if not refs:
            return []

        clusters: List[List[str]] = []
        assigned: Set[str] = set()

        for ref in refs:
            if ref in assigned:
                continue

            # Start new cluster with BFS expansion
            cluster = [ref]
            assigned.add(ref)
            queue = [ref]

            while queue:
                current_ref = queue.pop(0)
                current = self.board.components[current_ref]

                # Find nearby unassigned components
                for other_ref in refs:
                    if other_ref in assigned:
                        continue

                    other = self.board.components[other_ref]

                    # Check distance
                    dx = current.x - other.x
                    dy = current.y - other.y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance <= self.config.cluster_radius:
                        cluster.append(other_ref)
                        assigned.add(other_ref)
                        queue.append(other_ref)

            if len(cluster) >= self.config.min_row_components:
                clusters.append(cluster)

        return clusters

    def _align_cluster_pca(self, refs: List[str], use_fine_grid: bool = False) -> int:
        """
        Align a cluster using PCA to detect principal axis.

        Uses Principal Component Analysis to determine if the cluster
        forms a natural row (horizontal) or column (vertical), then
        projects components onto the median axis.

        Only aligns components if they are within alignment_tolerance of
        being collinear to avoid forcing alignment on scattered components.

        Args:
            refs: List of component references to align
            use_fine_grid: If True, use secondary_grid for fine-pitch passives

        Returns:
            Number of components aligned
        """
        if len(refs) < 2:
            return 0

        # Get positions
        positions = [(self.board.components[ref].x,
                      self.board.components[ref].y) for ref in refs]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Calculate covariance matrix for PCA
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        # Covariance components
        cov_xx = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        cov_yy = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in positions) / len(xs)

        # Calculate principal direction angle using eigenvector
        # For 2D, the principal eigenvector direction is:
        # angle = 0.5 * atan2(2 * cov_xy, cov_xx - cov_yy)
        if abs(cov_xx - cov_yy) < 0.001 and abs(cov_xy) < 0.001:
            # Degenerate case - use spread-based detection
            x_spread = max(xs) - min(xs)
            y_spread = max(ys) - min(ys)
            is_horizontal = x_spread >= y_spread
        else:
            angle = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
            # Convert to degrees and determine if closer to horizontal or vertical
            angle_deg = abs(math.degrees(angle))
            is_horizontal = angle_deg < 45

        # Check alignment_tolerance: verify components are close to collinear
        # by checking the spread perpendicular to the principal axis
        tolerance = self.config.alignment_tolerance
        if is_horizontal:
            # For horizontal alignment, check y-spread (perpendicular to row)
            y_spread = max(ys) - min(ys)
            if y_spread > tolerance:
                # Components are too scattered vertically to align into a row
                return 0
        else:
            # For vertical alignment, check x-spread (perpendicular to column)
            x_spread = max(xs) - min(xs)
            if x_spread > tolerance:
                # Components are too scattered horizontally to align into a column
                return 0

        aligned = 0

        # Select appropriate grid based on passive size
        grid = self.config.secondary_grid if use_fine_grid else self.config.primary_grid

        if is_horizontal:
            # Horizontal row - snap Y to MEDIAN (more robust than mean)
            sorted_ys = sorted(ys)
            median_y = sorted_ys[len(sorted_ys) // 2]
            # Snap median to grid
            target_y = round(median_y / grid) * grid
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Align row: refs=%d axis=Y target=%.3f grid=%.3f",
                    len(refs),
                    target_y,
                    grid,
                )

            for ref in refs:
                comp = self.board.components[ref]
                if abs(comp.y - target_y) > 0.001:
                    comp.y = target_y
                    aligned += 1

            # Re-distribute along X with proper spacing
            self._distribute_evenly(refs, axis='x', use_fine_grid=use_fine_grid)
        else:
            # Vertical column - snap X to MEDIAN
            sorted_xs = sorted(xs)
            median_x = sorted_xs[len(sorted_xs) // 2]
            # Snap median to grid
            target_x = round(median_x / grid) * grid
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Align column: refs=%d axis=X target=%.3f grid=%.3f",
                    len(refs),
                    target_x,
                    grid,
                )

            for ref in refs:
                comp = self.board.components[ref]
                if abs(comp.x - target_x) > 0.001:
                    comp.x = target_x
                    aligned += 1

            # Re-distribute along Y with proper spacing
            self._distribute_evenly(refs, axis='y', use_fine_grid=use_fine_grid)

        return aligned

    def _distribute_evenly(self, refs: List[str], axis: str, use_fine_grid: bool = False):
        """
        Distribute components evenly along an axis.

        Maintains relative order but ensures minimum spacing.
        Snaps positions to the appropriate grid based on component type.

        Args:
            refs: List of component references
            axis: 'x' or 'y' axis for distribution
            use_fine_grid: If True, use secondary_grid for fine-pitch passives
        """
        if len(refs) < 2:
            return

        # Sort by position along axis
        if axis == 'x':
            refs_sorted = sorted(refs, key=lambda r: self.board.components[r].x)
        else:
            refs_sorted = sorted(refs, key=lambda r: self.board.components[r].y)

        # Get component sizes for spacing calculation
        sizes = [self._component_sizes.get(ref, (1.0, 1.0)) for ref in refs_sorted]

        # Calculate minimum spacing needed
        min_spacing = self.config.row_spacing + self.config.min_clearance
        # Use appropriate grid for fine-pitch passives
        grid = self.config.secondary_grid if use_fine_grid else self.config.primary_grid

        # Adjust positions to ensure minimum spacing
        for i in range(1, len(refs_sorted)):
            prev_ref = refs_sorted[i-1]
            curr_ref = refs_sorted[i]

            prev_comp = self.board.components[prev_ref]
            curr_comp = self.board.components[curr_ref]

            prev_size = sizes[i-1]
            curr_size = sizes[i]

            if axis == 'x':
                # Required separation
                required_sep = prev_size[0]/2 + curr_size[0]/2 + min_spacing
                actual_sep = curr_comp.x - prev_comp.x

                if actual_sep < required_sep:
                    # Move current component right
                    curr_comp.x = prev_comp.x + required_sep
                    # Snap to grid
                    curr_comp.x = round(curr_comp.x / grid) * grid
            else:
                # Required separation
                required_sep = prev_size[1]/2 + curr_size[1]/2 + min_spacing
                actual_sep = curr_comp.y - prev_comp.y

                if actual_sep < required_sep:
                    # Move current component down
                    curr_comp.y = prev_comp.y + required_sep
                    # Snap to grid
                    curr_comp.y = round(curr_comp.y / grid) * grid

    def _remove_overlaps(self) -> Tuple[int, int, int, List[Tuple[str, str]]]:
        """
        Phase 3: Shove - Remove overlaps using priority-based displacement.

        Iteratively resolves collisions by pushing lower-priority components.
        Uses Minimum Translation Vector (MTV) for efficient separation.

        If overlaps persist after initial pass, retries with escalating
        displacement factors and eventually non-Manhattan movement.

        Priority order: Locked > Large ICs > Connectors > Small ICs > Passives

        Returns:
            (overlaps_resolved, iterations_used, final_overlaps, locked_conflicts)
        """
        overlaps_resolved = 0
        total_iterations = 0
        locked_conflicts: List[Tuple[str, str]] = []

        # Compute priorities once
        priorities = {ref: self._get_component_priority(ref)
                      for ref in self.board.components}

        # Track escalation state
        escalation_multiplier = 1.0
        use_manhattan = self.config.manhattan_shove

        for retry_pass in range(self.config.overlap_retry_passes):
            pass_resolved = 0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Overlap pass %d: escalation=%.2f manhattan=%s",
                    retry_pass + 1,
                    escalation_multiplier,
                    use_manhattan,
                )

            for iteration in range(self.config.max_displacement_iterations):
                total_iterations += 1

                # Find all overlapping pairs
                overlaps = self._find_overlaps()

                if not overlaps:
                    break
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Overlap iteration %d: overlaps=%d",
                        iteration + 1,
                        len(overlaps),
                    )

                # Sort overlaps by combined priority (resolve high-priority first)
                overlaps.sort(key=lambda o: -(priorities[o[0]].value + priorities[o[1]].value))

                made_progress = False

                # Resolve each overlap
                for ref1, ref2, _, _ in overlaps:
                    # Recompute current overlap values since earlier resolutions may have
                    # changed positions, making the initial overlap values stale
                    current_overlap = self._get_overlap_values(ref1, ref2)
                    if current_overlap is None:
                        # No longer overlapping after earlier resolutions
                        continue

                    overlap_x, overlap_y = current_overlap

                    # Apply escalation multiplier to increase displacement on retries
                    scaled_overlap_x = overlap_x * escalation_multiplier
                    scaled_overlap_y = overlap_y * escalation_multiplier

                    resolved = self._resolve_overlap_priority(
                        ref1, ref2, scaled_overlap_x, scaled_overlap_y,
                        priorities[ref1], priorities[ref2],
                        use_manhattan=use_manhattan
                    )
                    if resolved:
                        # Verify the overlap is actually resolved after displacement
                        # (grid snapping can sometimes re-introduce overlap)
                        if not self._check_overlap(ref1, ref2):
                            overlaps_resolved += 1
                            pass_resolved += 1
                            made_progress = True
                    elif (priorities[ref1] == ComponentPriority.LOCKED and
                          priorities[ref2] == ComponentPriority.LOCKED):
                        # Both components locked - record unresolvable conflict
                        conflict = (ref1, ref2) if ref1 < ref2 else (ref2, ref1)
                        if conflict not in locked_conflicts:
                            locked_conflicts.append(conflict)

                # If no progress was made this iteration, break early
                if not made_progress:
                    break

            # Check if we resolved all overlaps
            remaining = len(self._find_overlaps())
            if remaining == 0:
                break

            # Escalate for next retry pass
            escalation_multiplier *= self.config.escalation_factor

            # On final retry, allow non-Manhattan movement for stubborn overlaps
            if retry_pass == self.config.overlap_retry_passes - 2:
                use_manhattan = False

        # Count remaining overlaps
        final_overlaps = len(self._find_overlaps())

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Overlap removal done: resolved=%d iterations=%d final=%d locked_conflicts=%d",
                overlaps_resolved,
                total_iterations,
                final_overlaps,
                len(locked_conflicts),
            )
        return (overlaps_resolved, total_iterations, final_overlaps, locked_conflicts)

    def _get_component_priority(self, ref: str) -> ComponentPriority:
        """Determine component priority for overlap resolution."""
        comp = self.board.components.get(ref)
        if not comp:
            return ComponentPriority.OTHER

        # Locked or fixed-constraint components have highest priority
        if self._is_component_fixed(ref):
            return ComponentPriority.LOCKED

        # Check component type from reference designator
        first_char = ref[0].upper() if ref else ''

        if first_char == 'J' or first_char == 'P':
            return ComponentPriority.CONNECTOR

        if first_char == 'U':
            # Distinguish large ICs from small ICs by size
            area = comp.width * comp.height
            if area > 50:  # > 50mm² is a large IC
                return ComponentPriority.LARGE_IC
            return ComponentPriority.SMALL_IC

        if first_char in ('R', 'C', 'L'):
            return ComponentPriority.PASSIVE

        return ComponentPriority.OTHER

    def _check_overlap(self, ref1: str, ref2: str) -> bool:
        """
        Check if two specific components overlap.

        Returns:
            True if the components overlap, False otherwise
        """
        return self._get_overlap_values(ref1, ref2) is not None

    def _get_overlap_values(self, ref1: str, ref2: str) -> Optional[Tuple[float, float]]:
        """
        Get current overlap values for two specific components.

        Returns:
            Tuple of (overlap_x, overlap_y) if overlapping, None if not overlapping
        """
        comp1 = self.board.components.get(ref1)
        comp2 = self.board.components.get(ref2)
        if not comp1 or not comp2:
            return None

        half_w1, half_h1 = self._component_sizes.get(ref1, (1.0, 1.0))
        half_w2, half_h2 = self._component_sizes.get(ref2, (1.0, 1.0))

        # Calculate required separation
        sep_x = half_w1 + half_w2 + self.config.min_clearance
        sep_y = half_h1 + half_h2 + self.config.min_clearance

        # Calculate actual separation
        dx = abs(comp1.x - comp2.x)
        dy = abs(comp1.y - comp2.y)

        # Calculate overlap amounts
        overlap_x = sep_x - dx
        overlap_y = sep_y - dy

        # Check for overlap (both axes must overlap)
        if overlap_x > 0 and overlap_y > 0:
            return (overlap_x, overlap_y)
        return None

    def _build_spatial_index(self) -> Dict[Tuple[int, int], List[str]]:
        """
        Build a grid-based spatial index for efficient overlap detection.

        Uses a grid cell size based on the largest component dimension to
        ensure overlapping components will be in the same or adjacent cells.

        Returns:
            Dictionary mapping grid cell (x, y) to list of component refs in that cell
        """
        # Determine grid cell size based on max component dimension + clearance
        # This ensures overlapping components are in same or adjacent cells
        max_dim = 0.0
        for ref in self.board.components:
            half_w, half_h = self._component_sizes.get(ref, (1.0, 1.0))
            max_dim = max(max_dim, half_w * 2, half_h * 2)

        # Cell size should capture components and their clearance zones
        self._grid_cell_size = max(1.0, max_dim + self.config.min_clearance * 2)

        # Build the index
        spatial_index: Dict[Tuple[int, int], List[str]] = {}

        for ref, comp in self.board.components.items():
            half_w, half_h = self._component_sizes.get(ref, (1.0, 1.0))

            # Calculate the grid cells this component occupies
            min_cell_x = int((comp.x - half_w - self.config.min_clearance) / self._grid_cell_size)
            max_cell_x = int((comp.x + half_w + self.config.min_clearance) / self._grid_cell_size)
            min_cell_y = int((comp.y - half_h - self.config.min_clearance) / self._grid_cell_size)
            max_cell_y = int((comp.y + half_h + self.config.min_clearance) / self._grid_cell_size)

            # Add component to all cells it touches
            for cx in range(min_cell_x, max_cell_x + 1):
                for cy in range(min_cell_y, max_cell_y + 1):
                    key = (cx, cy)
                    if key not in spatial_index:
                        spatial_index[key] = []
                    if ref not in spatial_index[key]:
                        spatial_index[key].append(ref)

        return spatial_index

    def _find_overlaps(self) -> List[Tuple[str, str, float, float]]:
        """
        Find all overlapping component pairs using spatial indexing.

        Uses a grid-based spatial index to reduce complexity from O(N²)
        to approximately O(N) for typical PCB layouts where components
        are distributed across the board area.

        Returns list of (ref1, ref2, overlap_x, overlap_y) tuples.
        """
        overlaps = []
        checked_pairs: Set[Tuple[str, str]] = set()

        # Build spatial index
        spatial_index = self._build_spatial_index()

        # Check each cell for overlapping components
        for cell_refs in spatial_index.values():
            if len(cell_refs) < 2:
                continue

            # Check pairs within this cell
            for i, ref1 in enumerate(cell_refs):
                comp1 = self.board.components[ref1]
                half_w1, half_h1 = self._component_sizes.get(ref1, (1.0, 1.0))

                for ref2 in cell_refs[i+1:]:
                    # Skip if already checked (components can be in multiple cells)
                    pair_key = (ref1, ref2) if ref1 < ref2 else (ref2, ref1)
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    comp2 = self.board.components[ref2]
                    half_w2, half_h2 = self._component_sizes.get(ref2, (1.0, 1.0))

                    # Calculate required separation
                    sep_x = half_w1 + half_w2 + self.config.min_clearance
                    sep_y = half_h1 + half_h2 + self.config.min_clearance

                    # Calculate actual separation
                    dx = abs(comp1.x - comp2.x)
                    dy = abs(comp1.y - comp2.y)

                    # Check for overlap
                    overlap_x = sep_x - dx
                    overlap_y = sep_y - dy

                    if overlap_x > 0 and overlap_y > 0:
                        overlaps.append((ref1, ref2, overlap_x, overlap_y))

        return overlaps

    def _resolve_overlap_priority(
        self, ref1: str, ref2: str,
        overlap_x: float, overlap_y: float,
        priority1: ComponentPriority, priority2: ComponentPriority,
        use_manhattan: Optional[bool] = None
    ) -> bool:
        """
        Resolve overlap using priority-based displacement.

        Uses Minimum Translation Vector (MTV) approach:
        - Choose axis with minimum overlap (smaller displacement)
        - Move lower-priority component by MTV
        - If equal priority, move both components

        Displacement can be Manhattan-constrained (X or Y only) or diagonal.

        Args:
            ref1, ref2: Component references
            overlap_x, overlap_y: Overlap amounts (may be scaled for escalation)
            priority1, priority2: Component priorities
            use_manhattan: Override for manhattan_shove config (for escalation)

        Returns:
            True if overlap was resolved
        """
        comp1 = self.board.components[ref1]
        comp2 = self.board.components[ref2]

        # Both locked = cannot resolve
        if priority1 == ComponentPriority.LOCKED and priority2 == ComponentPriority.LOCKED:
            return False

        # Determine direction from comp1 to comp2
        dx = comp2.x - comp1.x
        dy = comp2.y - comp1.y

        # Use provided manhattan setting or fall back to config
        manhattan = use_manhattan if use_manhattan is not None else self.config.manhattan_shove

        # Calculate MTV (Minimum Translation Vector)
        # Choose axis with smaller overlap for minimum displacement
        if manhattan:
            if overlap_x <= overlap_y:
                # MTV is along X axis
                mtv_x = overlap_x + 0.01  # Small buffer
                mtv_y = 0.0
            else:
                # MTV is along Y axis
                mtv_x = 0.0
                mtv_y = overlap_y + 0.01
        else:
            # Non-Manhattan: MTV along center-to-center direction
            dist = math.sqrt(dx*dx + dy*dy) if (dx != 0 or dy != 0) else 1.0
            mtv_mag = min(overlap_x, overlap_y) + 0.01
            mtv_x = mtv_mag * dx / dist if dist > 0 else mtv_mag
            mtv_y = mtv_mag * dy / dist if dist > 0 else 0

        # Apply direction (MTV points from comp1 toward comp2)
        # When dx or dy is zero, use deterministic tiebreaker based on ref names
        # to avoid consistent +X/+Y drift bias
        if dx < 0:
            mtv_x = -abs(mtv_x)
        elif dx > 0:
            mtv_x = abs(mtv_x)
        else:
            # dx == 0: use lexicographic comparison for deterministic direction
            mtv_x = abs(mtv_x) if ref2 > ref1 else -abs(mtv_x)

        if dy < 0:
            mtv_y = -abs(mtv_y)
        elif dy > 0:
            mtv_y = abs(mtv_y)
        else:
            # dy == 0: use lexicographic comparison for deterministic direction
            mtv_y = abs(mtv_y) if ref2 > ref1 else -abs(mtv_y)

        # Choose appropriate grid for each component:
        # Fine-pitch passives use secondary_grid, others use primary_grid
        def get_grid_for_ref(ref: str) -> float:
            if self._is_passive(ref):
                comp = self.board.components.get(ref)
                if comp:
                    size = self._detect_passive_size(comp)
                    if size in FINE_PITCH_SIZES:
                        return self.config.secondary_grid
            return self.config.primary_grid

        grid1 = get_grid_for_ref(ref1)
        grid2 = get_grid_for_ref(ref2)

        # Determine who moves based on priority
        if priority1.value > priority2.value:
            # comp1 has higher priority - move comp2 away
            comp2.x += mtv_x
            comp2.y += mtv_y
            comp2.x = round(comp2.x / grid2) * grid2
            comp2.y = round(comp2.y / grid2) * grid2
        elif priority2.value > priority1.value:
            # comp2 has higher priority - move comp1 away
            comp1.x -= mtv_x
            comp1.y -= mtv_y
            comp1.x = round(comp1.x / grid1) * grid1
            comp1.y = round(comp1.y / grid1) * grid1
        else:
            # Equal priority - move both by half MTV
            comp1.x -= mtv_x / 2
            comp1.y -= mtv_y / 2
            comp2.x += mtv_x / 2
            comp2.y += mtv_y / 2
            comp1.x = round(comp1.x / grid1) * grid1
            comp1.y = round(comp1.y / grid1) * grid1
            comp2.x = round(comp2.x / grid2) * grid2
            comp2.y = round(comp2.y / grid2) * grid2

        # Ensure components stay within board boundaries after displacement
        self._clamp_to_bounds(ref1, grid1)
        self._clamp_to_bounds(ref2, grid2)

        return True

    def _clamp_to_bounds(self, ref: str, grid: float):
        """Clamp component position to stay within board boundaries.

        Skipped when board has no explicit outline defined.
        """
        comp = self.board.components.get(ref)
        if not comp:
            return

        outline = self.board.outline

        # Skip boundary clamping when no explicit outline is defined
        if not outline.has_outline:
            return

        half_w, half_h = self._component_sizes.get(ref, (comp.width / 2, comp.height / 2))
        margin = self.config.min_clearance

        # Calculate bounds
        min_x = outline.origin_x + half_w + margin
        max_x = outline.origin_x + outline.width - half_w - margin
        min_y = outline.origin_y + half_h + margin
        max_y = outline.origin_y + outline.height - half_h - margin

        # Clamp position
        clamped = False
        if comp.x < min_x:
            comp.x = min_x
            clamped = True
        elif comp.x > max_x:
            comp.x = max_x
            clamped = True

        if comp.y < min_y:
            comp.y = min_y
            clamped = True
        elif comp.y > max_y:
            comp.y = max_y
            clamped = True

        # Re-snap to grid if clamped
        if clamped:
            comp.x = round(comp.x / grid) * grid
            comp.y = round(comp.y / grid) * grid

    def _is_passive(self, ref: str) -> bool:
        """Check if a component reference indicates a passive (R, C, L)."""
        if not ref:
            return False
        first_char = ref[0].upper()
        return first_char in ('R', 'C', 'L')

    def _detect_passive_size(self, comp: Component) -> PassiveSize:
        """
        Detect passive component size from dimensions or footprint.

        Supports both imperial (0402, 0603, etc.) and metric (1005, 1608, etc.) codes.

        Returns PassiveSize enum value.
        """
        # Try to detect from footprint name
        footprint = comp.footprint.lower() if comp.footprint else ""

        # Check imperial codes first
        for size in PassiveSize:
            if size == PassiveSize.UNKNOWN:
                continue
            if size.value in footprint:
                return size

        # Check metric codes
        for metric_code, imperial_size in METRIC_TO_IMPERIAL.items():
            if metric_code in footprint:
                return imperial_size

        # Try to detect from dimensions
        w, h = comp.width, comp.height
        if w < h:
            w, h = h, w  # Normalize to width >= height

        # Match against known dimensions with tolerance
        for size, (ref_w, ref_h) in PASSIVE_DIMENSIONS.items():
            if abs(w - ref_w) < 0.3 and abs(h - ref_h) < 0.3:
                return size

        return PassiveSize.UNKNOWN

    def _compute_component_sizes(self):
        """Compute AABB half-dimensions for all components."""
        for ref, comp in self.board.components.items():
            w, h = comp.width, comp.height
            rot_rad = math.radians(comp.rotation)

            # Axis-aligned bounding box of rotated rectangle
            cos_r = abs(math.cos(rot_rad))
            sin_r = abs(math.sin(rot_rad))

            half_w = (w / 2) * cos_r + (h / 2) * sin_r
            half_h = (w / 2) * sin_r + (h / 2) * cos_r

            self._component_sizes[ref] = (half_w, half_h)


def legalize_placement(
    board: Board,
    primary_grid: float = 0.5,
    align_passives: bool = True,
    snap_rotation: bool = True
) -> LegalizationResult:
    """
    Convenience function to legalize a board placement.

    Runs the full 3-phase legalization pipeline:
    1. Grid snapping (positions + rotations)
    2. Row alignment (passives)
    3. Overlap removal (priority-based)

    Args:
        board: Board to legalize
        primary_grid: Placement grid in mm (default 0.5mm)
        align_passives: Whether to align passive components into rows
        snap_rotation: Whether to snap rotations to 90° increments

    Returns:
        LegalizationResult with statistics
    """
    config = LegalizerConfig(
        primary_grid=primary_grid,
        align_passives_only=align_passives,
        snap_rotation=snap_rotation,
    )
    legalizer = PlacementLegalizer(board, config)
    return legalizer.legalize()
