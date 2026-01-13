"""
Placement Constraints

Defines constraint types and a constraint solver for placement optimization.
Constraints can be extracted from natural language or defined programmatically.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math

from ..board.abstraction import Board, Component


class ConstraintType(Enum):
    """Types of placement constraints."""
    PROXIMITY = "proximity"          # Keep components close
    EDGE_PLACEMENT = "edge"          # Place on board edge
    ZONE_ASSIGNMENT = "zone"         # Keep in specific area
    ORIENTATION = "orientation"      # Specific rotation
    LAYER_PREFERENCE = "layer"       # Top/bottom layer
    KEEP_OUT = "keepout"             # Exclude from area
    GROUPING = "group"               # Keep components together
    ALIGNMENT = "alignment"          # Align components
    SEPARATION = "separation"        # Keep components apart
    FIXED = "fixed"                  # Lock position


@dataclass
class PlacementConstraint:
    """Base class for all placement constraints."""
    constraint_type: ConstraintType = ConstraintType.PROXIMITY  # Default, overridden by subclasses
    priority: str = "preferred"  # "required", "preferred", "optional"
    description: str = ""
    source_text: str = ""  # Original natural language if applicable
    confidence: float = 1.0  # How confident we are in this constraint

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        """
        Check if constraint is satisfied.

        Returns:
            (satisfied, violation_amount) - violation is 0 if satisfied
        """
        raise NotImplementedError

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        """
        Calculate force to satisfy this constraint.

        Returns:
            (fx, fy) force vector
        """
        raise NotImplementedError


@dataclass
class ProximityConstraint(PlacementConstraint):
    """Keep target component close to anchor component."""
    target_ref: str = ""
    anchor_ref: str = ""
    max_distance: float = 5.0  # mm
    ideal_distance: float = 2.0  # mm

    def __post_init__(self):
        self.constraint_type = ConstraintType.PROXIMITY
        if not self.description:
            self.description = f"Keep {self.target_ref} within {self.max_distance}mm of {self.anchor_ref}"

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        target = board.get_component(self.target_ref)
        anchor = board.get_component(self.anchor_ref)

        if not target or not anchor:
            return (False, float('inf'))

        distance = target.distance_to(anchor)
        if distance <= self.max_distance:
            return (True, 0.0)
        return (False, distance - self.max_distance)

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref != self.target_ref:
            return (0.0, 0.0)

        target = board.get_component(self.target_ref)
        anchor = board.get_component(self.anchor_ref)

        if not target or not anchor:
            return (0.0, 0.0)

        dx = anchor.x - target.x
        dy = anchor.y - target.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance <= self.ideal_distance or distance < 0.1:
            return (0.0, 0.0)

        # Pull toward anchor
        force_mag = strength * (distance - self.ideal_distance)
        return (force_mag * dx / distance, force_mag * dy / distance)


@dataclass
class EdgeConstraint(PlacementConstraint):
    """Place component on a board edge."""
    component_ref: str = ""
    edge: str = "left"  # "left", "right", "top", "bottom"
    offset: float = 2.0  # mm from edge

    def __post_init__(self):
        self.constraint_type = ConstraintType.EDGE_PLACEMENT
        if not self.description:
            self.description = f"Place {self.component_ref} on {self.edge} edge"

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        comp = board.get_component(self.component_ref)
        if not comp:
            return (False, float('inf'))

        tolerance = 1.0  # mm

        # Get component bounding box to account for component size
        bbox = comp.get_bounding_box()  # (min_x, min_y, max_x, max_y)
        comp_left = bbox[0]
        comp_bottom = bbox[1]
        comp_right = bbox[2]
        comp_top = bbox[3]

        # Use get_edge() which properly handles polygon outlines
        try:
            edge_coord = board.outline.get_edge(self.edge)
        except ValueError:
            return (False, float('inf'))

        if self.edge in ("left", "right"):
            # For left edge: component's left side should be at offset from edge
            # For right edge: component's right side should be at offset from edge
            if self.edge == "left":
                target = edge_coord + self.offset
                violation = abs(comp_left - target)
            else:
                target = edge_coord - self.offset
                violation = abs(comp_right - target)
        else:
            # For top edge: component's top should be at offset from edge
            # For bottom edge: component's bottom should be at offset from edge
            if self.edge == "top":
                target = edge_coord + self.offset
                violation = abs(comp_top - target)
            else:
                target = edge_coord - self.offset
                violation = abs(comp_bottom - target)

        return (violation <= tolerance, max(0, violation - tolerance))

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref != self.component_ref:
            return (0.0, 0.0)

        comp = board.get_component(self.component_ref)
        if not comp:
            return (0.0, 0.0)

        # Get component bounding box to account for component size
        bbox = comp.get_bounding_box()
        comp_left = bbox[0]
        comp_bottom = bbox[1]
        comp_right = bbox[2]
        comp_top = bbox[3]

        # Use get_edge() which properly handles polygon outlines
        try:
            edge_coord = board.outline.get_edge(self.edge)
        except ValueError:
            return (0.0, 0.0)

        fx, fy = 0.0, 0.0

        if self.edge == "left":
            target = edge_coord + self.offset
            fx = strength * (target - comp_left)
        elif self.edge == "right":
            target = edge_coord - self.offset
            fx = strength * (target - comp_right)
        elif self.edge == "top":
            target = edge_coord + self.offset
            fy = strength * (target - comp_top)
        elif self.edge == "bottom":
            target = edge_coord - self.offset
            fy = strength * (target - comp_bottom)

        return (fx, fy)


@dataclass
class ZoneConstraint(PlacementConstraint):
    """Keep components within a specific zone."""
    components: List[str] = field(default_factory=list)
    zone_x: float = 0.0
    zone_y: float = 0.0
    zone_width: float = 50.0
    zone_height: float = 50.0

    def __post_init__(self):
        self.constraint_type = ConstraintType.ZONE_ASSIGNMENT
        if not self.description:
            self.description = f"Keep {len(self.components)} components in zone"

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        total_violation = 0.0

        for ref in self.components:
            comp = board.get_component(ref)
            if not comp:
                continue

            # Get component bounding box to account for component size
            bbox = comp.get_bounding_box()  # (min_x, min_y, max_x, max_y)
            comp_left = bbox[0]
            comp_bottom = bbox[1]
            comp_right = bbox[2]
            comp_top = bbox[3]

            # Check if entire component body is within zone
            if comp_left < self.zone_x:
                total_violation += self.zone_x - comp_left
            if comp_right > self.zone_x + self.zone_width:
                total_violation += comp_right - (self.zone_x + self.zone_width)

            if comp_bottom < self.zone_y:
                total_violation += self.zone_y - comp_bottom
            if comp_top > self.zone_y + self.zone_height:
                total_violation += comp_top - (self.zone_y + self.zone_height)

        return (total_violation == 0, total_violation)

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref not in self.components:
            return (0.0, 0.0)

        comp = board.get_component(ref)
        if not comp:
            return (0.0, 0.0)

        # Get component bounding box to account for component size
        bbox = comp.get_bounding_box()
        comp_left = bbox[0]
        comp_bottom = bbox[1]
        comp_right = bbox[2]
        comp_top = bbox[3]

        fx, fy = 0.0, 0.0

        # Push component into zone if any part is outside
        if comp_left < self.zone_x:
            fx = strength * (self.zone_x - comp_left)
        elif comp_right > self.zone_x + self.zone_width:
            fx = strength * (self.zone_x + self.zone_width - comp_right)

        if comp_bottom < self.zone_y:
            fy = strength * (self.zone_y - comp_bottom)
        elif comp_top > self.zone_y + self.zone_height:
            fy = strength * (self.zone_y + self.zone_height - comp_top)

        return (fx, fy)


@dataclass
class GroupingConstraint(PlacementConstraint):
    """Keep a group of components together with optional bounding box optimization."""
    components: List[str] = field(default_factory=list)
    max_spread: float = 15.0  # mm maximum distance from centroid

    # Bounding box optimization parameters
    optimize_bbox: bool = False  # Enable bbox minimization
    bbox_strength: float = 1.0   # Multiplier for bbox forces relative to centroid
    min_clearance: float = 0.25  # Minimum clearance between components in group (mm)

    def __post_init__(self):
        self.constraint_type = ConstraintType.GROUPING
        if not self.description:
            bbox_note = " (bbox optimized)" if self.optimize_bbox else ""
            self.description = f"Group {len(self.components)} components together{bbox_note}"

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        if len(self.components) < 2:
            return (True, 0.0)

        # Calculate centroid
        cx, cy = self._get_centroid(board)

        # Check distances from centroid
        max_dist = 0.0
        for ref in self.components:
            comp = board.get_component(ref)
            if comp:
                dist = math.sqrt((comp.x - cx)**2 + (comp.y - cy)**2)
                max_dist = max(max_dist, dist)

        if max_dist <= self.max_spread:
            return (True, 0.0)
        return (False, max_dist - self.max_spread)

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref not in self.components:
            return (0.0, 0.0)

        comp = board.get_component(ref)
        if not comp:
            return (0.0, 0.0)

        cx, cy = self._get_centroid(board)
        dx = cx - comp.x
        dy = cy - comp.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance <= self.max_spread / 2 or distance < 0.1:
            return (0.0, 0.0)

        # Pull toward centroid
        force_mag = strength * (distance - self.max_spread / 2) / distance
        return (force_mag * dx, force_mag * dy)

    def _get_centroid(self, board: Board) -> Tuple[float, float]:
        """Calculate centroid of all components in group."""
        total_x, total_y = 0.0, 0.0
        count = 0

        for ref in self.components:
            comp = board.get_component(ref)
            if comp:
                total_x += comp.x
                total_y += comp.y
                count += 1

        if count == 0:
            return (0.0, 0.0)

        return (total_x / count, total_y / count)

    def calculate_forces(self, state, board: Board,
                         strength: float) -> Dict[str, Tuple[float, float]]:
        """Calculate forces for all components in group.

        Combines centroid-pull with optional bounding box optimization.
        This method is called by ForceDirectedRefiner when the constraint
        implements the state-aware interface.

        Args:
            state: PlacementState with current positions
            board: Board instance
            strength: Base force strength

        Returns:
            Dict mapping component refs to (fx, fy) force vectors
        """
        forces: Dict[str, Tuple[float, float]] = {}

        if len(self.components) < 2:
            return forces

        # Get current positions from state
        positions = {}
        for ref in self.components:
            if ref in state.positions:
                positions[ref] = state.positions[ref]

        if len(positions) < 2:
            return forces

        # 1. Calculate centroid forces (existing logic)
        centroid_forces = self._calculate_centroid_forces(positions, board, strength)

        # 2. Calculate bbox forces if enabled
        bbox_forces: Dict[str, Tuple[float, float]] = {}
        if self.optimize_bbox:
            bbox_forces = self._calculate_bbox_forces(
                positions, board, strength * self.bbox_strength
            )

        # 3. Combine forces (sum them)
        for ref in self.components:
            fx_c, fy_c = centroid_forces.get(ref, (0.0, 0.0))
            fx_b, fy_b = bbox_forces.get(ref, (0.0, 0.0))
            combined_fx = fx_c + fx_b
            combined_fy = fy_c + fy_b
            if combined_fx != 0 or combined_fy != 0:
                forces[ref] = (combined_fx, combined_fy)

        return forces

    def _calculate_centroid_forces(self, positions: Dict[str, Tuple[float, float]],
                                   board: Board,
                                   strength: float) -> Dict[str, Tuple[float, float]]:
        """Calculate forces pulling components toward group centroid."""
        forces: Dict[str, Tuple[float, float]] = {}

        if len(positions) < 2:
            return forces

        # Calculate centroid from positions
        cx = sum(p[0] for p in positions.values()) / len(positions)
        cy = sum(p[1] for p in positions.values()) / len(positions)

        for ref, (x, y) in positions.items():
            dx = cx - x
            dy = cy - y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= self.max_spread / 2 or distance < 0.1:
                continue

            # Pull toward centroid
            force_mag = strength * (distance - self.max_spread / 2) / distance
            forces[ref] = (force_mag * dx, force_mag * dy)

        return forces

    def _calculate_bbox_forces(self, positions: Dict[str, Tuple[float, float]],
                               board: Board,
                               strength: float) -> Dict[str, Tuple[float, float]]:
        """Calculate inward forces to minimize module bounding box.

        Algorithm:
        1. Compute current AABB of the module (including component sizes)
        2. Compute minimum feasible AABB based on total component area
        3. For each component on the periphery, apply inward force
        4. Scale force by excess bbox dimension and module size
        """
        forces: Dict[str, Tuple[float, float]] = {}

        if len(positions) < 2:
            return forces

        # Get component sizes for accurate bbox calculation
        comp_sizes: Dict[str, Tuple[float, float]] = {}  # ref -> (half_width, half_height)
        for ref in positions:
            comp = board.get_component(ref)
            if comp:
                bbox = comp.get_bounding_box()
                half_w = (bbox[2] - bbox[0]) / 2
                half_h = (bbox[3] - bbox[1]) / 2
                comp_sizes[ref] = (half_w, half_h)
            else:
                comp_sizes[ref] = (1.0, 1.0)  # Default 2mm x 2mm

        # 1. Calculate current module bounding box
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for ref, (x, y) in positions.items():
            hw, hh = comp_sizes.get(ref, (1.0, 1.0))
            min_x = min(min_x, x - hw)
            min_y = min(min_y, y - hh)
            max_x = max(max_x, x + hw)
            max_y = max(max_y, y + hh)

        current_width = max_x - min_x
        current_height = max_y - min_y

        # Prevent division by zero
        if current_width < 0.1 or current_height < 0.1:
            return forces

        # 2. Estimate minimum feasible dimensions
        # Based on total component area + clearance overhead
        total_area = sum(4 * hw * hh for hw, hh in comp_sizes.values())
        clearance_overhead = len(positions) * self.min_clearance * 2

        # Preserve current aspect ratio
        aspect = current_width / current_height
        min_height = math.sqrt(total_area / max(aspect, 0.1)) + clearance_overhead
        min_width = aspect * min_height + clearance_overhead

        # 3. Calculate excess dimensions
        excess_width = max(0.0, current_width - min_width)
        excess_height = max(0.0, current_height - min_height)

        # Early exit if already compact
        if excess_width < 0.1 and excess_height < 0.1:
            return forces

        # 4. Scale strength by module size (larger modules need more force)
        n = len(positions)
        if n <= 3:
            size_mult = 0.3  # Small modules: light touch
        elif n <= 9:
            size_mult = 1.0  # Medium modules: standard
        else:
            size_mult = 1.5  # Large modules: stronger force to counter repulsion

        effective_strength = strength * size_mult

        # 5. Apply inward forces to peripheral components
        edge_threshold = 0.5  # mm - component is "on edge" if within this distance

        for ref, (x, y) in positions.items():
            hw, hh = comp_sizes.get(ref, (1.0, 1.0))
            fx, fy = 0.0, 0.0

            # Distance from component edge to module bbox edge
            dist_to_left = (x - hw) - min_x
            dist_to_right = max_x - (x + hw)
            dist_to_bottom = (y - hh) - min_y
            dist_to_top = max_y - (y + hh)

            # Cap excess to prevent overshooting (max 2mm force contribution)
            capped_excess_w = min(excess_width, 2.0)
            capped_excess_h = min(excess_height, 2.0)

            # Pull components on left edge toward center (positive x)
            if dist_to_left < edge_threshold and excess_width > 0:
                fx += effective_strength * capped_excess_w * 0.5

            # Pull components on right edge toward center (negative x)
            if dist_to_right < edge_threshold and excess_width > 0:
                fx -= effective_strength * capped_excess_w * 0.5

            # Pull components on bottom edge toward center (positive y)
            if dist_to_bottom < edge_threshold and excess_height > 0:
                fy += effective_strength * capped_excess_h * 0.5

            # Pull components on top edge toward center (negative y)
            if dist_to_top < edge_threshold and excess_height > 0:
                fy -= effective_strength * capped_excess_h * 0.5

            if fx != 0 or fy != 0:
                forces[ref] = (fx, fy)

        return forces


@dataclass
class SeparationConstraint(PlacementConstraint):
    """Keep two groups of components separated."""
    group_a: List[str] = field(default_factory=list)
    group_b: List[str] = field(default_factory=list)
    min_separation: float = 10.0  # mm between group centroids

    def __post_init__(self):
        self.constraint_type = ConstraintType.SEPARATION
        if not self.description:
            self.description = f"Keep groups separated by {self.min_separation}mm"

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        # If either group is empty, constraint is trivially satisfied
        group_a_valid = [r for r in self.group_a if board.get_component(r)]
        group_b_valid = [r for r in self.group_b if board.get_component(r)]
        if not group_a_valid or not group_b_valid:
            return (True, 0.0)

        cx_a, cy_a = self._get_centroid(board, self.group_a)
        cx_b, cy_b = self._get_centroid(board, self.group_b)

        distance = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)

        if distance >= self.min_separation:
            return (True, 0.0)
        return (False, self.min_separation - distance)

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        # Determine which group this component belongs to
        if ref in self.group_a:
            other_group = self.group_b
        elif ref in self.group_b:
            other_group = self.group_a
        else:
            return (0.0, 0.0)

        comp = board.get_component(ref)
        if not comp:
            return (0.0, 0.0)

        # Check if other group has any valid components - avoid pushing toward origin
        if not other_group or not any(board.get_component(r) for r in other_group):
            return (0.0, 0.0)

        # Push away from other group's centroid
        cx, cy = self._get_centroid(board, other_group)
        dx = comp.x - cx
        dy = comp.y - cy
        distance = math.sqrt(dx*dx + dy*dy)

        if distance >= self.min_separation or distance < 0.1:
            return (0.0, 0.0)

        # Force points from centroid toward component (pushes away)
        force_mag = strength * (self.min_separation - distance) / distance
        return (force_mag * dx, force_mag * dy)

    def _get_centroid(self, board: Board, refs: List[str]) -> Tuple[float, float]:
        """Calculate centroid of components."""
        total_x, total_y = 0.0, 0.0
        count = 0

        for ref in refs:
            comp = board.get_component(ref)
            if comp:
                total_x += comp.x
                total_y += comp.y
                count += 1

        if count == 0:
            return (0.0, 0.0)

        return (total_x / count, total_y / count)


@dataclass
class FixedConstraint(PlacementConstraint):
    """Lock a component at a specific position and/or rotation."""
    component_ref: str = ""
    x: Optional[float] = None  # None means don't constrain position
    y: Optional[float] = None
    rotation: Optional[float] = None  # Target rotation in degrees
    position_only: bool = False  # If True, only constrain position
    rotation_only: bool = False  # If True, only constrain rotation

    def __post_init__(self):
        self.constraint_type = ConstraintType.FIXED
        if not self.description:
            parts = []
            if self.x is not None and self.y is not None:
                parts.append(f"at ({self.x}, {self.y})")
            if self.rotation is not None:
                parts.append(f"rotated {self.rotation}°")
            self.description = f"Fix {self.component_ref} " + " ".join(parts)

    def is_satisfied(self, board: Board) -> Tuple[bool, float]:
        comp = board.get_component(self.component_ref)
        if not comp:
            return (False, float('inf'))

        violation = 0.0

        # Check position if specified
        if self.x is not None and self.y is not None and not self.rotation_only:
            dist = math.sqrt((comp.x - self.x)**2 + (comp.y - self.y)**2)
            violation += dist

        # Check rotation if specified
        if self.rotation is not None and not self.position_only:
            rot_diff = abs(comp.rotation - self.rotation)
            # Handle wraparound (e.g., 350° vs 10° should be 20° difference)
            rot_diff = min(rot_diff, 360 - rot_diff)
            violation += rot_diff / 10  # Scale rotation violation

        return (violation < 0.1, violation)

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref != self.component_ref:
            return (0.0, 0.0)

        comp = board.get_component(self.component_ref)
        if not comp:
            return (0.0, 0.0)

        # Only apply position force if position is constrained
        if self.x is None or self.y is None or self.rotation_only:
            return (0.0, 0.0)

        # Very strong force to fixed position
        return (strength * 10 * (self.x - comp.x),
                strength * 10 * (self.y - comp.y))

    def get_target_rotation(self) -> Optional[float]:
        """Get the target rotation if this is a rotation constraint."""
        if self.rotation is not None and not self.position_only:
            return self.rotation
        return None


class ConstraintSolver:
    """Manages and evaluates placement constraints."""

    def __init__(self, board: Board):
        self.board = board
        self.constraints: List[PlacementConstraint] = []

    def add_constraint(self, constraint: PlacementConstraint):
        """Add a constraint."""
        self.constraints.append(constraint)

    def add_constraints(self, constraints: List[PlacementConstraint]):
        """Add multiple constraints."""
        self.constraints.extend(constraints)

    def clear(self):
        """Clear all constraints."""
        self.constraints.clear()

    def evaluate_all(self) -> Dict[str, Tuple[bool, float]]:
        """
        Evaluate all constraints.

        Returns:
            Dict mapping constraint description to (satisfied, violation)
        """
        results = {}
        for constraint in self.constraints:
            satisfied, violation = constraint.is_satisfied(self.board)
            results[constraint.description] = (satisfied, violation)
        return results

    def get_violations(self) -> List[Tuple[PlacementConstraint, float]]:
        """Get all violated constraints with violation amounts."""
        violations = []
        for constraint in self.constraints:
            satisfied, violation = constraint.is_satisfied(self.board)
            if not satisfied:
                violations.append((constraint, violation))
        return violations

    def all_satisfied(self) -> bool:
        """Check if all required constraints are satisfied."""
        for constraint in self.constraints:
            if constraint.priority == "required":
                satisfied, _ = constraint.is_satisfied(self.board)
                if not satisfied:
                    return False
        return True

    def calculate_forces(self, ref: str, strength: float
                         ) -> List[Tuple[float, float]]:
        """Calculate all constraint forces for a component."""
        forces = []
        for constraint in self.constraints:
            fx, fy = constraint.calculate_force(self.board, ref, strength)
            if fx != 0 or fy != 0:
                forces.append((fx, fy))
        return forces

    def get_constraint_score(self) -> float:
        """
        Calculate overall constraint satisfaction score.

        Returns:
            Score from 0.0 (all violated) to 1.0 (all satisfied)
        """
        if not self.constraints:
            return 1.0

        total_weight = 0.0
        satisfied_weight = 0.0

        weight_map = {"required": 3.0, "preferred": 2.0, "optional": 1.0}

        for constraint in self.constraints:
            weight = weight_map.get(constraint.priority, 1.0)
            total_weight += weight

            satisfied, _ = constraint.is_satisfied(self.board)
            if satisfied:
                satisfied_weight += weight

        return satisfied_weight / total_weight if total_weight > 0 else 1.0
