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

        if self.edge == "left":
            target = board.outline.origin_x + self.offset
            violation = abs(comp.x - target)
        elif self.edge == "right":
            target = board.outline.origin_x + board.outline.width - self.offset
            violation = abs(comp.x - target)
        elif self.edge == "top":
            target = board.outline.origin_y + self.offset
            violation = abs(comp.y - target)
        elif self.edge == "bottom":
            target = board.outline.origin_y + board.outline.height - self.offset
            violation = abs(comp.y - target)
        else:
            return (False, float('inf'))

        return (violation <= tolerance, max(0, violation - tolerance))

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref != self.component_ref:
            return (0.0, 0.0)

        comp = board.get_component(self.component_ref)
        if not comp:
            return (0.0, 0.0)

        fx, fy = 0.0, 0.0

        if self.edge == "left":
            target = board.outline.origin_x + self.offset
            fx = strength * (target - comp.x)
        elif self.edge == "right":
            target = board.outline.origin_x + board.outline.width - self.offset
            fx = strength * (target - comp.x)
        elif self.edge == "top":
            target = board.outline.origin_y + self.offset
            fy = strength * (target - comp.y)
        elif self.edge == "bottom":
            target = board.outline.origin_y + board.outline.height - self.offset
            fy = strength * (target - comp.y)

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

            # Check if within zone
            if comp.x < self.zone_x:
                total_violation += self.zone_x - comp.x
            elif comp.x > self.zone_x + self.zone_width:
                total_violation += comp.x - (self.zone_x + self.zone_width)

            if comp.y < self.zone_y:
                total_violation += self.zone_y - comp.y
            elif comp.y > self.zone_y + self.zone_height:
                total_violation += comp.y - (self.zone_y + self.zone_height)

        return (total_violation == 0, total_violation)

    def calculate_force(self, board: Board, ref: str,
                        strength: float) -> Tuple[float, float]:
        if ref not in self.components:
            return (0.0, 0.0)

        comp = board.get_component(ref)
        if not comp:
            return (0.0, 0.0)

        fx, fy = 0.0, 0.0

        if comp.x < self.zone_x:
            fx = strength * (self.zone_x - comp.x)
        elif comp.x > self.zone_x + self.zone_width:
            fx = strength * (self.zone_x + self.zone_width - comp.x)

        if comp.y < self.zone_y:
            fy = strength * (self.zone_y - comp.y)
        elif comp.y > self.zone_y + self.zone_height:
            fy = strength * (self.zone_y + self.zone_height - comp.y)

        return (fx, fy)


@dataclass
class GroupingConstraint(PlacementConstraint):
    """Keep a group of components together."""
    components: List[str] = field(default_factory=list)
    max_spread: float = 15.0  # mm maximum distance from centroid

    def __post_init__(self):
        self.constraint_type = ConstraintType.GROUPING
        if not self.description:
            self.description = f"Group {len(self.components)} components together"

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
