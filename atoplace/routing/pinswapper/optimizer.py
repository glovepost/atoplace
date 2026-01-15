"""Bipartite matching optimization for pin assignment.

Uses the Hungarian algorithm (also known as Kuhn-Munkres) to find optimal
pin-to-net assignments that minimize total wire length and crossings.

The key insight is that swapping pins within a group is equivalent to finding
the optimal matching between:
- Set A: Nets that need to connect to this component
- Set B: Available pins in the swap group

The cost matrix captures how "expensive" each assignment is based on:
- Wire length to other pads on the same net
- Crossing count impact
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math

from .detector import SwapGroup, SwappablePin

logger = logging.getLogger(__name__)


@dataclass
class SwapAssignment:
    """A single pin swap assignment."""
    from_pin: str    # Original pad number
    to_pin: str      # New pad number
    net: str         # Net being moved
    improvement: float  # Estimated improvement (lower is better)

    def __repr__(self) -> str:
        return f"{self.net}: {self.from_pin} → {self.to_pin}"


@dataclass
class MatchingResult:
    """Result of bipartite matching optimization."""
    success: bool
    assignments: List[SwapAssignment] = field(default_factory=list)

    # Metrics
    original_cost: float = 0.0
    optimized_cost: float = 0.0
    improvement_percent: float = 0.0

    # Swap count
    swaps_performed: int = 0

    failure_reason: str = ""

    @property
    def total_improvement(self) -> float:
        """Total cost reduction."""
        return self.original_cost - self.optimized_cost


class BipartiteMatcher:
    """Optimizes pin assignments using bipartite matching.

    Models the pin assignment problem as a minimum-cost bipartite matching:
    - Left side: Nets requiring pins
    - Right side: Available pins in the swap group
    - Edge weights: Cost of assigning net to pin (wire length, crossings)

    Uses the Hungarian algorithm for O(n³) optimal matching.
    """

    def __init__(self, board: "Board"):
        """
        Initialize matcher.

        Args:
            board: Board abstraction
        """
        self.board = board

    def optimize_group(
        self,
        group: SwapGroup,
        target_positions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> MatchingResult:
        """
        Find optimal pin assignments for a swap group.

        Args:
            group: The swap group to optimize
            target_positions: Optional dict of net_name -> target (x, y)
                            If not provided, calculates from other pads on net

        Returns:
            MatchingResult with optimal assignments
        """
        # Get connected pins (pins with nets assigned)
        connected = group.connected_pins
        if len(connected) < 2:
            return MatchingResult(
                success=False,
                failure_reason="Need at least 2 connected pins to optimize"
            )

        # Build target positions for each net
        if target_positions is None:
            target_positions = self._calculate_targets(group, connected)

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(group, connected, target_positions)

        # Run Hungarian algorithm
        assignment = self._hungarian(cost_matrix)

        # Convert assignment to swaps
        assignments, original_cost, optimized_cost = self._build_assignments(
            group, connected, cost_matrix, assignment
        )

        # Calculate improvement
        if original_cost > 0:
            improvement = (original_cost - optimized_cost) / original_cost * 100
        else:
            improvement = 0.0

        logger.info(
            f"Optimized {group.name}: {original_cost:.2f} -> {optimized_cost:.2f} "
            f"({improvement:.1f}% improvement)"
        )

        return MatchingResult(
            success=True,
            assignments=assignments,
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            improvement_percent=improvement,
            swaps_performed=sum(1 for a in assignments if a.from_pin != a.to_pin)
        )

    def _calculate_targets(
        self,
        group: SwapGroup,
        connected: List[SwappablePin]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate target positions for each net based on other pads."""
        targets = {}

        for pin in connected:
            if not pin.net_name:
                continue

            net = self.board.get_net(pin.net_name)
            if not net:
                continue

            # Find center of other pads on this net
            other_positions = []
            for comp_ref, pad_num in net.connections:
                # Skip the pad in our swap group
                if comp_ref == group.component_ref and pad_num == pin.pad_number:
                    continue

                comp = self.board.get_component(comp_ref)
                if not comp:
                    continue

                pad = comp.get_pad_by_number(pad_num)
                if not pad:
                    continue

                x, y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                other_positions.append((x, y))

            if other_positions:
                # Use centroid of other pads as target
                avg_x = sum(p[0] for p in other_positions) / len(other_positions)
                avg_y = sum(p[1] for p in other_positions) / len(other_positions)
                targets[pin.net_name] = (avg_x, avg_y)
            else:
                # No other pads - use current position as target
                targets[pin.net_name] = (pin.x, pin.y)

        return targets

    def _build_cost_matrix(
        self,
        group: SwapGroup,
        connected: List[SwappablePin],
        targets: Dict[str, Tuple[float, float]]
    ) -> List[List[float]]:
        """
        Build cost matrix for assignment problem.

        cost[i][j] = cost of assigning net i to pin j

        Cost is based on:
        - Euclidean distance from pin to target position
        - Could be extended with crossing impact
        """
        n = len(connected)
        cost_matrix = [[0.0] * n for _ in range(n)]

        for i, pin in enumerate(connected):
            if not pin.net_name or pin.net_name not in targets:
                continue

            target = targets[pin.net_name]

            # Calculate cost to each possible pin position
            for j, dest_pin in enumerate(group.pins):
                # Distance from destination pin to target
                dx = dest_pin.x - target[0]
                dy = dest_pin.y - target[1]
                distance = math.sqrt(dx*dx + dy*dy)

                cost_matrix[i][j] = distance

        return cost_matrix

    def _hungarian(self, cost_matrix: List[List[float]]) -> List[int]:
        """
        Hungarian algorithm for minimum cost bipartite matching.

        Args:
            cost_matrix: n x n cost matrix

        Returns:
            Assignment where assignment[i] = j means row i assigned to column j
        """
        n = len(cost_matrix)
        if n == 0:
            return []

        # Make a copy to avoid modifying original
        cost = [row[:] for row in cost_matrix]

        # Step 1: Subtract row minimum from each row
        for i in range(n):
            min_val = min(cost[i])
            for j in range(n):
                cost[i][j] -= min_val

        # Step 2: Subtract column minimum from each column
        for j in range(n):
            min_val = min(cost[i][j] for i in range(n))
            for i in range(n):
                cost[i][j] -= min_val

        # Initialize labels and slack
        u = [0.0] * n  # Row labels
        v = [0.0] * n  # Column labels
        match_row = [-1] * n  # match_row[j] = i if column j matched to row i
        match_col = [-1] * n  # match_col[i] = j if row i matched to column j

        for i in range(n):
            # Start augmenting path from row i
            links = [-1] * n  # links[j] = previous column in augmenting path
            mins = [float('inf')] * n  # mins[j] = minimum slack for column j
            visited = [False] * n

            # Initial slack calculation
            for j in range(n):
                if cost[i][j] - u[i] - v[j] < mins[j]:
                    mins[j] = cost[i][j] - u[i] - v[j]
                    links[j] = -1

            cur_row = i
            cur_col = -1

            while True:
                # Find minimum slack unvisited column
                delta = float('inf')
                for j in range(n):
                    if not visited[j] and mins[j] < delta:
                        delta = mins[j]
                        cur_col = j

                # Update labels
                u[cur_row] += delta
                for j in range(n):
                    if visited[j]:
                        v[j] -= delta
                    else:
                        mins[j] -= delta

                visited[cur_col] = True

                # Check if this column is unmatched
                if match_row[cur_col] < 0:
                    break

                # Move to the matched row
                cur_row = match_row[cur_col]

                # Update slack from new row
                for j in range(n):
                    if not visited[j]:
                        slack = cost[cur_row][j] - u[cur_row] - v[j]
                        if slack < mins[j]:
                            mins[j] = slack
                            links[j] = cur_col

            # Reconstruct augmenting path
            while cur_col >= 0:
                prev_col = links[cur_col]
                if prev_col < 0:
                    match_row[cur_col] = i
                    match_col[i] = cur_col
                else:
                    prev_row = match_row[prev_col]
                    match_row[cur_col] = prev_row
                    match_col[prev_row] = cur_col
                cur_col = prev_col

        return match_col

    def _build_assignments(
        self,
        group: SwapGroup,
        connected: List[SwappablePin],
        cost_matrix: List[List[float]],
        assignment: List[int]
    ) -> Tuple[List[SwapAssignment], float, float]:
        """Build swap assignments from Hungarian result."""
        assignments = []
        original_cost = 0.0
        optimized_cost = 0.0

        for i, pin in enumerate(connected):
            if not pin.net_name:
                continue

            # Original assignment (diagonal)
            original_cost += cost_matrix[i][i] if i < len(cost_matrix[i]) else 0

            # New assignment
            new_idx = assignment[i] if i < len(assignment) else i
            optimized_cost += cost_matrix[i][new_idx] if new_idx < len(cost_matrix[i]) else 0

            # Get the destination pin
            if new_idx < len(group.pins):
                dest_pin = group.pins[new_idx]

                improvement = (cost_matrix[i][i] - cost_matrix[i][new_idx]
                              if i < len(cost_matrix[i]) else 0)

                assignments.append(SwapAssignment(
                    from_pin=pin.pad_number,
                    to_pin=dest_pin.pad_number,
                    net=pin.net_name,
                    improvement=improvement
                ))

        return assignments, original_cost, optimized_cost

    def estimate_improvement(
        self,
        group: SwapGroup,
        crossing_counter: "CrossingCounter" = None
    ) -> float:
        """
        Estimate potential improvement from optimizing a group.

        Useful for prioritizing which groups to optimize first.

        Args:
            group: Swap group to evaluate
            crossing_counter: Optional crossing counter for enhanced estimation

        Returns:
            Estimated improvement percentage (0-100)
        """
        connected = group.connected_pins
        if len(connected) < 2:
            return 0.0

        # Calculate current total wire length
        targets = self._calculate_targets(group, connected)
        current_length = 0.0
        min_possible = 0.0

        for pin in connected:
            if not pin.net_name or pin.net_name not in targets:
                continue

            target = targets[pin.net_name]
            dx = pin.x - target[0]
            dy = pin.y - target[1]
            current_length += math.sqrt(dx*dx + dy*dy)

            # Find minimum possible distance to any pin in group
            min_dist = float('inf')
            for gpin in group.pins:
                dx = gpin.x - target[0]
                dy = gpin.y - target[1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
            min_possible += min_dist

        if current_length <= 0:
            return 0.0

        # Estimate improvement as ratio of reducible length
        potential = (current_length - min_possible) / current_length * 100
        return max(0.0, potential)
