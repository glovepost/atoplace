"""Ratsnest crossing analysis for measuring routing complexity.

Counts wire crossings in the ratsnest (direct point-to-point connections)
to quantify how tangled the current pin assignment is. This metric drives
the optimization - fewer crossings = easier routing.

Uses efficient line segment intersection algorithms:
- Sweep line for global crossing count
- Local counting for swap impact analysis
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RatsnestEdge:
    """A single ratsnest connection between two pads."""
    net_name: str
    start: Tuple[float, float]  # (x, y) of start pad
    end: Tuple[float, float]    # (x, y) of end pad
    start_ref: str              # Component reference for start
    end_ref: str                # Component reference for end
    start_pad: str              # Pad number for start
    end_pad: str                # Pad number for end

    def __hash__(self):
        return hash((self.net_name, self.start_pad, self.end_pad))


@dataclass
class CrossingResult:
    """Result of crossing analysis."""
    total_crossings: int
    edges_analyzed: int

    # Crossings per component
    crossings_by_component: Dict[str, int] = field(default_factory=dict)

    # Most tangled nets (sorted by crossing count)
    worst_nets: List[Tuple[str, int]] = field(default_factory=list)

    # Crossing pairs (which nets cross each other)
    crossing_pairs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def crossing_density(self) -> float:
        """Crossings per edge (lower is better)."""
        if self.edges_analyzed == 0:
            return 0.0
        return self.total_crossings / self.edges_analyzed


class CrossingCounter:
    """Counts wire crossings in the ratsnest.

    Uses line segment intersection to count how many ratsnest wires cross.
    This is a key metric for pin swap optimization - minimizing crossings
    typically results in easier routing.
    """

    def __init__(self, board: "Board"):
        """
        Initialize crossing counter.

        Args:
            board: Board abstraction with components and nets
        """
        self.board = board
        self._edges: List[RatsnestEdge] = []
        self._build_ratsnest()

    def _build_ratsnest(self):
        """Build ratsnest edges from board connectivity."""
        self._edges = []

        for net_name, net in self.board.nets.items():
            if net.is_power or net.is_ground:
                # Skip power/ground for crossing analysis
                continue

            if len(net.connections) < 2:
                continue

            # Build minimum spanning tree for multi-pin nets
            # For simplicity, connect sequentially (first to second, second to third, etc.)
            pads_with_pos = []
            for comp_ref, pad_num in net.connections:
                comp = self.board.get_component(comp_ref)
                if not comp:
                    continue

                pad = comp.get_pad_by_number(pad_num)
                if not pad:
                    continue

                x, y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                pads_with_pos.append((comp_ref, pad_num, x, y))

            # Create edges using nearest neighbor chain
            if len(pads_with_pos) >= 2:
                edges = self._nearest_neighbor_mst(pads_with_pos, net_name)
                self._edges.extend(edges)

    def _nearest_neighbor_mst(
        self,
        pads: List[Tuple[str, str, float, float]],
        net_name: str
    ) -> List[RatsnestEdge]:
        """Build ratsnest edges using nearest neighbor MST approximation."""
        if len(pads) < 2:
            return []

        edges = []
        remaining = set(range(len(pads)))
        current = 0
        remaining.remove(0)

        while remaining:
            # Find nearest unconnected pad
            best_idx = None
            best_dist = float('inf')

            cx, cy = pads[current][2], pads[current][3]

            for idx in remaining:
                px, py = pads[idx][2], pads[idx][3]
                dist = (px - cx) ** 2 + (py - cy) ** 2

                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                # Create edge
                c_ref, c_pad = pads[current][0], pads[current][1]
                n_ref, n_pad = pads[best_idx][0], pads[best_idx][1]

                edge = RatsnestEdge(
                    net_name=net_name,
                    start=(pads[current][2], pads[current][3]),
                    end=(pads[best_idx][2], pads[best_idx][3]),
                    start_ref=c_ref,
                    end_ref=n_ref,
                    start_pad=c_pad,
                    end_pad=n_pad
                )
                edges.append(edge)

                remaining.remove(best_idx)
                current = best_idx

        return edges

    def count_all(self) -> CrossingResult:
        """
        Count all crossings in the ratsnest.

        Returns:
            CrossingResult with crossing statistics
        """
        total_crossings = 0
        crossings_by_net: Dict[str, int] = defaultdict(int)
        crossings_by_component: Dict[str, int] = defaultdict(int)
        crossing_pairs: List[Tuple[str, str]] = []

        # O(n²) pairwise comparison - acceptable for typical board sizes
        n = len(self._edges)
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = self._edges[i], self._edges[j]

                # Skip edges from same net (they share endpoints)
                if e1.net_name == e2.net_name:
                    continue

                if self._segments_intersect(e1.start, e1.end, e2.start, e2.end):
                    total_crossings += 1
                    crossings_by_net[e1.net_name] += 1
                    crossings_by_net[e2.net_name] += 1
                    crossing_pairs.append((e1.net_name, e2.net_name))

                    # Track by component
                    crossings_by_component[e1.start_ref] += 1
                    crossings_by_component[e1.end_ref] += 1
                    crossings_by_component[e2.start_ref] += 1
                    crossings_by_component[e2.end_ref] += 1

        # Sort nets by crossing count
        worst_nets = sorted(
            crossings_by_net.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]  # Top 20 worst nets

        logger.info(f"Counted {total_crossings} crossings in {n} edges")

        return CrossingResult(
            total_crossings=total_crossings,
            edges_analyzed=n,
            crossings_by_component=dict(crossings_by_component),
            worst_nets=worst_nets,
            crossing_pairs=crossing_pairs[:100]  # Limit stored pairs
        )

    def count_for_component(self, ref: str) -> int:
        """
        Count crossings involving a specific component.

        Args:
            ref: Component reference

        Returns:
            Number of crossings involving this component
        """
        crossings = 0
        component_edges = [
            e for e in self._edges
            if e.start_ref == ref or e.end_ref == ref
        ]

        for e1 in component_edges:
            for e2 in self._edges:
                if e1 is e2 or e1.net_name == e2.net_name:
                    continue

                if self._segments_intersect(e1.start, e1.end, e2.start, e2.end):
                    crossings += 1

        return crossings

    def count_for_swap(
        self,
        ref: str,
        pin_swaps: Dict[str, str]
    ) -> int:
        """
        Count crossings if specific pins were swapped.

        This is used to evaluate potential swap assignments without
        actually modifying the board.

        Args:
            ref: Component reference
            pin_swaps: Dict mapping old pad numbers to new pad numbers

        Returns:
            Estimated crossing count with the swap applied
        """
        comp = self.board.get_component(ref)
        if not comp:
            return 0

        # Create modified edges with swapped positions
        modified_edges = []

        for edge in self._edges:
            new_edge = edge

            # Check if this edge needs modification
            if edge.start_ref == ref and edge.start_pad in pin_swaps:
                new_pad = comp.get_pad_by_number(pin_swaps[edge.start_pad])
                if new_pad:
                    new_start = new_pad.absolute_position(comp.x, comp.y, comp.rotation)
                    new_edge = RatsnestEdge(
                        net_name=edge.net_name,
                        start=new_start,
                        end=edge.end,
                        start_ref=edge.start_ref,
                        end_ref=edge.end_ref,
                        start_pad=pin_swaps[edge.start_pad],
                        end_pad=edge.end_pad
                    )

            elif edge.end_ref == ref and edge.end_pad in pin_swaps:
                new_pad = comp.get_pad_by_number(pin_swaps[edge.end_pad])
                if new_pad:
                    new_end = new_pad.absolute_position(comp.x, comp.y, comp.rotation)
                    new_edge = RatsnestEdge(
                        net_name=edge.net_name,
                        start=edge.start,
                        end=new_end,
                        start_ref=edge.start_ref,
                        end_ref=edge.end_ref,
                        start_pad=edge.start_pad,
                        end_pad=pin_swaps[edge.end_pad]
                    )

            modified_edges.append(new_edge)

        # Count crossings with modified edges
        crossings = 0
        n = len(modified_edges)
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = modified_edges[i], modified_edges[j]

                if e1.net_name == e2.net_name:
                    continue

                if self._segments_intersect(e1.start, e1.end, e2.start, e2.end):
                    crossings += 1

        return crossings

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> bool:
        """
        Check if two line segments intersect.

        Uses the cross product method for robust intersection detection.

        Args:
            p1, p2: Endpoints of first segment
            p3, p4: Endpoints of second segment

        Returns:
            True if segments intersect (excluding endpoints)
        """
        def cross(o, a, b):
            """2D cross product of OA and OB vectors."""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        # Check for proper intersection (segments cross each other)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Skip collinear and endpoint cases for crossing count
        return False

    @property
    def edges(self) -> List[RatsnestEdge]:
        """Get all ratsnest edges."""
        return self._edges

    def refresh(self):
        """Rebuild ratsnest after board changes."""
        self._build_ratsnest()
