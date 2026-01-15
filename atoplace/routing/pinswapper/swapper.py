"""Main PinSwapper orchestrator class.

Coordinates pin swap detection, optimization, and application:
1. Detects swap groups on components (FPGAs, MCUs, connectors)
2. Analyzes ratsnest crossings to measure current complexity
3. Optimizes pin assignments using bipartite matching
4. Applies swaps to the board and generates constraint files

This is Phase 0 of the routing pipeline - run before actual routing.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .detector import SwapGroupDetector, SwapGroup, SwapGroupType
from .crossing import CrossingCounter, CrossingResult
from .optimizer import BipartiteMatcher, MatchingResult, SwapAssignment
from .constraints import ConstraintGenerator, ConstraintFormat

logger = logging.getLogger(__name__)


@dataclass
class SwapConfig:
    """Configuration for pin swapping."""
    # Detection settings
    min_group_size: int = 2          # Minimum pins in group to optimize
    min_confidence: float = 0.5      # Minimum detection confidence

    # Optimization settings
    min_improvement: float = 5.0     # Minimum improvement % to apply
    preserve_diff_pairs: bool = True # Don't break differential pairs

    # What to optimize
    optimize_fpga: bool = True
    optimize_mcu: bool = True
    optimize_connectors: bool = True
    optimize_memory: bool = True


@dataclass
class SwapResult:
    """Result of pin swap optimization."""
    success: bool
    component_ref: str

    # Groups and results
    groups_detected: int = 0
    groups_optimized: int = 0
    total_swaps: int = 0

    # Individual results per group
    group_results: Dict[str, MatchingResult] = field(default_factory=dict)

    # Crossing metrics
    original_crossings: int = 0
    final_crossings: int = 0

    # Wire length metrics
    original_wire_length: float = 0.0
    optimized_wire_length: float = 0.0

    failure_reason: str = ""

    @property
    def crossing_improvement(self) -> float:
        """Percentage reduction in crossings."""
        if self.original_crossings == 0:
            return 0.0
        return (self.original_crossings - self.final_crossings) / self.original_crossings * 100

    @property
    def wire_improvement(self) -> float:
        """Percentage reduction in wire length."""
        if self.original_wire_length <= 0:
            return 0.0
        return (self.original_wire_length - self.optimized_wire_length) / self.original_wire_length * 100


class PinSwapper:
    """Orchestrates pin swap optimization.

    Combines detection, analysis, optimization, and application into a
    high-level interface for reducing routing complexity through pin swaps.

    Usage:
        swapper = PinSwapper(board)

        # Optimize a specific component
        result = swapper.optimize_component("U1")

        # Or optimize all swappable components
        results = swapper.optimize_all()

        # Export constraint updates
        swapper.export_constraints("constraints.xdc")
    """

    def __init__(self, board: "Board", config: Optional[SwapConfig] = None):
        """
        Initialize pin swapper.

        Args:
            board: Board abstraction
            config: Optional configuration
        """
        self.board = board
        self.config = config or SwapConfig()

        # Initialize sub-components
        self._detector = SwapGroupDetector(board)
        self._crossing_counter = CrossingCounter(board)
        self._matcher = BipartiteMatcher(board)
        self._constraint_gen = ConstraintGenerator(board)

        # Track results
        self._results: Dict[str, SwapResult] = {}
        self._applied_swaps: List[Tuple[str, SwapAssignment]] = []

    def analyze_component(self, ref: str) -> Dict:
        """
        Analyze a component without applying changes.

        Args:
            ref: Component reference

        Returns:
            Analysis dict with groups, potential improvement, crossings
        """
        comp = self.board.get_component(ref)
        if not comp:
            return {"error": f"Component {ref} not found"}

        # Detect swap groups
        groups = self._detector.detect_component(ref)

        # Count current crossings involving this component
        current_crossings = self._crossing_counter.count_for_component(ref)

        # Estimate improvement potential for each group
        group_analysis = []
        for group in groups:
            if len(group.connected_pins) < self.config.min_group_size:
                continue

            potential = self._matcher.estimate_improvement(group)
            group_analysis.append({
                "name": group.name,
                "type": group.group_type.value,
                "pins": group.size,
                "connected_pins": len(group.connected_pins),
                "potential_improvement": f"{potential:.1f}%",
                "confidence": group.confidence
            })

        return {
            "component": ref,
            "footprint": comp.footprint,
            "total_pads": len(comp.pads),
            "swap_groups": len(groups),
            "current_crossings": current_crossings,
            "groups": group_analysis
        }

    def optimize_component(
        self,
        ref: str,
        apply: bool = True,
        dry_run: bool = False
    ) -> SwapResult:
        """
        Optimize pin assignments for a component.

        Args:
            ref: Component reference
            apply: Whether to apply swaps to the board
            dry_run: If True, calculate but don't apply

        Returns:
            SwapResult with optimization details
        """
        comp = self.board.get_component(ref)
        if not comp:
            return SwapResult(
                success=False,
                component_ref=ref,
                failure_reason=f"Component {ref} not found"
            )

        # Detect swap groups
        groups = self._detector.detect_component(ref)
        if not groups:
            return SwapResult(
                success=True,
                component_ref=ref,
                groups_detected=0
            )

        # Filter groups by config
        valid_groups = self._filter_groups(groups)

        if not valid_groups:
            return SwapResult(
                success=True,
                component_ref=ref,
                groups_detected=len(groups),
                failure_reason="No groups meet optimization criteria"
            )

        # Measure initial state
        initial_crossings = self._crossing_counter.count_for_component(ref)
        initial_wire_length = self._calculate_wire_length(ref, valid_groups)

        # Optimize each group
        group_results = {}
        total_swaps = 0
        optimized_count = 0

        for group in valid_groups:
            result = self._matcher.optimize_group(group)
            group_results[group.name] = result

            if result.success and result.improvement_percent >= self.config.min_improvement:
                optimized_count += 1
                total_swaps += result.swaps_performed

                # Apply swaps if requested
                if apply and not dry_run:
                    self._apply_group_swaps(ref, group, result)
                    self._constraint_gen.add_result(result, ref)

        # Measure final state
        if apply and not dry_run:
            self._crossing_counter.refresh()
        final_crossings = self._crossing_counter.count_for_component(ref)
        final_wire_length = self._calculate_wire_length(ref, valid_groups)

        result = SwapResult(
            success=True,
            component_ref=ref,
            groups_detected=len(groups),
            groups_optimized=optimized_count,
            total_swaps=total_swaps,
            group_results=group_results,
            original_crossings=initial_crossings,
            final_crossings=final_crossings,
            original_wire_length=initial_wire_length,
            optimized_wire_length=final_wire_length
        )

        self._results[ref] = result
        return result

    def optimize_all(
        self,
        refs: Optional[List[str]] = None,
        apply: bool = True
    ) -> Dict[str, SwapResult]:
        """
        Optimize pin assignments for multiple components.

        Args:
            refs: Specific components to optimize (None = all detected)
            apply: Whether to apply swaps to the board

        Returns:
            Dict mapping component refs to their SwapResults
        """
        if refs is None:
            # Detect all swappable components
            all_groups = self._detector.detect_all()
            refs = list(all_groups.keys())

        results = {}
        for ref in refs:
            logger.info(f"Optimizing {ref}...")
            result = self.optimize_component(ref, apply=apply)
            results[ref] = result

            if result.success and result.total_swaps > 0:
                logger.info(
                    f"  {ref}: {result.total_swaps} swaps, "
                    f"{result.crossing_improvement:.1f}% crossing reduction"
                )

        # Summary
        total_swaps = sum(r.total_swaps for r in results.values())
        total_crossing_reduction = sum(
            r.original_crossings - r.final_crossings
            for r in results.values()
        )

        logger.info(
            f"Optimized {len(results)} components: "
            f"{total_swaps} total swaps, "
            f"{total_crossing_reduction} crossings eliminated"
        )

        return results

    def _filter_groups(self, groups: List[SwapGroup]) -> List[SwapGroup]:
        """Filter groups based on configuration."""
        valid = []

        for group in groups:
            # Check minimum size
            if len(group.connected_pins) < self.config.min_group_size:
                continue

            # Check confidence
            if group.confidence < self.config.min_confidence:
                continue

            # Check type filters
            if group.group_type == SwapGroupType.FPGA_BANK and not self.config.optimize_fpga:
                continue
            if group.group_type == SwapGroupType.MCU_GPIO and not self.config.optimize_mcu:
                continue
            if group.group_type == SwapGroupType.CONNECTOR and not self.config.optimize_connectors:
                continue
            if group.group_type in (SwapGroupType.MEMORY_DATA, SwapGroupType.MEMORY_ADDR):
                if not self.config.optimize_memory:
                    continue

            valid.append(group)

        return valid

    def _apply_group_swaps(
        self,
        ref: str,
        group: SwapGroup,
        result: MatchingResult
    ):
        """Apply swaps from a matching result to the board."""
        comp = self.board.get_component(ref)
        if not comp:
            return

        for assignment in result.assignments:
            if assignment.from_pin == assignment.to_pin:
                continue

            # Get the net being swapped
            net = self.board.get_net(assignment.net)
            if not net:
                continue

            # Update the net connection
            # Remove old connection
            old_conn = (ref, assignment.from_pin)
            if old_conn in net.connections:
                net.connections.remove(old_conn)

            # Add new connection
            new_conn = (ref, assignment.to_pin)
            if new_conn not in net.connections:
                net.connections.append(new_conn)

            # Update pad net assignments
            from_pad = comp.get_pad_by_number(assignment.from_pin)
            to_pad = comp.get_pad_by_number(assignment.to_pin)

            if from_pad:
                from_pad.net = None
            if to_pad:
                to_pad.net = assignment.net

            # Track the swap
            self._applied_swaps.append((ref, assignment))

            logger.debug(
                f"Swapped {assignment.net}: "
                f"{assignment.from_pin} -> {assignment.to_pin}"
            )

    def _calculate_wire_length(
        self,
        ref: str,
        groups: List[SwapGroup]
    ) -> float:
        """Calculate total wire length for nets in swap groups."""
        total = 0.0
        seen_nets = set()

        for group in groups:
            for pin in group.connected_pins:
                if not pin.net_name or pin.net_name in seen_nets:
                    continue
                seen_nets.add(pin.net_name)

                net = self.board.get_net(pin.net_name)
                if not net:
                    continue

                # Calculate MST wire length for this net
                total += self._net_wire_length(net)

        return total

    def _net_wire_length(self, net: "Net") -> float:
        """Calculate wire length for a net using MST approximation."""
        if len(net.connections) < 2:
            return 0.0

        positions = []
        for comp_ref, pad_num in net.connections:
            comp = self.board.get_component(comp_ref)
            if not comp:
                continue
            pad = comp.get_pad_by_number(pad_num)
            if not pad:
                continue
            x, y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            positions.append((x, y))

        if len(positions) < 2:
            return 0.0

        # Simple MST approximation: connect nearest neighbors
        import math
        total = 0.0
        remaining = set(range(1, len(positions)))
        current = 0

        while remaining:
            best_dist = float('inf')
            best_idx = None

            cx, cy = positions[current]
            for idx in remaining:
                px, py = positions[idx]
                dist = math.sqrt((px - cx)**2 + (py - cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                total += best_dist
                remaining.remove(best_idx)
                current = best_idx

        return total

    def export_constraints(
        self,
        path: Path,
        format: Optional[ConstraintFormat] = None,
        comment: str = ""
    ):
        """
        Export accumulated pin swap constraints to a file.

        Args:
            path: Output file path
            format: Output format (auto-detected if not specified)
            comment: Optional header comment
        """
        self._constraint_gen.save(path, format, comment)

    def get_constraint_preview(
        self,
        format: ConstraintFormat = ConstraintFormat.XDC
    ) -> str:
        """
        Get a preview of the constraint file content.

        Args:
            format: Output format

        Returns:
            Rendered constraint file content
        """
        constraint_file = self._constraint_gen.generate(format)
        return constraint_file.render()

    def get_crossing_analysis(self) -> CrossingResult:
        """Get full ratsnest crossing analysis."""
        return self._crossing_counter.count_all()

    def reset(self):
        """Reset all state (clear applied swaps and results)."""
        self._results = {}
        self._applied_swaps = []
        self._constraint_gen.clear()
        self._crossing_counter.refresh()

    @property
    def results(self) -> Dict[str, SwapResult]:
        """Get all optimization results."""
        return self._results

    @property
    def applied_swaps(self) -> List[Tuple[str, SwapAssignment]]:
        """Get list of all applied swaps."""
        return self._applied_swaps

    @property
    def pending_constraint_count(self) -> int:
        """Get count of pending constraint updates."""
        return self._constraint_gen.update_count
