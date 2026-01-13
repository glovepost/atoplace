"""
AtoPlace Core API: Atomic Actions (Layout DSL)

This module provides high-level, atomic geometric operations that act as the
"hands" of the LLM. These functions enforce design rules and handle the
math of relative positioning, so the LLM doesn't have to.

Usage:
    from atoplace.api.actions import LayoutActions
    actions = LayoutActions(board)
    actions.place_next_to("C1", "U1", side="right", clearance=0.5)
"""

import math
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass

from ..board.abstraction import Board, Component

@dataclass
class ActionResult:
    """Result of an atomic action."""
    success: bool
    message: str
    modified_refs: List[str]


class LayoutActions:
    """The Layout Domain Specific Language (DSL)."""

    def __init__(self, board: Board):
        self.board = board

    def move_absolute(self, ref: str, x: float, y: float, rotation: Optional[float] = None) -> ActionResult:
        """Move component to absolute coordinates."""
        comp = self.board.components.get(ref)
        if not comp:
            return ActionResult(False, f"Component {ref} not found", [])

        if comp.locked:
            return ActionResult(False, f"Component {ref} is locked", [])

        comp.x = x
        comp.y = y
        if rotation is not None:
            comp.rotation = rotation

        return ActionResult(True, f"Moved {ref} to ({x:.2f}, {y:.2f})", [ref])

    def move_relative(self, ref: str, dx: float, dy: float) -> ActionResult:
        """Move component by a relative delta."""
        comp = self.board.components.get(ref)
        if not comp:
            return ActionResult(False, f"Component {ref} not found", [])

        if comp.locked:
            return ActionResult(False, f"Component {ref} is locked", [])

        comp.x += dx
        comp.y += dy
        return ActionResult(True, f"Moved {ref} by ({dx:.2f}, {dy:.2f})", [ref])

    def rotate(self, ref: str, angle: float) -> ActionResult:
        """Set absolute rotation of component."""
        comp = self.board.components.get(ref)
        if not comp:
            return ActionResult(False, f"Component {ref} not found", [])
        
        if comp.locked:
            return ActionResult(False, f"Component {ref} is locked", [])

        comp.rotation = angle % 360
        return ActionResult(True, f"Rotated {ref} to {angle:.1f}°", [ref])

    def place_next_to(
        self, 
        ref: str, 
        target_ref: str, 
        side: Literal["top", "bottom", "left", "right"] = "right", 
        clearance: float = 0.5,
        align: Literal["center", "top", "bottom", "left", "right"] = "center"
    ) -> ActionResult:
        """
        Place 'ref' next to 'target_ref' with specific clearance.
        
        This calculates the bounding boxes and snaps 'ref' to the edge of 'target_ref'.
        """
        comp = self.board.components.get(ref)
        target = self.board.components.get(target_ref)
        
        if not comp or not target:
            return ActionResult(False, "Component not found", [])
            
        if comp.locked:
            return ActionResult(False, f"Component {ref} is locked", [])

        # Get dimensions (accounting for rotation roughly - assuming 90 degree increments for Manhattan)
        w1, h1 = self._get_dims(comp)
        w2, h2 = self._get_dims(target)
        
        x2, y2 = target.x, target.y
        new_x, new_y = comp.x, comp.y

        if side == "right":
            new_x = x2 + (w2 / 2) + clearance + (w1 / 2)
            if align == "center": new_y = y2
            elif align == "top": new_y = y2 - (h2 / 2) + (h1 / 2)
            elif align == "bottom": new_y = y2 + (h2 / 2) - (h1 / 2)
            
        elif side == "left":
            new_x = x2 - (w2 / 2) - clearance - (w1 / 2)
            if align == "center": new_y = y2
            elif align == "top": new_y = y2 - (h2 / 2) + (h1 / 2)
            elif align == "bottom": new_y = y2 + (h2 / 2) - (h1 / 2)
            
        elif side == "top":
            new_y = y2 - (h2 / 2) - clearance - (h1 / 2)
            if align == "center": new_x = x2
            elif align == "left": new_x = x2 - (w2 / 2) + (w1 / 2)
            elif align == "right": new_x = x2 + (w2 / 2) - (w1 / 2)
            
        elif side == "bottom":
            new_y = y2 + (h2 / 2) + clearance + (h1 / 2)
            if align == "center": new_x = x2
            elif align == "left": new_x = x2 - (w2 / 2) + (w1 / 2)
            elif align == "right": new_x = x2 + (w2 / 2) - (w1 / 2)

        comp.x = new_x
        comp.y = new_y
        
        return ActionResult(True, f"Placed {ref} {side} of {target_ref}", [ref])

    def align_components(
        self, 
        refs: List[str], 
        axis: Literal["x", "y"] = "x", 
        anchor: Literal["first", "last", "center"] = "first"
    ) -> ActionResult:
        """Align a list of components along an axis."""
        if len(refs) < 2:
            return ActionResult(False, "Need at least 2 components to align", [])
            
        components = []
        for r in refs:
            c = self.board.components.get(r)
            if not c: return ActionResult(False, f"Component {r} not found", [])
            components.append(c)

        # Determine anchor value
        target_val = 0.0
        if axis == "x":
            # Align Y coordinates (make them a row)
            if anchor == "first": target_val = components[0].y
            elif anchor == "last": target_val = components[-1].y
            elif anchor == "center": target_val = sum(c.y for c in components) / len(components)
            
            for c in components:
                if not c.locked: c.y = target_val
                
        elif axis == "y":
            # Align X coordinates (make them a col)
            if anchor == "first": target_val = components[0].x
            elif anchor == "last": target_val = components[-1].x
            elif anchor == "center": target_val = sum(c.x for c in components) / len(components)
            
            for c in components:
                if not c.locked: c.x = target_val

        return ActionResult(True, f"Aligned {len(refs)} components", refs)

    def distribute_evenly(
        self,
        refs: List[str],
        start_ref: Optional[str] = None,
        end_ref: Optional[str] = None,
        axis: Literal["x", "y", "auto"] = "auto",
    ) -> ActionResult:
        """
        Distribute components evenly between two points or outer extremes.
        """
        if len(refs) < 3:
            return ActionResult(False, "Need at least 3 components to distribute", [])

        components = []
        for r in refs:
            c = self.board.components.get(r)
            if not c: return ActionResult(False, f"Component {r} not found", [])
            components.append(c)

        # Determine start/end components (anchors)
        if start_ref:
            start_comp = self.board.components.get(start_ref)
            if not start_comp: return ActionResult(False, f"Start ref {start_ref} not found", [])
        else:
            # Find extreme based on axis guess
            if axis == "auto":
                # Guess axis based on spread
                x_spread = max(c.x for c in components) - min(c.x for c in components)
                y_spread = max(c.y for c in components) - min(c.y for c in components)
                axis = "x" if x_spread > y_spread else "y"
            
            # Sort by axis
            attr = "x" if axis == "x" else "y"
            components.sort(key=lambda c: getattr(c, attr))
            start_comp = components[0]
            end_comp = components[-1]

        if end_ref:
            end_comp = self.board.components.get(end_ref)
            if not end_comp: return ActionResult(False, f"End ref {end_ref} not found", [])

        # Calculate pitch
        count = len(components)
        if axis == "x":
            total_dist = end_comp.x - start_comp.x
            pitch = total_dist / (count - 1)
            start_pos = start_comp.x
            
            for i, comp in enumerate(components):
                if comp.reference in (start_comp.reference, end_comp.reference) and comp.locked:
                    continue
                if not comp.locked:
                    comp.x = start_pos + (i * pitch)
        else:
            total_dist = end_comp.y - start_comp.y
            pitch = total_dist / (count - 1)
            start_pos = start_comp.y
            
            for i, comp in enumerate(components):
                if comp.reference in (start_comp.reference, end_comp.reference) and comp.locked:
                    continue
                if not comp.locked:
                    comp.y = start_pos + (i * pitch)

        return ActionResult(True, f"Distributed {len(components)} components along {axis}", refs)

    def stack_components(
        self,
        refs: List[str],
        direction: Literal["up", "down", "left", "right"] = "down",
        spacing: float = 0.5,
        alignment: Literal["center", "left", "right", "top", "bottom"] = "center"
    ) -> ActionResult:
        """
        Stack components sequentially in a direction.
        Similar to place_next_to but for a list.
        """
        if len(refs) < 2:
            return ActionResult(False, "Need at least 2 components to stack", [])

        # Start with the first component as anchor
        anchor_ref = refs[0]
        modified = []

        # Determine side for place_next_to
        # If stacking DOWN, we place NEXT component on BOTTOM
        side_map = {
            "down": "bottom",
            "up": "top",
            "right": "right",
            "left": "left"
        }
        side = side_map.get(direction, "bottom")

        # Determine alignment
        # Vertical stack (up/down) -> align center/left/right
        # Horizontal stack (left/right) -> align center/top/bottom
        if direction in ("up", "down") and alignment not in ("center", "left", "right"):
            alignment = "center"
        if direction in ("left", "right") and alignment not in ("center", "top", "bottom"):
            alignment = "center"

        for i in range(1, len(refs)):
            target = refs[i]
            prev = refs[i-1]
            
            res = self.place_next_to(target, prev, side=side, clearance=spacing, align=alignment)
            if res.success:
                modified.append(target)
            else:
                return ActionResult(False, f"Stack failed at {target}: {res.message}", modified)

        return ActionResult(True, f"Stacked {len(refs)} components {direction}", modified)

    def group_components(self, refs: List[str], group_name: str) -> ActionResult:
        """Group components logically (store in properties)."""
        if not refs:
            return ActionResult(False, "No components to group", [])
            
        modified = []
        for r in refs:
            c = self.board.components.get(r)
            if c:
                c.properties["group"] = group_name
                modified.append(r)
                
        return ActionResult(True, f"Grouped {len(modified)} components into '{group_name}'", modified)

    def lock_components(self, refs: List[str], locked: bool = True) -> ActionResult:
        """Set locked state for components."""
        if not refs:
            return ActionResult(False, "No components specified", [])
            
        modified = []
        for r in refs:
            c = self.board.components.get(r)
            if c:
                c.locked = locked
                modified.append(r)
                
        state = "Locked" if locked else "Unlocked"
        return ActionResult(True, f"{state} {len(modified)} components", modified)

    def arrange_pattern(
        self,
        refs: List[str],
        pattern: Literal["grid", "row", "column", "circular"] = "grid",
        spacing: float = 2.0,
        cols: Optional[int] = None,
        radius: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ) -> ActionResult:
        """
        Arrange components in a pattern (grid, row, column, or circular).

        Args:
            refs: List of component references to arrange
            pattern: Arrangement pattern type
            spacing: Spacing between components in mm
            cols: Number of columns for grid pattern (auto-calculated if None)
            radius: Radius for circular pattern (auto-calculated if None)
            center: Center point for arrangement (uses centroid if None)
        """
        components = []
        for r in refs:
            c = self.board.components.get(r)
            if c and not c.locked:
                components.append(c)

        if not components:
            return ActionResult(False, "No moveable components found", [])

        # Calculate center if not provided
        if center is None:
            cx = sum(c.x for c in components) / len(components)
            cy = sum(c.y for c in components) / len(components)
            center = (cx, cy)

        modified = []

        if pattern == "grid":
            cols = cols or int(math.ceil(math.sqrt(len(components))))
            rows = int(math.ceil(len(components) / cols))

            for i, comp in enumerate(components):
                row, col = divmod(i, cols)
                comp.x = center[0] + (col - (cols - 1) / 2) * spacing
                comp.y = center[1] + (row - (rows - 1) / 2) * spacing
                modified.append(comp.reference)

        elif pattern == "row":
            for i, comp in enumerate(components):
                comp.x = center[0] + (i - (len(components) - 1) / 2) * spacing
                comp.y = center[1]
                modified.append(comp.reference)

        elif pattern == "column":
            for i, comp in enumerate(components):
                comp.x = center[0]
                comp.y = center[1] + (i - (len(components) - 1) / 2) * spacing
                modified.append(comp.reference)

        elif pattern == "circular":
            radius = radius or spacing * len(components) / (2 * math.pi)
            for i, comp in enumerate(components):
                angle = 2 * math.pi * i / len(components)
                comp.x = center[0] + radius * math.cos(angle)
                comp.y = center[1] + radius * math.sin(angle)
                modified.append(comp.reference)

        return ActionResult(True, f"Arranged {len(modified)} components in {pattern}", modified)

    def cluster_around(
        self,
        anchor_ref: str,
        target_refs: List[str],
        side: Literal["top", "bottom", "left", "right", "nearest"] = "nearest",
        clearance: float = 0.5,
    ) -> ActionResult:
        """
        Cluster components around an anchor on a specified side.

        Args:
            anchor_ref: Reference component to cluster around
            target_refs: List of components to cluster
            side: Which side to cluster on ("nearest" auto-selects based on current positions)
            clearance: Gap between anchor and clustered components
        """
        anchor = self.board.components.get(anchor_ref)
        if not anchor:
            return ActionResult(False, f"Anchor {anchor_ref} not found", [])

        targets = []
        for r in target_refs:
            c = self.board.components.get(r)
            if c and not c.locked:
                targets.append(c)

        if not targets:
            return ActionResult(False, "No moveable target components", [])

        # Determine side if "nearest"
        if side == "nearest":
            avg_x = sum(t.x for t in targets) / len(targets)
            avg_y = sum(t.y for t in targets) / len(targets)
            dx = avg_x - anchor.x
            dy = avg_y - anchor.y

            if abs(dx) > abs(dy):
                side = "right" if dx > 0 else "left"
            else:
                side = "bottom" if dy > 0 else "top"

        # Get anchor dimensions
        aw, ah = self._get_dims(anchor)

        # Arrange targets in a compact row/column on the chosen side
        modified = []

        # Calculate total width/height of targets for layout
        target_dims = [self._get_dims(t) for t in targets]

        if side in ("left", "right"):
            # Arrange vertically along the side
            total_height = sum(d[1] for d in target_dims) + clearance * (len(targets) - 1)
            start_y = anchor.y - total_height / 2

            base_x = anchor.x + (aw / 2 + clearance) if side == "right" else anchor.x - (aw / 2 + clearance)

            current_y = start_y
            for i, t in enumerate(targets):
                tw, th = target_dims[i]
                t.x = base_x + (tw / 2 if side == "right" else -tw / 2)
                t.y = current_y + th / 2
                current_y += th + clearance
                modified.append(t.reference)

        else:  # top or bottom
            # Arrange horizontally along the side
            total_width = sum(d[0] for d in target_dims) + clearance * (len(targets) - 1)
            start_x = anchor.x - total_width / 2

            base_y = anchor.y - (ah / 2 + clearance) if side == "top" else anchor.y + (ah / 2 + clearance)

            current_x = start_x
            for i, t in enumerate(targets):
                tw, th = target_dims[i]
                t.x = current_x + tw / 2
                t.y = base_y + (-th / 2 if side == "top" else th / 2)
                current_x += tw + clearance
                modified.append(t.reference)

        return ActionResult(True, f"Clustered {len(modified)} components {side} of {anchor_ref}", modified)

    def _get_dims(self, comp: Component) -> Tuple[float, float]:
        """Get effective width/height considering 90 degree rotation."""
        rot = comp.rotation % 180
        # If rotated 90 or 270, swap w/h
        if 45 < rot < 135:
            return comp.height, comp.width
        return comp.width, comp.height