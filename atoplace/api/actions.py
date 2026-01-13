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
        # TODO: Use precise bounding box from visualizer logic if available
        # For now, simple w/h check
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

    def _get_dims(self, comp: Component) -> Tuple[float, float]:
        """Get effective width/height considering 90 degree rotation."""
        rot = comp.rotation % 180
        # If rotated 90 or 270, swap w/h
        if 45 < rot < 135:
            return comp.height, comp.width
        return comp.width, comp.height
