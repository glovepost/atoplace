"""
AtoPlace Core API

High-level API for LLM-driven PCB layout manipulation.

Modules:
- actions: Atomic geometric operations (move, align, place_next_to)
- session: State management with undo/redo support
"""

from .actions import LayoutActions, ActionResult
from .session import Session

__all__ = [
    "LayoutActions",
    "ActionResult",
    "Session",
]
