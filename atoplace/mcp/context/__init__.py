"""
AtoPlace MCP Context Generators

Provides multi-level context for LLM spatial reasoning:
- Macro: Executive summary, semantic grid, module map
- Micro: Precision local geometry and gap analysis
- Vision: Visual representations for multimodal models
"""

from .macro import BoardSummary, MacroContext, ModuleMap, SemanticGrid
from .micro import GapView, Microscope, MicroscopeData, ObjectView, Viewport
from .vision import VisionContext

__all__ = [
    # Micro
    "Microscope",
    "MicroscopeData",
    "ObjectView",
    "GapView",
    "Viewport",
    # Macro
    "MacroContext",
    "BoardSummary",
    "SemanticGrid",
    "ModuleMap",
    # Vision
    "VisionContext",
]
