"""
AtoPlace - AI-Powered PCB Placement and Routing Tool

An LLM-augmented PCB layout tool that accelerates professional EE workflows
by automating placement optimization and routing for medium-complexity boards.
"""

__version__ = "0.1.0"
__author__ = "AtoPlace Team"

from .board.abstraction import Board, Component, Net
from .validation.confidence import ConfidenceReport, ConfidenceScorer
from .dfm.profiles import DFMProfile, get_profile

__all__ = [
    "Board",
    "Component",
    "Net",
    "ConfidenceReport",
    "ConfidenceScorer",
    "DFMProfile",
    "get_profile",
]
