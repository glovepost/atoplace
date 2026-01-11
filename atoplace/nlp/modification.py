"""
Modification Handler

Handles natural language modification requests for placed designs.
Supports move, rotate, swap, and other layout adjustments.
"""

from .constraint_parser import ModificationHandler

# Re-export ModificationHandler from constraint_parser
__all__ = ["ModificationHandler"]
