"""Board inspection and validation operations.

This module contains shared inspection logic used by both the MCP server
and RPC worker to avoid code duplication.

All operations take a Board instance and return structured data that can
be easily consumed by both RPC and MCP interfaces.
"""

from typing import List, Dict, Optional, Tuple
from ..board.abstraction import Board, Component


class BoardInspector:
    """Shared board inspection operations.

    This class provides inspection and validation methods that are used
    by both the MCP server and RPC worker to avoid duplicating logic.
    """

    def __init__(self, board: Board):
        """Initialize inspector with a board.

        Args:
            board: Board instance to inspect
        """
        self.board = board

    def check_overlaps(
        self,
        refs: Optional[List[str]] = None,
        include_pads: bool = True
    ) -> List[Dict[str, any]]:
        """Check for component overlaps.

        Uses AABB (axis-aligned bounding box) collision detection with proper
        rotation and optional pad extent handling.

        Args:
            refs: Optional list of component refs to check. If None, checks all components.
            include_pads: If True (default), use bounding boxes that include pad extents.
                         This catches overlaps where pads protrude beyond the component body
                         (e.g., edge-mounted connectors, irregular footprints).

        Returns:
            List of overlap dictionaries with keys:
            - refs: [ref1, ref2] - pair of overlapping component references
            - overlap_x: float - overlap distance in X axis (mm)
            - overlap_y: float - overlap distance in Y axis (mm)
        """
        # Select components to check
        components = []
        if refs:
            components = [self.board.components[r] for r in refs if r in self.board.components]
        else:
            components = list(self.board.components.values())

        # Check all pairs for overlaps using proper bounding boxes
        overlaps = []
        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
                # Get proper bounding boxes (with rotation and optionally pads)
                if include_pads:
                    bbox1 = c1.get_bounding_box_with_pads()
                    bbox2 = c2.get_bounding_box_with_pads()
                else:
                    bbox1 = c1.get_bounding_box()
                    bbox2 = c2.get_bounding_box()

                # Unpack bounding boxes: (min_x, min_y, max_x, max_y)
                min_x1, min_y1, max_x1, max_y1 = bbox1
                min_x2, min_y2, max_x2, max_y2 = bbox2

                # AABB collision: boxes overlap if they intersect on both axes
                overlap_x = min(max_x1, max_x2) - max(min_x1, min_x2)
                overlap_y = min(max_y1, max_y2) - max(min_y1, min_y2)

                if overlap_x > 0 and overlap_y > 0:
                    overlaps.append({
                        "refs": [c1.reference, c2.reference],
                        "overlap_x": round(overlap_x, 3),
                        "overlap_y": round(overlap_y, 3),
                    })

        return overlaps

    def find_components(
        self,
        query: str,
        filter_by: str = "ref"
    ) -> List[Dict[str, any]]:
        """Find components matching a search query.

        Performs case-insensitive substring matching on the specified field.

        Args:
            query: Search string to match
            filter_by: Field to search - one of "ref", "value", or "footprint"

        Returns:
            List of component dictionaries with keys:
            - ref: Component reference designator
            - value: Component value
            - footprint: Footprint name
            - x, y: Component position (mm)
            - rotation: Component rotation (degrees)

        Raises:
            ValueError: If filter_by is not a valid field name
        """
        filter_by = filter_by.lower()
        valid_filters = {"ref", "value", "footprint"}
        if filter_by not in valid_filters:
            raise ValueError(
                f"Invalid filter '{filter_by}'. Must be one of: {', '.join(valid_filters)}"
            )

        matches = []
        for ref, comp in self.board.components.items():
            # Get the field to search
            search_value = ""
            if filter_by == "ref":
                search_value = ref
            elif filter_by == "value":
                search_value = comp.value
            elif filter_by == "footprint":
                search_value = comp.footprint

            # Case-insensitive substring match
            if query.lower() in search_value.lower():
                matches.append({
                    "ref": ref,
                    "value": comp.value,
                    "footprint": comp.footprint,
                    "x": round(comp.x, 3),
                    "y": round(comp.y, 3),
                    "rotation": comp.rotation,
                })

        return matches

    def get_component_info(self, ref: str) -> Optional[Dict[str, any]]:
        """Get detailed information about a component.

        Args:
            ref: Component reference designator

        Returns:
            Dictionary with component details, or None if component not found
        """
        if ref not in self.board.components:
            return None

        comp = self.board.components[ref]
        return {
            "ref": ref,
            "value": comp.value,
            "footprint": comp.footprint,
            "x": round(comp.x, 3),
            "y": round(comp.y, 3),
            "rotation": comp.rotation,
            "width": round(comp.width, 3),
            "height": round(comp.height, 3),
            "layer": comp.layer.name,
            "locked": comp.locked,
            "pad_count": len(comp.pads),
        }

    def get_board_stats(self) -> Dict[str, any]:
        """Get board statistics summary.

        Returns:
            Dictionary with board statistics:
            - component_count: Total number of components
            - net_count: Total number of nets
            - layer_count: Number of copper layers
            - board_area: Board area in mm²
        """
        return {
            "component_count": len(self.board.components),
            "net_count": len(self.board.nets),
            "layer_count": self.board.layer_count,
            "board_area": round(self.board.calculate_board_area(), 2)
                          if self.board.outline else None,
        }
