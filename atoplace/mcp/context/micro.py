"""
Microscope Context Generator

Generates high-precision local context for LLM spatial reasoning.
Includes logic to calculate exact gaps between components.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import json
import math

from ...board.abstraction import Board, Component

@dataclass
class Viewport:
    center: Tuple[float, float]
    size: Tuple[float, float]
    units: str = "mm"

@dataclass
class ObjectView:
    ref: str
    type: str
    layer: str
    location: Tuple[float, float]
    rotation: float
    bbox: Dict[str, Tuple[float, float]]
    pads: List[Dict] = field(default_factory=list)

@dataclass
class GapView:
    between: List[str]
    distance: float
    vector: Tuple[float, float]

@dataclass
class MicroscopeData:
    viewport: Viewport
    grid_aligned: bool
    objects: List[ObjectView]
    gaps: List[GapView]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class Microscope:
    """The 'eyes' of the LLM - providing precision local data."""

    def __init__(self, board: Board):
        self.board = board

    def inspect_region(self, refs: List[str], padding: float = 5.0) -> MicroscopeData:
        """
        Inspect a specific region defined by a list of components.
        Returns precise JSON geometry and gap analysis.
        """
        components = []
        for r in refs:
            c = self.board.components.get(r)
            if c: components.append(c)
            
        if not components:
            # Fallback: return center of board if empty
            return self._empty_view()

        # Calculate bounding box of selection
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        object_views = []
        
        for comp in components:
            bbox = self._get_bbox(comp)
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
            
            # Create object view
            obj = ObjectView(
                ref=comp.reference,
                type="Component", # TODO: Infer type (IC, R, C)
                layer=str(comp.layer),
                location=(comp.x, comp.y),
                rotation=comp.rotation,
                bbox={"min": (bbox[0], bbox[1]), "max": (bbox[2], bbox[3])},
                pads=[{"num": p.number, "pos": (comp.x + p.x, comp.y + p.y), "net": p.net} for p in comp.pads] # Simplified
            )
            object_views.append(obj)

        # Apply padding to viewport
        vp_min_x = min_x - padding
        vp_min_y = min_y - padding
        vp_max_x = max_x + padding
        vp_max_y = max_y + padding
        
        vp_width = vp_max_x - vp_min_x
        vp_height = vp_max_y - vp_min_y
        vp_center = (vp_min_x + vp_width/2, vp_min_y + vp_height/2)

        # Calculate gaps between requested components
        gaps = self._calculate_gaps(components)

        return MicroscopeData(
            viewport=Viewport(center=vp_center, size=(vp_width, vp_height)),
            grid_aligned=self._check_grid_alignment(components),
            objects=object_views,
            gaps=gaps
        )

    def _get_bbox(self, comp: Component) -> Tuple[float, float, float, float]:
        """Get precise AABB."""
        # Simple implementation - should ideally match visualizer logic
        w, h = comp.width, comp.height
        if 45 < (comp.rotation % 180) < 135:
            w, h = h, w
        
        return (comp.x - w/2, comp.y - h/2, comp.x + w/2, comp.y + h/2)

    def _calculate_gaps(self, components: List[Component]) -> List[GapView]:
        """Calculate gaps between all pairs of components."""
        gaps = []
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                c1 = components[i]
                c2 = components[j]
                
                bb1 = self._get_bbox(c1)
                bb2 = self._get_bbox(c2)
                
                # Calculate signed distance in X and Y
                # dist > 0 means gap, dist < 0 means overlap
                # X gap: max(min1, min2) - min(max1, max2) ?? No
                # X gap: left of rightmost - right of leftmost ??
                # Let's use center distance minus half-widths
                
                dx = abs(c1.x - c2.x) - (c1.width/2 + c2.width/2) # Rough approx ignoring rotation for now
                # Correct AABB gap logic:
                gap_x = max(bb1[0], bb2[0]) - min(bb1[2], bb2[2]) # Overlap amount (negative if gap?)
                # Actually we want distance between edges.
                
                left = sorted([c1, c2], key=lambda c: c.x)
                l_comp, r_comp = left[0], left[1]
                l_bb = self._get_bbox(l_comp)
                r_bb = self._get_bbox(r_comp)
                
                real_gap_x = r_bb[0] - l_bb[2]
                
                top = sorted([c1, c2], key=lambda c: c.y)
                t_comp, b_comp = top[0], top[1]
                t_bb = self._get_bbox(t_comp)
                b_bb = self._get_bbox(b_comp)
                
                real_gap_y = b_bb[1] - t_bb[3]
                
                # We report the MAX gap to indicate separation
                # If both are negative, they overlap
                
                dist = max(real_gap_x, real_gap_y)
                
                # Vector from c1 to c2
                vx = c2.x - c1.x
                vy = c2.y - c1.y
                mag = math.sqrt(vx*vx + vy*vy)
                if mag > 0:
                    vx, vy = vx/mag, vy/mag
                
                gaps.append(GapView(
                    between=[c1.reference, c2.reference],
                    distance=dist,
                    vector=(vx, vy)
                ))
        return gaps

    def _check_grid_alignment(self, components: List[Component], grid: float = 0.5) -> bool:
        """Check if all components are on grid."""
        for c in components:
            if abs(c.x % grid) > 0.001 or abs(c.y % grid) > 0.001:
                return False
        return True

    def _empty_view(self) -> MicroscopeData:
        return MicroscopeData(
            viewport=Viewport((0,0), (0,0)), 
            grid_aligned=True, 
            objects=[], 
            gaps=[]
        )
