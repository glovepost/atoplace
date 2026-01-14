"""
Vision Context Generator

Generates visual representations of board regions for multimodal LLM understanding.
Produces annotated SVG/PNG images with dimensions and component labels.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import math

from ...board.abstraction import Board, Component


@dataclass
class ViewportSpec:
    """Specification for a viewport region."""
    center_x: float
    center_y: float
    width: float
    height: float
    padding: float = 5.0


@dataclass
class AnnotatedImage:
    """Result of vision context generation."""
    svg_content: str
    viewport: ViewportSpec
    component_count: int
    annotations: List[str]


class VisionContext:
    """
    Generates annotated visual representations for multimodal LLMs.

    Creates lightweight SVG images of board regions with:
    - Component outlines and labels
    - Dimension arrows showing gap sizes
    - Ratsnest connections
    - Alignment guides
    """

    # Color palette
    COLORS = {
        "background": "#1a1a2e",
        "board": "#16213e",
        "component": "#4a6fa5",
        "component_stroke": "#ffffff",
        "pad": "#ffd700",
        "dimension": "#ff6b6b",
        "ratsnest": "#3a7ca5",
        "annotation": "#ffffff",
        "grid": "#2a2a4e",
    }

    def __init__(self, board: Board):
        self.board = board

    def render_region(
        self,
        refs: List[str],
        padding: float = 5.0,
        show_dimensions: bool = True,
        show_ratsnest: bool = True,
        image_width: int = 400
    ) -> AnnotatedImage:
        """
        Render a specific region defined by component references.

        Args:
            refs: List of component references to include
            padding: Padding around the region in mm
            show_dimensions: Whether to show gap dimensions
            show_ratsnest: Whether to show net connections
            image_width: Output image width in pixels

        Returns:
            AnnotatedImage with SVG content
        """
        # Get components
        components = [self.board.components[r] for r in refs if r in self.board.components]

        if not components:
            return self._empty_image()

        # Calculate bounding box
        min_x, min_y, max_x, max_y = self._get_bounding_box(components)

        # Apply padding
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        viewport = ViewportSpec(
            center_x=(min_x + max_x) / 2,
            center_y=(min_y + max_y) / 2,
            width=max_x - min_x,
            height=max_y - min_y,
            padding=padding
        )

        # Calculate scale
        aspect = viewport.height / viewport.width if viewport.width > 0 else 1
        image_height = int(image_width * aspect)
        scale = image_width / viewport.width if viewport.width > 0 else 1

        svg_padding = 30  # Padding for labels
        svg_width = image_width + 2 * svg_padding
        svg_height = image_height + 2 * svg_padding

        def tx(x: float) -> float:
            return svg_padding + (x - min_x) * scale

        def ty(y: float) -> float:
            return svg_padding + (max_y - y) * scale

        def ts(size: float) -> float:
            return size * scale

        # Start SVG
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_width}" height="{svg_height}" '
            f'viewBox="0 0 {svg_width} {svg_height}">'
        ]

        # Background
        svg_parts.append(
            f'<rect width="100%" height="100%" fill="{self.COLORS["background"]}"/>'
        )

        # Board region
        svg_parts.append(
            f'<rect x="{svg_padding}" y="{svg_padding}" '
            f'width="{image_width}" height="{image_height}" '
            f'fill="{self.COLORS["board"]}" stroke="#0f3460" stroke-width="1"/>'
        )

        # Grid lines (1mm spacing)
        grid_spacing = 1.0
        if ts(grid_spacing) > 5:  # Only show if visible
            x = math.ceil(min_x / grid_spacing) * grid_spacing
            while x <= max_x:
                svg_parts.append(
                    f'<line x1="{tx(x)}" y1="{ty(min_y)}" '
                    f'x2="{tx(x)}" y2="{ty(max_y)}" '
                    f'stroke="{self.COLORS["grid"]}" stroke-width="0.5"/>'
                )
                x += grid_spacing

            y = math.ceil(min_y / grid_spacing) * grid_spacing
            while y <= max_y:
                svg_parts.append(
                    f'<line x1="{tx(min_x)}" y1="{ty(y)}" '
                    f'x2="{tx(max_x)}" y2="{ty(y)}" '
                    f'stroke="{self.COLORS["grid"]}" stroke-width="0.5"/>'
                )
                y += grid_spacing

        annotations = []

        # Render ratsnest
        if show_ratsnest:
            self._render_ratsnest(svg_parts, components, tx, ty)

        # Render components
        for comp in components:
            self._render_component(svg_parts, comp, tx, ty, ts)

        # Render dimension annotations
        if show_dimensions and len(components) >= 2:
            self._render_dimensions(svg_parts, components, tx, ty, ts, annotations)

        svg_parts.append('</svg>')

        return AnnotatedImage(
            svg_content='\n'.join(svg_parts),
            viewport=viewport,
            component_count=len(components),
            annotations=annotations
        )

    def render_full_board(self, image_width: int = 800) -> AnnotatedImage:
        """Render full board overview."""
        all_refs = list(self.board.components.keys())
        return self.render_region(all_refs, padding=10.0, image_width=image_width)

    def save_svg(self, image: AnnotatedImage, path: Path):
        """Save annotated image to SVG file."""
        path.write_text(image.svg_content)

    def _get_bounding_box(self, components: List[Component]) -> Tuple[float, float, float, float]:
        """Calculate bounding box of components."""
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for comp in components:
            w, h = self._get_rotated_size(comp)
            min_x = min(min_x, comp.x - w/2)
            min_y = min(min_y, comp.y - h/2)
            max_x = max(max_x, comp.x + w/2)
            max_y = max(max_y, comp.y + h/2)

        return min_x, min_y, max_x, max_y

    def _get_rotated_size(self, comp: Component) -> Tuple[float, float]:
        """Get component AABB size accounting for arbitrary rotation."""
        # Calculate proper AABB dimensions for arbitrary rotations
        bbox = comp.get_bounding_box()
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])  # (width, height)

    def _render_component(self, svg_parts: List[str], comp: Component,
                          tx, ty, ts):
        """Render a single component."""
        cx, cy = tx(comp.x), ty(comp.y)
        w, h = self._get_rotated_size(comp)
        hw, hh = ts(w/2), ts(h/2)

        # Component body (rotated rectangle)
        svg_parts.append(
            f'<g transform="rotate({-comp.rotation} {cx} {cy})">'
        )

        # Body
        svg_parts.append(
            f'<rect x="{cx - hw}" y="{cy - hh}" '
            f'width="{ts(w)}" height="{ts(h)}" '
            f'fill="{self.COLORS["component"]}" fill-opacity="0.7" '
            f'stroke="{self.COLORS["component_stroke"]}" stroke-width="1"/>'
        )

        # Pads
        for pad in comp.pads:
            # Transform pad position (accounting for component rotation)
            pad_x, pad_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            px, py = tx(pad_x), ty(pad_y)
            pw = max(ts(pad.width/2), 2)
            ph = max(ts(pad.height/2), 2)

            svg_parts.append(
                f'<rect x="{px - pw}" y="{py - ph}" '
                f'width="{pw*2}" height="{ph*2}" '
                f'fill="{self.COLORS["pad"]}" rx="1"/>'
            )

        svg_parts.append('</g>')

        # Reference label (outside rotation group)
        font_size = max(8, min(12, ts(min(w, h) * 0.4)))
        svg_parts.append(
            f'<text x="{cx}" y="{cy - hh - 5}" '
            f'font-family="monospace" font-size="{font_size}" '
            f'fill="{self.COLORS["annotation"]}" text-anchor="middle">'
            f'{comp.reference}</text>'
        )

    def _render_ratsnest(self, svg_parts: List[str], components: List[Component],
                         tx, ty):
        """Render ratsnest connections between components."""
        comp_refs = {c.reference for c in components}

        # Build pad positions for relevant nets
        for comp in components:
            for pad in comp.pads:
                if not pad.net:
                    continue

                net = self.board.nets.get(pad.net)
                if not net:
                    continue

                # Find other pads on same net within our component set
                for other_comp_ref, other_pad_num in net.connections:
                    if (other_comp_ref and
                        other_comp_ref != comp.reference and
                        other_comp_ref in comp_refs):

                        other_comp = self.board.components.get(other_comp_ref)
                        if other_comp:
                            # Find the pad on the other component
                            other_pad = other_comp.get_pad_by_number(other_pad_num)
                            if other_pad:
                                pad1_x, pad1_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                                pad2_x, pad2_y = other_pad.absolute_position(other_comp.x, other_comp.y, other_comp.rotation)
                                x1 = tx(pad1_x)
                                y1 = ty(pad1_y)
                                x2 = tx(pad2_x)
                                y2 = ty(pad2_y)

                                svg_parts.append(
                                    f'<line x1="{x1}" y1="{y1}" '
                                    f'x2="{x2}" y2="{y2}" '
                                    f'stroke="{self.COLORS["ratsnest"]}" '
                                    f'stroke-width="0.5" opacity="0.5"/>'
                                )

    def _render_dimensions(self, svg_parts: List[str], components: List[Component],
                           tx, ty, ts, annotations: List[str]):
        """Render dimension annotations between components."""
        if len(components) < 2:
            return

        # Find closest pair
        min_gap = float('inf')
        closest_pair = None

        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
                gap = self._calculate_gap(c1, c2)
                if gap < min_gap:
                    min_gap = gap
                    closest_pair = (c1, c2)

        if closest_pair and min_gap < float('inf'):
            c1, c2 = closest_pair

            # Draw dimension line
            x1, y1 = tx(c1.x), ty(c1.y)
            x2, y2 = tx(c2.x), ty(c2.y)

            # Midpoint for label
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2

            # Dimension line
            svg_parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{self.COLORS["dimension"]}" stroke-width="1" '
                f'stroke-dasharray="4,2"/>'
            )

            # End markers
            svg_parts.append(
                f'<circle cx="{x1}" cy="{y1}" r="3" fill="{self.COLORS["dimension"]}"/>'
            )
            svg_parts.append(
                f'<circle cx="{x2}" cy="{y2}" r="3" fill="{self.COLORS["dimension"]}"/>'
            )

            # Gap label with background
            gap_text = f"{min_gap:.2f}mm"
            svg_parts.append(
                f'<rect x="{mx - 25}" y="{my - 8}" width="50" height="16" '
                f'fill="{self.COLORS["background"]}" rx="2"/>'
            )
            svg_parts.append(
                f'<text x="{mx}" y="{my + 4}" '
                f'font-family="monospace" font-size="10" '
                f'fill="{self.COLORS["dimension"]}" text-anchor="middle">'
                f'{gap_text}</text>'
            )

            annotations.append(f"Gap between {c1.reference} and {c2.reference}: {min_gap:.2f}mm")

    def _calculate_gap(self, c1: Component, c2: Component) -> float:
        """Calculate edge-to-edge gap between two components."""
        w1, h1 = self._get_rotated_size(c1)
        w2, h2 = self._get_rotated_size(c2)

        # Calculate separation in X and Y
        dx = abs(c1.x - c2.x) - (w1/2 + w2/2)
        dy = abs(c1.y - c2.y) - (h1/2 + h2/2)

        # If they overlap in both dimensions, gap is negative
        if dx < 0 and dy < 0:
            return max(dx, dy)  # Less negative = smaller overlap

        # If they overlap in one dimension, gap is the other
        if dx < 0:
            return dy
        if dy < 0:
            return dx

        # No overlap: gap is the shorter of the two distances
        return min(dx, dy)

    def _empty_image(self) -> AnnotatedImage:
        """Return empty placeholder image."""
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
            <rect width="100%" height="100%" fill="#1a1a2e"/>
            <text x="100" y="50" fill="#888" text-anchor="middle">No components</text>
        </svg>'''
        return AnnotatedImage(
            svg_content=svg,
            viewport=ViewportSpec(0, 0, 0, 0),
            component_count=0,
            annotations=[]
        )
