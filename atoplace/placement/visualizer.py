"""Placement visualization for debugging force-directed refinement.

Generates interactive HTML reports showing:
- Component bodies with reference designations
- Pad positions and shapes
- Bounding boxes
- Module group membership (color-coded)
- Force vectors during refinement
- Step-by-step algorithm playback
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlacementFrame:
    """A single frame in the placement visualization."""
    index: int
    label: str
    iteration: int = 0
    phase: str = ""  # "initial", "refinement", "legalization", "final"

    # Component states: ref -> (x, y, rotation, width, height)
    components: Dict[str, Tuple[float, float, float, float, float]] = field(default_factory=dict)

    # Pad positions: ref -> list of (x, y, width, height, net_name)
    pads: Dict[str, List[Tuple[float, float, float, float, str]]] = field(default_factory=dict)

    # Module groups: ref -> module_type
    modules: Dict[str, str] = field(default_factory=dict)

    # Forces for this frame: ref -> (fx, fy, force_type)
    forces: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)

    # Overlapping pairs
    overlaps: List[Tuple[str, str]] = field(default_factory=list)

    # Net connections for visualization: list of (ref1, ref2, net_name)
    connections: List[Tuple[str, str, str]] = field(default_factory=list)

    # Component movement from initial: ref -> (dx, dy, total_distance)
    movement: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    # Statistics
    energy: float = 0.0
    max_move: float = 0.0
    overlap_count: int = 0
    total_wire_length: float = 0.0  # Sum of connection distances


# Module type to color mapping
MODULE_COLORS = {
    "power_supply": "#e74c3c",      # Red
    "microcontroller": "#3498db",   # Blue
    "rf_frontend": "#9b59b6",       # Purple
    "sensor": "#2ecc71",            # Green
    "connector": "#f39c12",         # Orange
    "crystal": "#1abc9c",           # Teal
    "led": "#e91e63",               # Pink
    "memory": "#00bcd4",            # Cyan
    "analog": "#ff5722",            # Deep Orange
    "digital": "#607d8b",           # Blue Grey
    "esd_protection": "#795548",    # Brown
    "level_shifter": "#9c27b0",     # Deep Purple
    "default": "#95a5a6",           # Grey
}

FORCE_COLORS = {
    "repulsion": "#e74c3c",    # Red
    "attraction": "#2ecc71",   # Green
    "boundary": "#3498db",     # Blue
    "constraint": "#f39c12",   # Orange
    "alignment": "#9b59b6",    # Purple
}


class PlacementVisualizer:
    """Visualizer for placement algorithm debugging.

    Captures frames during force-directed refinement and legalization,
    then exports to interactive HTML for step-by-step analysis.

    Key features (based on @seveibar's autorouter lessons):
    - Overlay mode to compare initial vs current state
    - Net connection visualization (wire length optimization)
    - Movement trails to show component path history
    - Energy/iteration graph for convergence analysis
    - Force vector visualization with magnitude scaling
    """

    def __init__(self, board=None, grid_spacing: float = 1.27):
        """
        Args:
            board: Optional Board instance for outline extraction
            grid_spacing: Grid spacing in mm for visualization (default 1.27mm = 50mil)
        """
        self.board = board
        self.frames: List[PlacementFrame] = []
        self.frame_count = 0
        self.grid_spacing = grid_spacing

        # Board bounds for SVG viewport
        self.min_x = 0
        self.max_x = 100
        self.min_y = 0
        self.max_y = 100

        # Track initial positions for movement visualization
        self.initial_positions: Dict[str, Tuple[float, float]] = {}

        # Pre-compute net connections for wire length visualization
        self.net_connections: List[Tuple[str, str, str]] = []

        if board:
            self._extract_bounds()
            self._extract_initial_positions()
            self._extract_net_connections()

    def _extract_bounds(self):
        """Extract board bounds from outline or components."""
        if self.board.outline:
            bbox = self.board.outline.get_bounding_box()
            if bbox:
                self.min_x, self.min_y, self.max_x, self.max_y = bbox
                # Add margin
                margin = 5
                self.min_x -= margin
                self.min_y -= margin
                self.max_x += margin
                self.max_y += margin
                return

        # Fall back to component bounds
        if self.board.components:
            xs = []
            ys = []
            for comp in self.board.components.values():
                bbox = comp.get_bounding_box()
                xs.extend([bbox[0], bbox[2]])
                ys.extend([bbox[1], bbox[3]])
            if xs and ys:
                margin = 10
                self.min_x = min(xs) - margin
                self.max_x = max(xs) + margin
                self.min_y = min(ys) - margin
                self.max_y = max(ys) + margin

    def _extract_initial_positions(self):
        """Store initial component positions for movement tracking."""
        for ref, comp in self.board.components.items():
            if not comp.dnp:
                self.initial_positions[ref] = (comp.x, comp.y)

    def _extract_net_connections(self):
        """Extract component-to-component connections via shared nets.

        Creates a list of (ref1, ref2, net_name) for visualization.
        Only includes one connection per net pair (minimum spanning tree-like).
        """
        # Build net -> component refs mapping
        net_to_refs: Dict[str, Set[str]] = {}

        for ref, comp in self.board.components.items():
            if comp.dnp:
                continue
            for pad in comp.pads:
                if pad.net:
                    net_to_refs.setdefault(pad.net, set()).add(ref)

        # For each net, create connections between components
        # Use simple chain: connect sequential components
        for net_name, refs in net_to_refs.items():
            if len(refs) < 2:
                continue

            ref_list = sorted(refs)  # Deterministic order
            for i in range(len(ref_list) - 1):
                self.net_connections.append((ref_list[i], ref_list[i + 1], net_name))

    def _calculate_wire_length(
        self, components: Dict[str, Tuple[float, float, float, float, float]]
    ) -> float:
        """Calculate total wire length based on component positions."""
        total = 0.0
        for ref1, ref2, _ in self.net_connections:
            if ref1 in components and ref2 in components:
                x1, y1 = components[ref1][:2]
                x2, y2 = components[ref2][:2]
                total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return total

    def _calculate_movements(
        self, components: Dict[str, Tuple[float, float, float, float, float]]
    ) -> Dict[str, Tuple[float, float, float]]:
        """Calculate movement from initial position for each component."""
        movements = {}
        for ref, (x, y, _, _, _) in components.items():
            if ref in self.initial_positions:
                ix, iy = self.initial_positions[ref]
                dx, dy = x - ix, y - iy
                dist = math.sqrt(dx * dx + dy * dy)
                movements[ref] = (dx, dy, dist)
        return movements

    def capture_frame(
        self,
        label: str,
        iteration: int = 0,
        phase: str = "refinement",
        components: Dict[str, Tuple[float, float, float, float, float]] = None,
        pads: Dict[str, List[Tuple[float, float, float, float, str]]] = None,
        modules: Dict[str, str] = None,
        forces: Dict[str, List[Tuple[float, float, str]]] = None,
        overlaps: List[Tuple[str, str]] = None,
        energy: float = 0.0,
        max_move: float = 0.0,
        overlap_count: int = 0,
    ):
        """Capture a visualization frame.

        Args:
            label: Frame description (e.g., "Iteration 50")
            iteration: Current iteration number
            phase: Algorithm phase
            components: Component states {ref: (x, y, rotation, width, height)}
            pads: Pad positions {ref: [(x, y, w, h, net), ...]}
            modules: Module assignments {ref: module_type}
            forces: Force vectors {ref: [(fx, fy, type), ...]}
            overlaps: List of overlapping component pairs
            energy: System energy
            max_move: Maximum component movement
            overlap_count: Number of overlaps
        """
        components = components or {}

        # Calculate additional metrics
        wire_length = self._calculate_wire_length(components) if components else 0.0
        movements = self._calculate_movements(components) if components else {}

        frame = PlacementFrame(
            index=self.frame_count,
            label=label,
            iteration=iteration,
            phase=phase,
            components=components,
            pads=pads or {},
            modules=modules or {},
            forces=forces or {},
            overlaps=overlaps or [],
            connections=self.net_connections,  # Include net connections
            movement=movements,
            energy=energy,
            max_move=max_move,
            overlap_count=overlap_count,
            total_wire_length=wire_length,
        )
        self.frames.append(frame)
        self.frame_count += 1

        logger.debug(f"Captured placement frame {self.frame_count}: {label} (wire_length={wire_length:.1f}mm)")

    def capture_from_board(
        self,
        label: str,
        iteration: int = 0,
        phase: str = "refinement",
        modules: Dict[str, str] = None,
        forces: Dict[str, List[Tuple[float, float, str]]] = None,
        energy: float = 0.0,
        max_move: float = 0.0,
    ):
        """Capture frame directly from board state.

        Convenience method that extracts component and pad data from self.board.
        """
        if not self.board:
            logger.warning("No board set for visualizer")
            return

        components = {}
        pads = {}
        overlaps = []

        for ref, comp in self.board.components.items():
            if comp.dnp:
                continue
            components[ref] = (
                comp.x,
                comp.y,
                comp.rotation,
                comp.width,
                comp.height,
            )

            # Extract pads
            pad_list = []
            for pad in comp.pads:
                # Transform pad to board coordinates
                px, py = self._transform_pad(comp, pad)
                pad_list.append((
                    px,
                    py,
                    pad.width,
                    pad.height,
                    pad.net or "",
                ))
            pads[ref] = pad_list

        # Find overlaps (format is (ref1, ref2, distance))
        overlap_pairs = self.board.find_overlaps(clearance=0.1)
        overlaps = [(pair[0], pair[1]) for pair in overlap_pairs]

        self.capture_frame(
            label=label,
            iteration=iteration,
            phase=phase,
            components=components,
            pads=pads,
            modules=modules or {},
            forces=forces or {},
            overlaps=overlaps,
            energy=energy,
            max_move=max_move,
            overlap_count=len(overlaps),
        )

    def _transform_pad(self, comp, pad) -> Tuple[float, float]:
        """Transform pad position from component-relative to board coordinates."""
        px, py = pad.x, pad.y

        if comp.rotation != 0:
            rad = math.radians(comp.rotation)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            px, py = (
                px * cos_r - py * sin_r,
                px * sin_r + py * cos_r,
            )

        return (comp.x + px, comp.y + py)

    def render_frame_svg(self, frame: PlacementFrame, width: int = 800) -> str:
        """Render a single frame as SVG.

        Args:
            frame: Frame to render
            width: SVG width in pixels

        Returns:
            SVG string
        """
        # Calculate dimensions
        board_width = self.max_x - self.min_x
        board_height = self.max_y - self.min_y

        if board_width <= 0 or board_height <= 0:
            board_width = 100
            board_height = 100

        aspect = board_height / board_width
        height = int(width * aspect)

        # Padding for labels
        padding = 50
        svg_width = width + 2 * padding
        svg_height = height + 2 * padding

        # Scale factor
        scale = width / board_width

        def tx(x: float) -> float:
            return padding + (x - self.min_x) * scale

        def ty(y: float) -> float:
            # Flip Y for SVG coordinate system
            return padding + (self.max_y - y) * scale

        def ts(size: float) -> float:
            return size * scale

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_width}" height="{svg_height}" '
            f'viewBox="0 0 {svg_width} {svg_height}">'
        ]

        # Background
        svg_parts.append(f'<rect width="100%" height="100%" fill="#1a1a2e"/>')

        # Board outline
        if self.board and self.board.outline:
            bbox = self.board.outline.get_bounding_box()
            if bbox:
                svg_parts.append(
                    f'<rect x="{tx(bbox[0])}" y="{ty(bbox[3])}" '
                    f'width="{ts(bbox[2]-bbox[0])}" height="{ts(bbox[3]-bbox[1])}" '
                    f'fill="#16213e" stroke="#0f3460" stroke-width="2" '
                    f'class="board-outline"/>'
                )

        # Render grid
        if self.grid_spacing > 0:
            # Start grid lines from nearest grid point
            grid_start_x = math.ceil(self.min_x / self.grid_spacing) * self.grid_spacing
            grid_start_y = math.ceil(self.min_y / self.grid_spacing) * self.grid_spacing

            # Vertical grid lines
            x = grid_start_x
            while x <= self.max_x:
                svg_parts.append(
                    f'<line x1="{tx(x)}" y1="{ty(self.min_y)}" '
                    f'x2="{tx(x)}" y2="{ty(self.max_y)}" '
                    f'stroke="#2a2a4e" stroke-width="0.5" class="grid-line"/>'
                )
                x += self.grid_spacing

            # Horizontal grid lines
            y = grid_start_y
            while y <= self.max_y:
                svg_parts.append(
                    f'<line x1="{tx(self.min_x)}" y1="{ty(y)}" '
                    f'x2="{tx(self.max_x)}" y2="{ty(y)}" '
                    f'stroke="#2a2a4e" stroke-width="0.5" class="grid-line"/>'
                )
                y += self.grid_spacing

        # Render module group bounding boxes
        # Group components by module type
        groups: Dict[str, List[str]] = {}
        for ref, module_type in frame.modules.items():
            if ref in frame.components and module_type and module_type != "default" and module_type != "unknown":
                if module_type not in groups:
                    groups[module_type] = []
                groups[module_type].append(ref)

        # Draw bounding box for each group
        for module_type, refs in groups.items():
            if len(refs) < 2:  # Only show groups with 2+ components
                continue

            # Calculate bounding box of all components in this group
            group_min_x = float('inf')
            group_min_y = float('inf')
            group_max_x = float('-inf')
            group_max_y = float('-inf')

            for ref in refs:
                x, y, rotation, w, h = frame.components[ref]
                # Account for component size
                hw, hh = w / 2, h / 2
                group_min_x = min(group_min_x, x - hw)
                group_min_y = min(group_min_y, y - hh)
                group_max_x = max(group_max_x, x + hw)
                group_max_y = max(group_max_y, y + hh)

            if group_min_x < float('inf'):
                # Add padding around the group
                padding = 1.5  # mm
                group_min_x -= padding
                group_min_y -= padding
                group_max_x += padding
                group_max_y += padding

                color = MODULE_COLORS.get(module_type, MODULE_COLORS["default"])

                # Draw transparent bounding box
                svg_parts.append(
                    f'<rect x="{tx(group_min_x)}" y="{ty(group_max_y)}" '
                    f'width="{ts(group_max_x - group_min_x)}" height="{ts(group_max_y - group_min_y)}" '
                    f'fill="{color}" fill-opacity="0.15" '
                    f'stroke="{color}" stroke-width="1.5" stroke-dasharray="4,2" '
                    f'class="module-group module-{module_type.replace("_", "-")}"/>'
                )

                # Draw group label
                label_x = tx((group_min_x + group_max_x) / 2)
                label_y = ty(group_max_y) - 5
                svg_parts.append(
                    f'<text x="{label_x}" y="{label_y}" '
                    f'font-family="monospace" font-size="10" font-weight="bold" '
                    f'fill="{color}" text-anchor="middle" '
                    f'class="module-group group-label module-{module_type.replace("_", "-")}">'
                    f'{module_type}</text>'
                )

        # Render overlapping pairs highlight
        overlap_refs = set()
        for overlap in frame.overlaps:
            # Handle both (ref1, ref2) and (ref1, ref2, distance) formats
            ref1, ref2 = overlap[0], overlap[1]
            overlap_refs.add(ref1)
            overlap_refs.add(ref2)

        # Render net connections (rats nest) - dim lines showing connectivity
        for ref1, ref2, net_name in frame.connections:
            if ref1 in frame.components and ref2 in frame.components:
                x1, y1 = frame.components[ref1][:2]
                x2, y2 = frame.components[ref2][:2]
                # Color based on net type
                if "gnd" in net_name.lower() or "vss" in net_name.lower():
                    line_color = "#2ecc71"  # Green for ground
                elif "vcc" in net_name.lower() or "vdd" in net_name.lower():
                    line_color = "#e74c3c"  # Red for power
                else:
                    line_color = "#4a6fa5"  # Dim blue for signals
                svg_parts.append(
                    f'<line x1="{tx(x1)}" y1="{ty(y1)}" x2="{tx(x2)}" y2="{ty(y2)}" '
                    f'stroke="{line_color}" stroke-width="0.5" opacity="0.3" '
                    f'class="ratsnest"/>'
                )

        # Movement trails are now rendered dynamically in JavaScript
        # to show breadcrumb paths across frames (see drawTrails function)

        # Render components
        for ref, (x, y, rotation, w, h) in frame.components.items():
            # Get module color
            module_type = frame.modules.get(ref, "default")
            color = MODULE_COLORS.get(module_type, MODULE_COLORS["default"])

            # Highlight overlapping components
            stroke_color = "#ff0000" if ref in overlap_refs else "#ffffff"
            stroke_width = 2 if ref in overlap_refs else 1

            # Determine layer class for filtering
            layer_class = "comp-top"  # Default to top
            if self.board and ref in self.board.components:
                comp_obj = self.board.components[ref]
                from ..board.abstraction import Layer
                if comp_obj.layer == Layer.BOTTOM_COPPER:
                    layer_class = "comp-bottom"

            # Component body (rotated rectangle)
            cx, cy = tx(x), ty(y)
            hw, hh = ts(w/2), ts(h/2)

            # Apply rotation transform with layer class and module type
            module_class = f"module-{module_type.replace('_', '-')}" if module_type else "module-default"
            svg_parts.append(
                f'<g class="{layer_class} {module_class}" transform="rotate({-rotation} {cx} {cy})">'
            )

            # Component body
            svg_parts.append(
                f'<rect x="{cx - hw}" y="{cy - hh}" '
                f'width="{ts(w)}" height="{ts(h)}" '
                f'fill="{color}" fill-opacity="0.6" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width}"/>'
            )

            svg_parts.append('</g>')

            # Reference label - placed outside component to avoid overlap
            font_size = max(7, min(11, ts(min(w, h) * 0.35)))
            label_padding = 3  # Pixels from component edge

            # Estimate label dimensions
            label_width = len(ref) * font_size * 0.6
            label_height = font_size

            # Choose best position: prefer above/below for wide, left/right for tall
            # Also check if component is small - if so, always place outside
            comp_is_small = ts(w) < 25 or ts(h) < 25

            if w >= h or comp_is_small:
                # Place above component
                label_x = cx
                label_y = cy - hh - label_padding - label_height / 2
                anchor = "middle"
            else:
                # Place to the right of component
                label_x = cx + hw + label_padding + label_width / 2
                label_y = cy
                anchor = "middle"

            svg_parts.append(
                f'<text x="{label_x}" y="{label_y}" '
                f'font-family="monospace" font-size="{font_size}" '
                f'fill="#dddddd" text-anchor="{anchor}" dominant-baseline="middle" '
                f'class="ref-label {layer_class} {module_class}">'
                f'{ref}</text>'
            )

        # Render pads
        for ref, pad_list in frame.pads.items():
            if ref not in frame.components:
                continue

            comp_data = frame.components[ref]
            rotation = comp_data[2]

            # Get module type for this component
            pad_module_type = frame.modules.get(ref, "default")
            pad_module_class = f"module-{pad_module_type.replace('_', '-')}" if pad_module_type else "module-default"

            # Determine layer class for this component's pads
            pad_layer_class = f"comp-top pad-element {pad_module_class}"  # Default to top
            if self.board and ref in self.board.components:
                comp_obj = self.board.components[ref]
                from ..board.abstraction import Layer
                if comp_obj.layer == Layer.BOTTOM_COPPER:
                    pad_layer_class = f"comp-bottom pad-element {pad_module_class}"

            for px, py, pw, ph, net in pad_list:
                pcx, pcy = tx(px), ty(py)
                phw, phh = ts(pw/2), ts(ph/2)

                # Pad color based on net
                if "gnd" in net.lower() or "vss" in net.lower():
                    pad_color = "#2ecc71"  # Green for ground
                elif "vcc" in net.lower() or "vdd" in net.lower() or "pwr" in net.lower():
                    pad_color = "#e74c3c"  # Red for power
                elif net:
                    pad_color = "#3498db"  # Blue for signals
                else:
                    pad_color = "#95a5a6"  # Grey for unconnected

                svg_parts.append(
                    f'<rect x="{pcx - phw}" y="{pcy - phh}" '
                    f'width="{ts(pw)}" height="{ts(ph)}" '
                    f'fill="{pad_color}" stroke="#ffffff" stroke-width="0.5" '
                    f'class="{pad_layer_class}"/>'
                )

        # Render force vectors with clamped length and colored arrows
        max_arrow_length = 25  # Maximum arrow length in pixels
        for ref, force_list in frame.forces.items():
            if ref not in frame.components:
                continue

            comp_x, comp_y = frame.components[ref][:2]
            cx, cy = tx(comp_x), ty(comp_y)

            for fx, fy, force_type in force_list:
                if abs(fx) < 0.1 and abs(fy) < 0.1:
                    continue

                color = FORCE_COLORS.get(force_type, "#888888")

                # Calculate arrow length and clamp it
                raw_length = math.sqrt(fx * fx + fy * fy)
                if raw_length < 0.1:
                    continue

                # Normalize and scale - use log scale for better visibility
                arrow_length = min(5 + math.log1p(raw_length) * 5, max_arrow_length)
                nx, ny = fx / raw_length, fy / raw_length

                end_x = cx + nx * arrow_length
                end_y = cy - ny * arrow_length  # Flip Y for SVG

                # Draw line
                svg_parts.append(
                    f'<line x1="{cx}" y1="{cy}" x2="{end_x}" y2="{end_y}" '
                    f'stroke="{color}" stroke-width="2" class="force-vector"/>'
                )

                # Draw arrow head as a small triangle at the end
                # Calculate perpendicular direction for arrow head
                head_size = 4
                px, py = -ny, nx  # Perpendicular
                # Arrow head points
                tip_x, tip_y = end_x, end_y
                base1_x = end_x - nx * head_size + px * head_size * 0.5
                base1_y = end_y + ny * head_size + py * head_size * 0.5
                base2_x = end_x - nx * head_size - px * head_size * 0.5
                base2_y = end_y + ny * head_size - py * head_size * 0.5

                svg_parts.append(
                    f'<polygon points="{tip_x},{tip_y} {base1_x},{base1_y} {base2_x},{base2_y}" '
                    f'fill="{color}" class="force-vector"/>'
                )

        # Arrow marker definitions for each force type (for future use)
        svg_parts.insert(1, '''
        <defs>
            <marker id="arrow-repulsion" markerWidth="10" markerHeight="10"
                    refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#ffffff"/>
            </marker>
        </defs>
        ''')

        # Board name and dimensions label at bottom-left of board outline
        if self.board:
            board_name = self.board.name or (self.board.source_file.stem if self.board.source_file else "Board")
            board_width_mm = self.max_x - self.min_x - 10  # Subtract margin
            board_height_mm = self.max_y - self.min_y - 10
            if self.board.outline and self.board.outline.has_outline:
                board_width_mm = self.board.outline.width
                board_height_mm = self.board.outline.height

            # Position label at bottom-left corner, just outside the board edge
            label_x = tx(self.min_x + 5)  # Slight offset from left edge
            label_y = ty(self.min_y) + 12  # Just below bottom edge

            svg_parts.append(
                f'<text x="{label_x}" y="{label_y}" '
                f'font-family="sans-serif" font-size="11" font-weight="bold" '
                f'fill="#888888" text-anchor="start">'
                f'{board_name}</text>'
            )
            svg_parts.append(
                f'<text x="{label_x}" y="{label_y + 12}" '
                f'font-family="monospace" font-size="9" '
                f'fill="#666666" text-anchor="start">'
                f'{board_width_mm:.1f} x {board_height_mm:.1f} mm</text>'
            )

        svg_parts.append('</svg>')

        return '\n'.join(svg_parts)

    def export_html_report(
        self,
        filename: str = "placement_debug.html",
        output_dir: str = "placement_debug"
    ) -> Path:
        """Export interactive HTML report with all frames.

        Args:
            filename: Output filename
            output_dir: Output directory

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        html_path = output_path / filename

        # Generate frame data as JSON-like structure
        frames_js = []
        for frame in self.frames:
            svg = self.render_frame_svg(frame)
            # Escape for JavaScript
            svg_escaped = svg.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '')

            # Extract component positions for trail rendering
            comp_positions = {ref: (x, y) for ref, (x, y, *_) in frame.components.items()}

            frames_js.append({
                'index': frame.index,
                'label': frame.label,
                'phase': frame.phase,
                'iteration': frame.iteration,
                'energy': frame.energy,
                'max_move': frame.max_move,
                'overlap_count': frame.overlap_count,
                'wire_length': frame.total_wire_length,
                'svg': svg_escaped,
                'positions': comp_positions,
            })

        # Build frames JavaScript array
        import json
        frames_json = "[\n"
        for f in frames_js:
            positions_json = json.dumps(f['positions'])
            frames_json += f"  {{'index': {f['index']}, 'label': '{f['label']}', "
            frames_json += f"'phase': '{f['phase']}', 'iteration': {f['iteration']}, "
            frames_json += f"'energy': {f['energy']:.4f}, 'max_move': {f['max_move']:.6f}, "
            frames_json += f"'overlap_count': {f['overlap_count']}, "
            frames_json += f"'wire_length': {f['wire_length']:.2f}, "
            frames_json += f"'positions': {positions_json}, "
            frames_json += f"'svg': '{f['svg']}'}},\n"
        frames_json += "]"

        # Collect unique module types for legend
        unique_modules = set()
        for frame in self.frames:
            unique_modules.update(frame.modules.values())
        unique_modules = sorted(unique_modules)

        # Generate module legend items HTML with checkboxes for toggling
        module_legend_items = ""
        for module_type in unique_modules:
            color = MODULE_COLORS.get(module_type, MODULE_COLORS["default"])
            safe_id = module_type.replace("_", "-")
            module_legend_items += f'''
                <div class="layer-item">
                    <input type="checkbox" class="layer-checkbox" id="show-module-{safe_id}" checked onchange="updateModuleVisibility()">
                    <span class="color-swatch" style="background: {color};"></span>
                    <span class="layer-name">{module_type}</span>
                </div>'''

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Placement Visualization Debug</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        html, body {{
            height: 100%;
            margin: 0;
            overflow: hidden;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #0f0f23;
            color: #cccccc;
            display: flex;
            flex-direction: column;
        }}
        .main-layout {{
            display: flex;
            flex: 1;
            min-height: 0;
            gap: 0;
        }}
        .content-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
            padding: 8px;
            padding-right: 4px;
        }}
        /* KiCad-style Side Panel */
        .side-panel {{
            width: 200px;
            background: #1a1a2e;
            border-left: 1px solid #2a2a4e;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            overflow-y: auto;
            transition: width 0.2s;
        }}
        .side-panel.collapsed {{
            width: 32px;
        }}
        .side-panel.collapsed .panel-content {{
            display: none;
        }}
        .panel-toggle {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 10px;
            background: #16213e;
            border-bottom: 1px solid #2a2a4e;
            cursor: pointer;
            user-select: none;
        }}
        .panel-toggle:hover {{
            background: #1a2a4e;
        }}
        .panel-toggle-icon {{
            font-size: 14px;
            color: #888;
            transition: transform 0.2s;
        }}
        .side-panel.collapsed .panel-toggle-icon {{
            transform: rotate(180deg);
        }}
        .panel-toggle-text {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
        }}
        .side-panel.collapsed .panel-toggle-text {{
            display: none;
        }}
        .panel-content {{
            padding: 0;
        }}
        .panel-section {{
            border-bottom: 1px solid #2a2a4e;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 10px;
            background: #1a1a3e;
            cursor: pointer;
            user-select: none;
            font-size: 11px;
            font-weight: 600;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        .section-header:hover {{
            background: #252545;
        }}
        .section-icon {{
            font-size: 10px;
            color: #666;
            transition: transform 0.2s;
        }}
        .section-header.collapsed .section-icon {{
            transform: rotate(-90deg);
        }}
        .section-content {{
            padding: 6px 10px;
            background: #16213e;
        }}
        .section-header.collapsed + .section-content {{
            display: none;
        }}
        .layer-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            cursor: pointer;
        }}
        .layer-item:hover {{
            background: rgba(255,255,255,0.03);
            margin: 0 -10px;
            padding: 4px 10px;
        }}
        .layer-checkbox {{
            width: 14px;
            height: 14px;
            cursor: pointer;
            accent-color: #3498db;
        }}
        .layer-icon {{
            width: 16px;
            height: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }}
        .color-swatch {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .layer-name {{
            font-size: 11px;
            color: #ccc;
            flex: 1;
        }}
        .layer-key {{
            font-size: 9px;
            color: #666;
            background: #252545;
            padding: 1px 4px;
            border-radius: 2px;
            font-family: monospace;
        }}
        /* Controls bar */
        .controls {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 8px;
            background: #1a1a3e;
            border-radius: 4px;
            margin: 4px 0;
            flex-shrink: 0;
        }}
        .controls button {{
            padding: 5px 10px;
            font-size: 12px;
            cursor: pointer;
            background: #252550;
            color: #ffffff;
            border: 1px solid #3498db;
            border-radius: 4px;
            transition: background 0.2s;
            min-width: 32px;
        }}
        .controls button:hover {{
            background: #3a3a6e;
        }}
        .controls-spacer {{
            flex: 1;
        }}
        .speed-control select {{
            padding: 5px 8px;
            background: #252550;
            color: #ffffff;
            border: 1px solid #3498db;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
        }}
        .info {{
            margin: 4px 0;
            padding: 6px 12px;
            background: #1a1a3e;
            border-radius: 4px;
            font-family: monospace;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            font-size: 11px;
            flex-shrink: 0;
        }}
        .info-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .info-label {{
            color: #888;
            font-size: 10px;
            text-transform: uppercase;
        }}
        .info-value {{
            color: #00d4ff;
            font-size: 12px;
            font-weight: bold;
        }}
        .frame-container {{
            border: 1px solid #3498db;
            border-radius: 4px;
            flex: 1;
            min-height: 0;
            overflow: hidden;
            background: #1a1a2e;
            position: relative;
            cursor: grab;
        }}
        .frame-container.dragging {{
            cursor: grabbing;
        }}
        .frame-container svg {{
            position: absolute;
            transform-origin: 0 0;
            transition: none;
        }}
        .zoom-controls {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .zoom-controls button {{
            width: 26px;
            height: 26px;
            padding: 0;
            font-size: 14px;
            font-weight: bold;
        }}
        .zoom-level {{
            font-size: 10px;
            color: #888;
            min-width: 40px;
            text-align: center;
        }}
        .energy-graph {{
            margin: 4px 0;
            padding: 5px;
            background: #1a1a3e;
            border-radius: 4px;
            height: 70px;
            flex-shrink: 0;
            cursor: pointer;
            user-select: none;
        }}
        .energy-graph:hover {{
            background: #1f1f4a;
        }}
        .energy-graph.seeking {{
            cursor: grabbing;
        }}
        .energy-graph canvas {{
            width: 100%;
            height: 50px;
            pointer-events: none;
        }}
        .graph-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 9px;
            color: #666;
            margin-top: 2px;
        }}
        .keyboard-hints {{
            padding: 5px 10px;
            font-size: 9px;
            color: #555;
            flex-shrink: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
        }}
        .keyboard-hints kbd {{
            background: #1a1a3e;
            padding: 1px 4px;
            border-radius: 2px;
            border: 1px solid #333;
            font-size: 8px;
            color: #777;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            font-size: 10px;
        }}
        .stat-label {{
            color: #888;
        }}
        .stat-value {{
            color: #00d4ff;
            font-weight: 600;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="main-layout">
        <div class="content-area">
            <div class="info">
                <div class="info-item">
                    <span class="info-label">Frame</span>
                    <span class="info-value"><span id="frame-num">0</span> / {len(self.frames) - 1}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Phase</span>
                    <span class="info-value" id="phase">-</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Iter</span>
                    <span class="info-value" id="iteration">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Energy</span>
                    <span class="info-value" id="energy">0.00</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Move</span>
                    <span class="info-value" id="max-move">0.000</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Overlaps</span>
                    <span class="info-value" id="overlaps">0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Wire</span>
                    <span class="info-value" id="wire-length">0.0mm</span>
                </div>
            </div>
            <div class="controls">
                <button onclick="firstFrame()" title="First frame">|&lt;</button>
                <button onclick="prevFrame()" title="Previous frame">&lt;</button>
                <button onclick="togglePlay()" id="play-btn" title="Play/Pause">&#9658;</button>
                <button onclick="nextFrame()" title="Next frame">&gt;</button>
                <button onclick="lastFrame()" title="Last frame">&gt;|</button>
                <div class="speed-control">
                    <select id="speed" onchange="updateSpeed()" title="Playback speed">
                        <option value="500">0.5x</option>
                        <option value="200" selected>1x</option>
                        <option value="100">2x</option>
                        <option value="50">4x</option>
                    </select>
                </div>
                <div class="controls-spacer"></div>
                <div class="zoom-controls">
                    <button onclick="zoomOut()" title="Zoom out">-</button>
                    <span class="zoom-level" id="zoom-level">100%</span>
                    <button onclick="zoomIn()" title="Zoom in">+</button>
                    <button onclick="resetView()" title="Reset view">&#8634;</button>
                </div>
            </div>

            <div class="frame-container" id="frame-container"></div>

            <div class="energy-graph">
                <canvas id="energy-canvas"></canvas>
                <div class="graph-labels">
                    <span>Energy (blue) / Wire Length (green)</span>
                    <span id="graph-range"></span>
                </div>
            </div>

            <div class="keyboard-hints">
                <span><kbd>&#x2190;</kbd>/<kbd>&#x2192;</kbd> Prev/Next</span>
                <span><kbd>Space</kbd> Play</span>
                <span><kbd>Home</kbd>/<kbd>End</kbd> First/Last</span>
                <span><kbd>G</kbd> Grid</span>
                <span><kbd>R</kbd> Nets</span>
                <span><kbd>T</kbd> Trails</span>
                <span><kbd>F</kbd> Forces</span>
                <span><kbd>L</kbd> Labels</span>
                <span><kbd>M</kbd> Groups</span>
                <span><kbd>+/-</kbd> Zoom</span>
                <span><kbd>0</kbd> Reset</span>
                <span>Wheel: Zoom | Drag: Pan</span>
            </div>
        </div>

        <!-- KiCad-style Side Panel -->
        <div class="side-panel" id="side-panel">
            <div class="panel-toggle" onclick="togglePanel()">
                <span class="panel-toggle-text">Appearance</span>
                <span class="panel-toggle-icon">&#9664;</span>
            </div>
            <div class="panel-content">
                <!-- Copper Layers Section -->
                <div class="panel-section">
                    <div class="section-header" onclick="toggleSection(this)">
                        <span>Copper Layers</span>
                        <span class="section-icon">&#9660;</span>
                    </div>
                    <div class="section-content">
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-top" checked onchange="updateLayers()">
                            <span class="color-swatch" style="background: #e74c3c;"></span>
                            <span class="layer-name">Top (F.Cu)</span>
                            <span class="layer-key">1</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-bottom" checked onchange="updateLayers()">
                            <span class="color-swatch" style="background: #3498db;"></span>
                            <span class="layer-name">Bottom (B.Cu)</span>
                            <span class="layer-key">2</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-edge" checked onchange="updateLayers()">
                            <span class="color-swatch" style="background: #0f3460;"></span>
                            <span class="layer-name">Edge.Cuts</span>
                            <span class="layer-key">E</span>
                        </div>
                    </div>
                </div>

                <!-- Display Layers Section -->
                <div class="panel-section">
                    <div class="section-header" onclick="toggleSection(this)">
                        <span>Display Layers</span>
                        <span class="section-icon">&#9660;</span>
                    </div>
                    <div class="section-content">
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-grid" onchange="updateLayers()">
                            <span class="layer-icon">&#9783;</span>
                            <span class="layer-name">Grid</span>
                            <span class="layer-key">G</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-ratsnest" checked onchange="updateLayers()">
                            <span class="layer-icon" style="color:#4a6fa5;">&#9644;</span>
                            <span class="layer-name">Ratsnest</span>
                            <span class="layer-key">R</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-trails" checked onchange="updateLayers()">
                            <span class="layer-icon" style="color:#ff6b6b;">&#8594;</span>
                            <span class="layer-name">Trails</span>
                            <span class="layer-key">T</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-forces" onchange="updateLayers()">
                            <span class="layer-icon" style="color:#e74c3c;">&#10140;</span>
                            <span class="layer-name">Forces</span>
                            <span class="layer-key">F</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-pads" checked onchange="updateLayers()">
                            <span class="layer-icon" style="color:#3498db;">&#9632;</span>
                            <span class="layer-name">Pads</span>
                            <span class="layer-key">P</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-labels" checked onchange="updateLayers()">
                            <span class="layer-icon">A</span>
                            <span class="layer-name">Labels</span>
                            <span class="layer-key">L</span>
                        </div>
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-groups" checked onchange="updateLayers()">
                            <span class="layer-icon" style="color:#9b59b6;">&#9634;</span>
                            <span class="layer-name">Groups</span>
                            <span class="layer-key">M</span>
                        </div>
                    </div>
                </div>

                <!-- Module Types Legend -->
                <div class="panel-section">
                    <div class="section-header" onclick="toggleSection(this)">
                        <span>Module Types</span>
                        <span class="section-icon">&#9660;</span>
                    </div>
                    <div class="section-content">
                        {module_legend_items}
                    </div>
                </div>

                <!-- Force Types Legend -->
                <div class="panel-section">
                    <div class="section-header" onclick="toggleSection(this)">
                        <span>Force Types</span>
                        <span class="section-icon">&#9660;</span>
                    </div>
                    <div class="section-content">
                        <div class="layer-item">
                            <span class="color-swatch" style="background: #e74c3c;"></span>
                            <span class="layer-name">Repulsion</span>
                        </div>
                        <div class="layer-item">
                            <span class="color-swatch" style="background: #2ecc71;"></span>
                            <span class="layer-name">Attraction</span>
                        </div>
                        <div class="layer-item">
                            <span class="color-swatch" style="background: #3498db;"></span>
                            <span class="layer-name">Boundary</span>
                        </div>
                        <div class="layer-item">
                            <span class="color-swatch" style="background: #f39c12;"></span>
                            <span class="layer-name">Constraint</span>
                        </div>
                        <div class="layer-item">
                            <span class="color-swatch" style="background: #9b59b6;"></span>
                            <span class="layer-name">Alignment</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const frames = {frames_json};
        let currentFrame = 0;
        let playing = false;
        let playInterval = null;
        let playSpeed = 200;

        // Board bounds for coordinate transformation (from Python)
        const boardBounds = {{
            minX: {self.min_x},
            maxX: {self.max_x},
            minY: {self.min_y},
            maxY: {self.max_y},
            padding: 50,
            width: 800
        }};

        // Zoom and pan state
        let zoom = 1;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let panStartX = 0;
        let panStartY = 0;
        const MIN_ZOOM = 0.25;
        const MAX_ZOOM = 10;
        const ZOOM_STEP = 0.25;

        // Panel functions
        function togglePanel() {{
            document.getElementById('side-panel').classList.toggle('collapsed');
        }}

        function toggleSection(header) {{
            header.classList.toggle('collapsed');
        }}

        function showFrame(idx) {{
            currentFrame = Math.max(0, Math.min(frames.length - 1, idx));
            const frame = frames[currentFrame];

            document.getElementById('frame-container').innerHTML = frame.svg;
            applyTransform();

            // Update stats in main info bar
            const frameNum = document.getElementById('frame-num');
            const phase = document.getElementById('phase');
            const iteration = document.getElementById('iteration');
            const energy = document.getElementById('energy');
            const maxMove = document.getElementById('max-move');
            const overlaps = document.getElementById('overlaps');
            const wireLength = document.getElementById('wire-length');

            if (frameNum) frameNum.textContent = currentFrame;
            if (phase) phase.textContent = frame.phase;
            if (iteration) iteration.textContent = frame.iteration;
            if (energy) energy.textContent = frame.energy.toFixed(2);
            if (maxMove) maxMove.textContent = frame.max_move.toFixed(4);
            if (overlaps) overlaps.textContent = frame.overlap_count;
            if (wireLength) wireLength.textContent = frame.wire_length.toFixed(1) + 'mm';


            // Apply layer visibility
            updateLayers();

            // Draw breadcrumb trails
            drawTrails();

            // Update energy graph cursor
            drawEnergyGraph();
        }}

        // Draw breadcrumb trails showing component movement history
        function drawTrails() {{
            const svg = document.querySelector('#frame-container svg');
            if (!svg || currentFrame < 1) return;

            // Remove existing trails
            svg.querySelectorAll('.movement-trail').forEach(el => el.remove());

            // Get SVG dimensions for coordinate transformation
            const svgWidth = parseFloat(svg.getAttribute('width')) || 800;
            const svgHeight = parseFloat(svg.getAttribute('height')) || 600;
            const boardWidth = boardBounds.maxX - boardBounds.minX;
            const boardHeight = boardBounds.maxY - boardBounds.minY;
            const scale = (svgWidth - 2 * boardBounds.padding) / boardWidth;

            // Transform board coords to SVG coords
            function tx(x) {{ return boardBounds.padding + (x - boardBounds.minX) * scale; }}
            function ty(y) {{ return boardBounds.padding + (boardBounds.maxY - y) * scale; }}

            // Check if trails should be visible
            const showTrails = document.getElementById('show-trails')?.checked || false;
            if (!showTrails) return;

            // Create SVG group for trails
            const trailGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            trailGroup.setAttribute('class', 'movement-trail');

            // Get all component refs from current frame
            const currentPositions = frames[currentFrame].positions;

            for (const ref of Object.keys(currentPositions)) {{
                // Collect position history for this component
                const history = [];
                for (let i = 0; i <= currentFrame; i++) {{
                    const pos = frames[i].positions[ref];
                    if (pos) {{
                        history.push({{ x: pos[0], y: pos[1], frame: i }});
                    }}
                }}

                if (history.length < 2) continue;

                // Check if component moved significantly
                const start = history[0];
                const end = history[history.length - 1];
                const totalDist = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
                if (totalDist < 0.5) continue;

                // Draw breadcrumb dots along the path
                for (let i = 0; i < history.length; i++) {{
                    const pos = history[i];
                    // Fade older positions
                    const age = (currentFrame - pos.frame) / Math.max(currentFrame, 1);
                    const opacity = 0.2 + (1 - age) * 0.6;
                    const radius = 1.5 + (1 - age) * 1.5;

                    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    dot.setAttribute('cx', tx(pos.x));
                    dot.setAttribute('cy', ty(pos.y));
                    dot.setAttribute('r', radius);
                    dot.setAttribute('fill', '#ff6b6b');
                    dot.setAttribute('opacity', opacity);
                    trailGroup.appendChild(dot);
                }}

                // Draw connecting line segments
                if (history.length > 1) {{
                    let pathD = `M ${{tx(history[0].x)}} ${{ty(history[0].y)}}`;
                    for (let i = 1; i < history.length; i++) {{
                        pathD += ` L ${{tx(history[i].x)}} ${{ty(history[i].y)}}`;
                    }}
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    path.setAttribute('d', pathD);
                    path.setAttribute('stroke', '#ff6b6b');
                    path.setAttribute('stroke-width', '1');
                    path.setAttribute('stroke-opacity', '0.4');
                    path.setAttribute('fill', 'none');
                    trailGroup.appendChild(path);
                }}

                // Mark initial position with a larger circle
                const startDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                startDot.setAttribute('cx', tx(start.x));
                startDot.setAttribute('cy', ty(start.y));
                startDot.setAttribute('r', '3');
                startDot.setAttribute('fill', 'none');
                startDot.setAttribute('stroke', '#ff6b6b');
                startDot.setAttribute('stroke-width', '1.5');
                startDot.setAttribute('opacity', '0.7');
                trailGroup.appendChild(startDot);
            }}

            // Insert trails before components (so they appear behind)
            const firstG = svg.querySelector('g');
            if (firstG) {{
                svg.insertBefore(trailGroup, firstG);
            }} else {{
                svg.appendChild(trailGroup);
            }}
        }}

        function updateLayers() {{
            const showGrid = document.getElementById('show-grid')?.checked ?? false;
            const showRatsnest = document.getElementById('show-ratsnest')?.checked ?? false;
            const showTrails = document.getElementById('show-trails')?.checked ?? false;
            const showForces = document.getElementById('show-forces')?.checked ?? false;
            const showPads = document.getElementById('show-pads')?.checked ?? false;
            const showLabels = document.getElementById('show-labels')?.checked ?? false;
            const showGroups = document.getElementById('show-groups')?.checked ?? false;
            const showTop = document.getElementById('show-top')?.checked ?? true;
            const showBottom = document.getElementById('show-bottom')?.checked ?? true;
            const showEdge = document.getElementById('show-edge')?.checked ?? true;

            // Toggle grid lines
            document.querySelectorAll('.grid-line').forEach(el => {{
                el.style.display = showGrid ? '' : 'none';
            }});

            // Toggle ratsnest lines
            document.querySelectorAll('.ratsnest').forEach(el => {{
                el.style.display = showRatsnest ? '' : 'none';
            }});

            // Toggle movement trails
            document.querySelectorAll('.movement-trail').forEach(el => {{
                el.style.display = showTrails ? '' : 'none';
            }});

            // Toggle force vectors (lines and arrow heads)
            document.querySelectorAll('.force-vector').forEach(el => {{
                el.style.display = showForces ? '' : 'none';
            }});

            // Toggle pads (rectangles with pad-element class)
            document.querySelectorAll('.pad-element').forEach(el => {{
                el.style.display = showPads ? '' : 'none';
            }});

            // Toggle module group bounding boxes
            document.querySelectorAll('.module-group').forEach(el => {{
                el.style.display = showGroups ? '' : 'none';
            }});

            // Toggle board outline (Edge.Cuts)
            document.querySelectorAll('.board-outline').forEach(el => {{
                el.style.display = showEdge ? '' : 'none';
            }});

            // Toggle top layer components
            document.querySelectorAll('.comp-top').forEach(el => {{
                el.style.display = showTop ? '' : 'none';
            }});

            // Toggle bottom layer components
            document.querySelectorAll('.comp-bottom').forEach(el => {{
                el.style.display = showBottom ? '' : 'none';
            }});

            // Handle labels with layer visibility
            document.querySelectorAll('.ref-label').forEach(el => {{
                const isTop = el.classList.contains('comp-top');
                const isBottom = el.classList.contains('comp-bottom');
                const layerVisible = (isTop && showTop) || (isBottom && showBottom) || (!isTop && !isBottom);
                el.style.display = (showLabels && layerVisible) ? '' : 'none';
            }});

            // Apply module visibility after layer visibility
            updateModuleVisibility();

            // Redraw trails if visibility changed
            if (showTrails) {{
                drawTrails();
            }}
        }}

        // Module visibility toggle - respects layer visibility
        function updateModuleVisibility() {{
            const showTop = document.getElementById('show-top')?.checked ?? true;
            const showBottom = document.getElementById('show-bottom')?.checked ?? true;

            // Find all module type checkboxes
            document.querySelectorAll('[id^="show-module-"]').forEach(checkbox => {{
                const moduleType = checkbox.id.replace('show-module-', '');
                const moduleVisible = checkbox.checked;

                // Toggle components with this module type
                document.querySelectorAll('.module-' + moduleType).forEach(el => {{
                    // Check layer visibility first
                    const isTop = el.classList.contains('comp-top');
                    const isBottom = el.classList.contains('comp-bottom');
                    const isModuleGroup = el.classList.contains('module-group');

                    // Module groups don't have layer class, always follow module visibility
                    let layerVisible = true;
                    if (isTop) layerVisible = showTop;
                    else if (isBottom) layerVisible = showBottom;

                    // Element visible only if both layer AND module are visible
                    const shouldShow = layerVisible && moduleVisible;
                    el.style.display = shouldShow ? '' : 'none';
                }});
            }});
        }}

        // Zoom and pan functions
        function applyTransform() {{
            const svg = document.querySelector('#frame-container svg');
            if (!svg) return;

            const container = document.getElementById('frame-container');
            const containerRect = container.getBoundingClientRect();

            // Get SVG dimensions
            const svgWidth = parseFloat(svg.getAttribute('width')) || 800;
            const svgHeight = parseFloat(svg.getAttribute('height')) || 600;

            // Calculate initial fit scale
            const fitScale = Math.min(
                containerRect.width / svgWidth,
                containerRect.height / svgHeight
            ) * 0.95;

            // Calculate centered position at zoom=1
            const baseOffsetX = (containerRect.width - svgWidth * fitScale) / 2;
            const baseOffsetY = (containerRect.height - svgHeight * fitScale) / 2;

            // Apply combined transform
            const finalScale = fitScale * zoom;
            const finalX = baseOffsetX + panX;
            const finalY = baseOffsetY + panY;

            svg.style.transform = `translate(${{finalX}}px, ${{finalY}}px) scale(${{finalScale}})`;
        }}

        function zoomIn() {{
            zoom = Math.min(MAX_ZOOM, zoom + ZOOM_STEP);
            updateZoomDisplay();
            applyTransform();
        }}

        function zoomOut() {{
            zoom = Math.max(MIN_ZOOM, zoom - ZOOM_STEP);
            updateZoomDisplay();
            applyTransform();
        }}

        function zoomTo(level) {{
            zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, level));
            updateZoomDisplay();
            applyTransform();
        }}

        function resetView() {{
            zoom = 1;
            panX = 0;
            panY = 0;
            updateZoomDisplay();
            applyTransform();
        }}

        function updateZoomDisplay() {{
            document.getElementById('zoom-level').textContent = Math.round(zoom * 100) + '%';
        }}

        // Mouse wheel zoom
        function handleWheel(e) {{
            e.preventDefault();
            const container = document.getElementById('frame-container');
            const rect = container.getBoundingClientRect();

            // Mouse position relative to container
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            // Zoom factor
            const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
            const oldZoom = zoom;
            zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom + delta));

            if (zoom !== oldZoom) {{
                // Adjust pan to zoom toward mouse position
                const zoomRatio = zoom / oldZoom;
                panX = mouseX - (mouseX - panX) * zoomRatio;
                panY = mouseY - (mouseY - panY) * zoomRatio;

                updateZoomDisplay();
                applyTransform();
            }}
        }}

        // Pan with mouse drag
        function handleMouseDown(e) {{
            if (e.button !== 0) return;  // Only left click
            isDragging = true;
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            panStartX = panX;
            panStartY = panY;
            document.getElementById('frame-container').classList.add('dragging');
            e.preventDefault();
        }}

        function handleMouseMove(e) {{
            if (!isDragging) return;
            panX = panStartX + (e.clientX - dragStartX);
            panY = panStartY + (e.clientY - dragStartY);
            applyTransform();
        }}

        function handleMouseUp(e) {{
            if (isDragging) {{
                isDragging = false;
                document.getElementById('frame-container').classList.remove('dragging');
            }}
        }}

        function drawEnergyGraph() {{
            const canvas = document.getElementById('energy-canvas');
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const w = rect.width;
            const h = rect.height;

            // Clear canvas
            ctx.fillStyle = '#1a1a3e';
            ctx.fillRect(0, 0, w, h);

            if (frames.length < 2) return;

            // Find max values for scaling
            const maxEnergy = Math.max(...frames.map(f => f.energy));
            const maxWireLength = Math.max(...frames.map(f => f.wire_length));

            // Draw energy line (blue)
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.beginPath();
            frames.forEach((f, i) => {{
                const x = (i / (frames.length - 1)) * w;
                const y = h - (f.energy / maxEnergy) * h * 0.9;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();

            // Draw wire length line (green)
            ctx.strokeStyle = '#2ecc71';
            ctx.lineWidth = 2;
            ctx.beginPath();
            frames.forEach((f, i) => {{
                const x = (i / (frames.length - 1)) * w;
                const y = h - (f.wire_length / maxWireLength) * h * 0.9;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();

            // Draw playhead line
            const curX = (currentFrame / (frames.length - 1)) * w;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fillRect(curX - 1, 0, 3, h);

            // Draw playhead handle
            ctx.beginPath();
            ctx.arc(curX, 5, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#00d4ff';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Update range label
            document.getElementById('graph-range').textContent =
                `Energy (blue): 0-${{maxEnergy.toFixed(0)}} | Wire Length (green): 0-${{maxWireLength.toFixed(0)}}mm`;
        }}

        function firstFrame() {{ showFrame(0); }}
        function lastFrame() {{ showFrame(frames.length - 1); }}
        function prevFrame() {{ showFrame(currentFrame - 1); }}
        function nextFrame() {{ showFrame(currentFrame + 1); }}
        function setFrame(idx) {{ showFrame(parseInt(idx)); }}

        function togglePlay() {{
            playing = !playing;
            document.getElementById('play-btn').innerHTML = playing ? '&#9724;' : '&#9658;';
            if (playing) {{
                playInterval = setInterval(() => {{
                    if (currentFrame >= frames.length - 1) {{
                        togglePlay();
                    }} else {{
                        nextFrame();
                    }}
                }}, playSpeed);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        function updateSpeed() {{
            playSpeed = parseInt(document.getElementById('speed').value);
            if (playing) {{
                clearInterval(playInterval);
                playInterval = setInterval(() => {{
                    if (currentFrame >= frames.length - 1) {{
                        togglePlay();
                    }} else {{
                        nextFrame();
                    }}
                }}, playSpeed);
            }}
        }}

        function jumpToPhase(phase) {{
            for (let i = 0; i < frames.length; i++) {{
                if (frames[i].phase === phase) {{
                    showFrame(i);
                    return;
                }}
            }}
        }}

        // Initialize
        showFrame(0);

        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') {{ prevFrame(); e.preventDefault(); }}
            if (e.key === 'ArrowRight') {{ nextFrame(); e.preventDefault(); }}
            if (e.key === ' ') {{ togglePlay(); e.preventDefault(); }}
            if (e.key === 'Home') {{ firstFrame(); e.preventDefault(); }}
            if (e.key === 'End') {{ lastFrame(); e.preventDefault(); }}
            if (e.key === 'g' || e.key === 'G') {{
                const cb = document.getElementById('show-grid');
                cb.checked = !cb.checked;
                updateLayers();
            }}
            if (e.key === 'r' || e.key === 'R') {{
                const cb = document.getElementById('show-ratsnest');
                cb.checked = !cb.checked;
                updateLayers();
            }}
            if (e.key === 't' || e.key === 'T') {{
                const cb = document.getElementById('show-trails');
                cb.checked = !cb.checked;
                updateLayers();
            }}
            if (e.key === 'f' || e.key === 'F') {{
                const cb = document.getElementById('show-forces');
                cb.checked = !cb.checked;
                updateLayers();
            }}
            if (e.key === 'l' || e.key === 'L') {{
                const cb = document.getElementById('show-labels');
                if (cb) {{ cb.checked = !cb.checked; updateLayers(); }}
            }}
            if (e.key === 'p' || e.key === 'P') {{
                const cb = document.getElementById('show-pads');
                if (cb) {{ cb.checked = !cb.checked; updateLayers(); }}
            }}
            if (e.key === 'm' || e.key === 'M') {{
                const cb = document.getElementById('show-groups');
                if (cb) {{ cb.checked = !cb.checked; updateLayers(); }}
            }}
            // Layer shortcuts: 1 = top, 2 = bottom, E = edge
            if (e.key === '1') {{
                const cb = document.getElementById('show-top');
                if (cb) {{ cb.checked = !cb.checked; updateLayers(); }}
            }}
            if (e.key === '2') {{
                const cb = document.getElementById('show-bottom');
                if (cb) {{ cb.checked = !cb.checked; updateLayers(); }}
            }}
            if (e.key === 'e' || e.key === 'E') {{
                const cb = document.getElementById('show-edge');
                if (cb) {{ cb.checked = !cb.checked; updateLayers(); }}
            }}
            // Zoom keyboard shortcuts
            if (e.key === '+' || e.key === '=') {{ zoomIn(); e.preventDefault(); }}
            if (e.key === '-' || e.key === '_') {{ zoomOut(); e.preventDefault(); }}
            if (e.key === '0' && !e.shiftKey) {{ resetView(); e.preventDefault(); }}
        }});

        // Mouse event listeners for pan/zoom
        const frameContainer = document.getElementById('frame-container');
        frameContainer.addEventListener('wheel', handleWheel, {{ passive: false }});
        frameContainer.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);

        // Histogram seeking
        let isSeekingHistogram = false;
        const energyGraph = document.querySelector('.energy-graph');

        function seekFromHistogram(e) {{
            const rect = energyGraph.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const frameIndex = Math.round((x / rect.width) * (frames.length - 1));
            showFrame(frameIndex);
        }}

        function histogramSeekStart(e) {{
            if (e.button !== 0) return;
            isSeekingHistogram = true;
            energyGraph.classList.add('seeking');
            seekFromHistogram(e);
            e.preventDefault();
        }}

        function histogramSeekMove(e) {{
            if (!isSeekingHistogram) return;
            seekFromHistogram(e);
        }}

        function histogramSeekEnd(e) {{
            if (isSeekingHistogram) {{
                isSeekingHistogram = false;
                energyGraph.classList.remove('seeking');
            }}
        }}

        energyGraph.addEventListener('mousedown', histogramSeekStart);
        document.addEventListener('mousemove', histogramSeekMove);
        document.addEventListener('mouseup', histogramSeekEnd);

        // Draw initial graph on load
        window.addEventListener('load', () => {{
            drawEnergyGraph();
        }});

        // Redraw on resize
        window.addEventListener('resize', () => {{
            drawEnergyGraph();
            applyTransform();
        }});
    </script>
</body>
</html>
'''

        html_path.write_text(html_content)
        logger.info(f"Exported placement visualization to {html_path}")

        return html_path


def create_visualizer_from_board(board) -> PlacementVisualizer:
    """Create a visualizer initialized with board data.

    Args:
        board: Board instance

    Returns:
        PlacementVisualizer ready for frame capture
    """
    return PlacementVisualizer(board)
