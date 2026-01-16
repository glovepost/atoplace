"""Routing visualization system for debugging.

Based on @seveibar's autorouter lesson #5:
"If you do not have a visualization for a problem, you will never solve it"

Key insight: You can't debug routing problems by staring at numbers.
Every algorithm component needs visualization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import math
import json
import warnings
import logging

from .spatial_index import Obstacle, SpatialHashIndex
from ..visualization_color_manager import get_color_manager

logger = logging.getLogger(__name__)


@dataclass
class RouteSegment:
    """A segment of a routed trace."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: int
    width: float
    net_id: Optional[int] = None
    net_name: Optional[str] = None  # For visualization net counting (Issue #31)


@dataclass
class Via:
    """A via connecting layers."""
    x: float
    y: float
    drill_diameter: float
    pad_diameter: float
    net_id: Optional[int] = None
    net_name: Optional[str] = None  # For visualization net counting (Issue #31)


@dataclass
class VisualizationFrame:
    """A single frame in a routing visualization.

    Captures the complete state at one point in time for debugging.
    """
    # Static elements
    board_outline: List[Tuple[float, float]] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    pads: List[Obstacle] = field(default_factory=list)  # Pads to connect

    # Dynamic routing state
    completed_traces: List[RouteSegment] = field(default_factory=list)
    completed_vias: List[Via] = field(default_factory=list)
    current_net_name: str = ""

    # A* debugging
    explored_nodes: Set[Tuple[float, float, int]] = field(default_factory=set)
    frontier_nodes: Set[Tuple[float, float, int]] = field(default_factory=set)
    current_path: List[Tuple[float, float, int]] = field(default_factory=list)

    # Metadata
    iteration: int = 0
    label: str = ""


# Color palette is now loaded from visualization_colors.yaml
# via the ColorManager. This allows users to customize colors and
# supports N-layer boards with dynamic color generation.
#
# Helper functions provide backward-compatible access.


def get_routing_color(element: str) -> str:
    """Get color for a routing element.

    Args:
        element: Element name (e.g., "board_outline", "obstacle", "via")

    Returns:
        Hex color string
    """
    return get_color_manager().get_routing_color(element)


def get_layer_color(layer: int, element_type: str = "pad") -> str:
    """Get color for a specific PCB layer.

    Supports multi-layer boards (4, 6, 8+ layers) with dynamic color generation.

    Args:
        layer: Layer number (0 = front, 1 = back, 2+ = inner)
        element_type: "pad" or "trace"

    Returns:
        Hex color string
    """
    return get_color_manager().get_layer_color(layer, element_type)


# Deprecated: Backward compatibility
# Legacy code may reference COLORS dict directly
COLORS = None  # Lazy-loaded on first access


def _get_colors_dict() -> Dict[str, str]:
    """Get legacy COLORS dict for backward compatibility."""
    cm = get_color_manager()
    return {
        "board_outline": cm.get_routing_color("board_outline"),
        "obstacle": cm.get_routing_color("obstacle"),
        "obstacle_stroke": cm.get_routing_color("obstacle_stroke"),
        "pad_f": cm.get_layer_color(0, "pad"),  # Front copper pads
        "pad_b": cm.get_layer_color(1, "pad"),  # Back copper pads
        "trace_f": cm.get_layer_color(0, "trace"),  # Front copper traces
        "trace_b": cm.get_layer_color(1, "trace"),  # Back copper traces
        "via": cm.get_routing_color("via"),
        "explored": cm.get_routing_color("explored"),
        "frontier": cm.get_routing_color("frontier"),
        "current_path": cm.get_routing_color("current_path"),
        "target_pad": cm.get_routing_color("target_pad"),
    }


class RouteVisualizer:
    """Real-time visualization of routing progress.

    Generates SVG frames and HTML reports for debugging routing algorithms.
    Can optionally include board components for context.
    """

    def __init__(
        self,
        board_bounds: Tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        output_dir: Optional[Path] = None,
        scale: float = 10.0,  # pixels per mm
        margin: float = 5.0,  # mm margin around board
        board=None,  # Optional Board object for component rendering
    ):
        """
        Args:
            board_bounds: Board bounding box (min_x, min_y, max_x, max_y)
            output_dir: Directory for output files
            scale: Rendering scale (pixels per mm)
            margin: Margin around board in mm
            board: Optional Board object to show components
        """
        self.bounds = board_bounds
        self.output_dir = output_dir or Path("./route_debug")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self.margin = margin
        self.board = board  # Store board reference

        self.frames: List[VisualizationFrame] = []

        # Calculate SVG dimensions
        self.board_width = board_bounds[2] - board_bounds[0]
        self.board_height = board_bounds[3] - board_bounds[1]
        self.svg_width = (self.board_width + 2 * margin) * scale
        self.svg_height = (self.board_height + 2 * margin) * scale

    def _to_svg_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Convert board coordinates to SVG coordinates."""
        svg_x = (x - self.bounds[0] + self.margin) * self.scale
        # Direct Y mapping (no inversion)
        svg_y = (y - self.bounds[1] + self.margin) * self.scale
        return (svg_x, svg_y)

    def _to_svg_size(self, size: float) -> float:
        """Convert board size to SVG size."""
        return size * self.scale

    def capture_frame(
        self,
        obstacles: List[Obstacle],
        pads: List[Obstacle],
        completed_traces: List[RouteSegment] = None,
        completed_vias: List[Via] = None,
        explored_nodes: Set[Tuple[float, float, int]] = None,
        frontier_nodes: Set[Tuple[float, float, int]] = None,
        current_path: List[Tuple[float, float, int]] = None,
        current_net: str = "",
        label: str = "",
    ):
        """Capture current routing state as a frame."""
        frame = VisualizationFrame(
            obstacles=obstacles,
            pads=pads,
            completed_traces=completed_traces or [],
            completed_vias=completed_vias or [],
            explored_nodes=explored_nodes or set(),
            frontier_nodes=frontier_nodes or set(),
            current_path=current_path or [],
            current_net_name=current_net,
            iteration=len(self.frames),
            label=label,
        )
        self.frames.append(frame)
        return frame

    def render_frame_svg(self, frame: VisualizationFrame) -> str:
        """Render a single frame as SVG string."""
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.svg_width}" height="{self.svg_height}" '
            f'viewBox="0 0 {self.svg_width} {self.svg_height}">'
        ]

        # Background
        lines.append(f'<rect width="100%" height="100%" fill="white"/>')

        # Board outline
        lines.append(self._render_board_outline())

        # Obstacles (components, keepouts)
        for obs in frame.obstacles:
            if obs.obstacle_type != "pad":
                lines.append(self._render_obstacle(obs))

        # Explored nodes (A* debug)
        if frame.explored_nodes:
            lines.append(self._render_explored_nodes(frame.explored_nodes))

        # Frontier nodes (A* debug)
        if frame.frontier_nodes:
            lines.append(self._render_frontier_nodes(frame.frontier_nodes))

        # Current path being explored
        if frame.current_path:
            lines.append(self._render_current_path(frame.current_path))

        # Completed traces
        for trace in frame.completed_traces:
            lines.append(self._render_trace(trace))

        # Completed vias
        for via in frame.completed_vias:
            lines.append(self._render_via(via))

        # Pads (render last so they're on top)
        for pad in frame.pads:
            lines.append(self._render_pad(pad))

        # Label
        if frame.label:
            lines.append(
                f'<text x="10" y="20" font-family="monospace" font-size="14">'
                f'{frame.label} (iter {frame.iteration})</text>'
            )
        if frame.current_net_name:
            lines.append(
                f'<text x="10" y="40" font-family="monospace" font-size="12">'
                f'Net: {frame.current_net_name}</text>'
            )

        lines.append('</svg>')
        return '\n'.join(lines)

    def _render_board_outline(self) -> str:
        """Render board outline rectangle."""
        x, y = self._to_svg_coords(self.bounds[0], self.bounds[1])
        w = self._to_svg_size(self.board_width)
        h = self._to_svg_size(self.board_height)
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="none" stroke="{get_routing_color("board_outline")}" stroke-width="2"/>'
        )

    def _render_obstacle(self, obs: Obstacle) -> str:
        """Render an obstacle as a rectangle."""
        x, y = self._to_svg_coords(obs.min_x, obs.min_y)
        w = self._to_svg_size(obs.max_x - obs.min_x)
        h = self._to_svg_size(obs.max_y - obs.min_y)
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{get_routing_color("obstacle")}" stroke="{get_routing_color("obstacle_stroke")}" '
            f'stroke-width="0.5" opacity="0.7"/>'
        )

    def _render_pad(self, pad: Obstacle) -> str:
        """Render a pad (connection point).

        Now supports N-layer boards with dynamic color generation.
        """
        x, y = self._to_svg_coords(pad.min_x, pad.min_y)
        w = self._to_svg_size(pad.max_x - pad.min_x)
        h = self._to_svg_size(pad.max_y - pad.min_y)
        color = get_layer_color(pad.layer, "pad")
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{color}" stroke="black" stroke-width="1" opacity="0.9"/>'
        )

    def _render_trace(self, trace: RouteSegment) -> str:
        """Render a routed trace segment.

        Now supports N-layer boards with dynamic color generation.
        """
        x1, y1 = self._to_svg_coords(trace.start[0], trace.start[1])
        x2, y2 = self._to_svg_coords(trace.end[0], trace.end[1])
        w = self._to_svg_size(trace.width)
        color = get_layer_color(trace.layer, "trace")
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{max(w, 1)}" '
            f'stroke-linecap="round"/>'
        )

    def _render_via(self, via: Via) -> str:
        """Render a via."""
        x, y = self._to_svg_coords(via.x, via.y)
        r = self._to_svg_size(via.pad_diameter / 2)
        drill_r = self._to_svg_size(via.drill_diameter / 2)
        return (
            f'<circle cx="{x}" cy="{y}" r="{r}" '
            f'fill="{get_routing_color("via")}" stroke="black" stroke-width="1"/>'
            f'<circle cx="{x}" cy="{y}" r="{drill_r}" fill="white"/>'
        )

    def _render_explored_nodes(self, nodes: Set[Tuple[float, float, int]]) -> str:
        """Render A* explored nodes as a heat map."""
        if not nodes:
            return ""
        circles = []
        r = self._to_svg_size(0.2)  # Small dot
        for nx, ny, layer in nodes:
            x, y = self._to_svg_coords(nx, ny)
            circles.append(
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{get_routing_color("explored")}" opacity="0.3"/>'
            )
        return f'<g class="explored">{"".join(circles)}</g>'

    def _render_frontier_nodes(self, nodes: Set[Tuple[float, float, int]]) -> str:
        """Render A* frontier nodes."""
        if not nodes:
            return ""
        circles = []
        r = self._to_svg_size(0.3)
        for nx, ny, layer in nodes:
            x, y = self._to_svg_coords(nx, ny)
            circles.append(
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{get_routing_color("frontier")}" '
                f'stroke="orange" stroke-width="0.5"/>'
            )
        return f'<g class="frontier">{"".join(circles)}</g>'

    def _render_current_path(self, path: List[Tuple[float, float, int]]) -> str:
        """Render the current path being explored."""
        if len(path) < 2:
            return ""
        points = []
        for px, py, layer in path:
            x, y = self._to_svg_coords(px, py)
            points.append(f"{x},{y}")
        return (
            f'<polyline points="{" ".join(points)}" '
            f'fill="none" stroke="{get_routing_color("current_path")}" '
            f'stroke-width="2" stroke-dasharray="5,3"/>'
        )

    def export_frame(self, frame: VisualizationFrame, filename: str = None):
        """Export a single frame as SVG file."""
        if filename is None:
            filename = f"frame_{frame.iteration:04d}.svg"
        path = self.output_dir / filename
        svg = self.render_frame_svg(frame)
        path.write_text(svg)
        logger.debug(f"Exported frame to {path}")
        return path

    def export_all_frames(self):
        """Export all captured frames."""
        for frame in self.frames:
            self.export_frame(frame)

    def export_html_report(self, filename: str = "routing_report.html"):
        """Generate interactive HTML report with all frames.

        .. deprecated::
            Use :meth:`export_svg_delta_html` instead for better performance
            and unified visualization experience.
        """
        warnings.warn(
            "export_html_report() is deprecated. Use export_svg_delta_html() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if not self.frames:
            logger.warning("No frames to export")
            return None

        path = self.output_dir / filename

        # Generate SVG data URIs for each frame
        frame_svgs = []
        for i, frame in enumerate(self.frames):
            svg = self.render_frame_svg(frame)
            frame_svgs.append({
                "index": i,
                "label": frame.label or f"Frame {i}",
                "net": frame.current_net_name,
                "svg": svg.replace('\n', ''),
            })

        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Routing Visualization Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .controls {{ margin: 20px 0; }}
        .controls button {{ margin-right: 10px; padding: 8px 16px; }}
        .frame-container {{ border: 1px solid #ccc; display: inline-block; }}
        .info {{ margin: 10px 0; font-family: monospace; }}
        #slider {{ width: 400px; }}
    </style>
</head>
<body>
    <h1>Routing Visualization Report</h1>
    <div class="info">
        <span>Frame: <span id="frame-num">0</span> / {len(self.frames) - 1}</span>
        <span style="margin-left: 20px;">Net: <span id="net-name">-</span></span>
    </div>
    <div class="controls">
        <button onclick="prevFrame()">&#9664; Prev</button>
        <button onclick="nextFrame()">Next &#9654;</button>
        <button onclick="togglePlay()" id="play-btn">&#9658; Play</button>
        <input type="range" id="slider" min="0" max="{len(self.frames) - 1}" value="0" onchange="setFrame(this.value)">
    </div>
    <div class="frame-container" id="frame-container"></div>

    <script>
        const frames = {frame_svgs};
        let currentFrame = 0;
        let playing = false;
        let playInterval = null;

        function showFrame(idx) {{
            currentFrame = Math.max(0, Math.min(frames.length - 1, idx));
            document.getElementById('frame-container').innerHTML = frames[currentFrame].svg;
            document.getElementById('frame-num').textContent = currentFrame;
            document.getElementById('net-name').textContent = frames[currentFrame].net || '-';
            document.getElementById('slider').value = currentFrame;
        }}

        function prevFrame() {{ showFrame(currentFrame - 1); }}
        function nextFrame() {{ showFrame(currentFrame + 1); }}
        function setFrame(idx) {{ showFrame(parseInt(idx)); }}

        function togglePlay() {{
            playing = !playing;
            document.getElementById('play-btn').textContent = playing ? '&#9724; Stop' : '&#9658; Play';
            if (playing) {{
                playInterval = setInterval(() => {{
                    if (currentFrame >= frames.length - 1) {{
                        togglePlay();
                    }} else {{
                        nextFrame();
                    }}
                }}, 200);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        // Initialize
        showFrame(0);

        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') prevFrame();
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
        }});
    </script>
</body>
</html>'''

        path.write_text(html)
        logger.info(f"Exported HTML report to {path}")
        return path

    def export_svg_delta_html(
        self,
        filename: str = "routing_debug.html",
        output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """Export routing visualization using SVG delta format.

        Uses the unified visualization system for better performance
        and consistent UI across placement and routing visualization.
        If a board was provided, renders components for context.

        Args:
            filename: Output filename
            output_dir: Output directory (defaults to self.output_dir)

        Returns:
            Path to generated HTML file, or None if no frames
        """
        if not self.frames:
            logger.warning("No frames to export")
            return None

        from ..visualization import get_svg_delta_viewer_js, get_styles_css
        from ..placement.viewer_template import generate_viewer_html_template

        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract static props and component data from board if available
        static_props_json = {}
        component_layers = {}
        netlist = {}
        initial_component_state = {}

        if self.board:
            from ..board.abstraction import Layer

            for ref, comp in self.board.components.items():
                # Extract pad geometry as tuples: [x, y, width, height, rotation, net]
                pad_geom = []
                for pad in comp.pads:
                    pad_rot = getattr(pad, 'rotation', 0.0) or 0.0
                    pad_geom.append([pad.x, pad.y, pad.width, pad.height, pad_rot, pad.net or ""])

                static_props_json[ref] = {
                    'width': comp.width,
                    'height': comp.height,
                    'pads': pad_geom
                }

                # Component layer
                if comp.layer == Layer.TOP_COPPER:
                    component_layers[ref] = 'top'
                elif comp.layer == Layer.BOTTOM_COPPER:
                    component_layers[ref] = 'bottom'
                else:
                    component_layers[ref] = 'top'

                # Initial position for first frame (array format: [x, y, rotation])
                initial_component_state[ref] = [comp.x, comp.y, comp.rotation]

            # Build netlist
            for ref, comp in self.board.components.items():
                if getattr(comp, 'dnp', False):
                    continue
                for pad_index, pad in enumerate(comp.pads):
                    net_name = (pad.net or "").strip()
                    if not net_name:
                        continue
                    netlist.setdefault(net_name, []).append([ref, pad_index])

            for net_name in sorted(netlist.keys()):
                netlist[net_name] = sorted(netlist[net_name])

        # Convert routing frames to delta format
        delta_frames = []
        for i, frame in enumerate(self.frames):
            # Convert traces to delta format
            traces = []
            for trace in frame.completed_traces:
                traces.append({
                    'start': [trace.start[0], trace.start[1]],
                    'end': [trace.end[0], trace.end[1]],
                    'layer': trace.layer,
                    'width': trace.width,
                    'net': str(trace.net_id) if trace.net_id else ''
                })

            # Convert vias to delta format
            vias = []
            for via in frame.completed_vias:
                vias.append({
                    'x': via.x,
                    'y': via.y,
                    'drill': via.drill_diameter,
                    'pad': via.pad_diameter,
                    'net': str(via.net_id) if via.net_id else ''
                })

            # Convert A* debug data
            astar_explored = list(frame.explored_nodes) if frame.explored_nodes else []
            astar_frontier = list(frame.frontier_nodes) if frame.frontier_nodes else []
            astar_path = frame.current_path if frame.current_path else []

            # First frame includes all component positions
            changed_components = {}
            if i == 0 and initial_component_state:
                changed_components = initial_component_state

            delta_frame = {
                'index': i,
                'label': frame.label or f"Frame {i}",
                'phase': frame.current_net_name or "Routing",
                'iteration': frame.iteration,
                'traces': traces,
                'vias': vias,
                'astar_explored': astar_explored,
                'astar_frontier': astar_frontier,
                'astar_path': astar_path,
                'changed_components': changed_components
            }
            delta_frames.append(delta_frame)

        # Calculate board bounds for JavaScript
        board_bounds = {
            'minX': self.bounds[0],
            'minY': self.bounds[1],
            'maxX': self.bounds[2],
            'maxY': self.bounds[3],
            'scale': self.scale,
            'padding': self.margin * self.scale
        }

        # Generate JavaScript with data
        js_code = get_svg_delta_viewer_js()
        data_js = f'''
// Board bounds for coordinate transforms
const boardBounds = {json.dumps(board_bounds)};

// Static properties for components
const staticProps = {json.dumps(static_props_json)};

// Delta frames
const deltaFrames = {json.dumps(delta_frames)};

// Total frames
const totalFrames = {len(delta_frames)};

// Module colors (empty for routing)
const moduleColors = {{}};

// Component layers
const componentLayers = {json.dumps(component_layers)};

// Netlist for ratsnest
const netlist = {json.dumps(netlist)};

{js_code}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {{
    showFrame(0);
    drawEnergyGraph();
}});
'''

        # Helper to transform coordinates
        def tx(x):
            return (x - self.bounds[0] + self.margin) * self.scale

        def ty(y):
            return (y - self.bounds[1] + self.margin) * self.scale

        def ts(size):
            return size * self.scale

        # Generate component SVG elements
        component_svg = []
        pad_svg = []
        label_svg = []

        if self.board:
            from ..board.abstraction import Layer

            for ref, comp in self.board.components.items():
                x, y, rotation = comp.x, comp.y, comp.rotation
                cx, cy = tx(x), ty(y)
                hw = ts(comp.width / 2)
                hh = ts(comp.height / 2)

                # Determine layer class
                layer_class = "comp-top"
                if comp.layer == Layer.BOTTOM_COPPER:
                    layer_class = "comp-bottom"

                # Component body
                component_svg.append(
                    f'<g class="component {layer_class}" data-ref="{ref}" '
                    f'transform="rotate({-rotation} {cx} {cy})">'
                    f'<rect x="{cx - hw}" y="{cy - hh}" '
                    f'width="{ts(comp.width)}" height="{ts(comp.height)}" '
                    f'fill="#2d5a3d" stroke="#1a1a1a" stroke-width="1" opacity="0.9"/>'
                    f'</g>'
                )

                # Component pads
                for pad_idx, pad in enumerate(comp.pads):
                    # Transform pad from component-relative to board coordinates
                    rad = math.radians(rotation)
                    cos_r, sin_r = math.cos(rad), math.sin(rad)
                    px = x + pad.x * cos_r - pad.y * sin_r
                    py = y + pad.x * sin_r + pad.y * cos_r

                    pad_cx, pad_cy = tx(px), ty(py)
                    pad_hw = ts(pad.width / 2)
                    pad_hh = ts(pad.height / 2)
                    pad_rot = rotation + (getattr(pad, 'rotation', 0) or 0)

                    pad_color = "#4a9" if comp.layer == Layer.TOP_COPPER else "#49a"

                    pad_svg.append(
                        f'<rect class="pad-element {layer_class}" data-ref="{ref}" data-pad="{pad_idx}" '
                        f'x="{pad_cx - pad_hw}" y="{pad_cy - pad_hh}" '
                        f'width="{ts(pad.width)}" height="{ts(pad.height)}" '
                        f'fill="{pad_color}" stroke="#1a1a1a" stroke-width="0.5" opacity="0.9" '
                        f'transform="rotate({-pad_rot} {pad_cx} {pad_cy})"/>'
                    )

                # Component label
                label_svg.append(
                    f'<text class="ref-label {layer_class}" data-ref="{ref}" '
                    f'x="{cx}" y="{cy}" '
                    f'text-anchor="middle" dominant-baseline="middle" '
                    f'font-size="8" fill="white" opacity="0.8">{ref}</text>'
                )

        components_group = '\n'.join(component_svg)
        pads_group = '\n'.join(pad_svg)
        labels_group = '\n'.join(label_svg)

        # Create initial SVG with board outline and components
        # Components are placed directly in SVG (not in a group) for JS compatibility
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg"
             id="routing-svg"
             width="{self.svg_width}" height="{self.svg_height}"
             viewBox="0 0 {self.svg_width} {self.svg_height}">
            <rect width="100%" height="100%" fill="#1a1a2e"/>
            <!-- Board outline -->
            <rect x="{self.margin * self.scale}"
                  y="{self.margin * self.scale}"
                  width="{self.board_width * self.scale}"
                  height="{self.board_height * self.scale}"
                  fill="none"
                  stroke="{get_routing_color('board_outline')}"
                  stroke-width="2"
                  class="board-outline"/>
            <!-- Dynamic content groups -->
            <g class="astar-debug-group"></g>
            <g class="ratsnest-group"></g>
            <g class="traces-group"></g>
            <!-- Components (direct children for JS compatibility) -->
            {components_group}
            <!-- Pads -->
            {pads_group}
            <!-- Vias group -->
            <g class="vias-group"></g>
            <!-- Labels -->
            {labels_group}
        </svg>'''

        # Generate HTML using shared template
        html = generate_viewer_html_template(
            title="Routing Visualization",
            static_content='<div id="svg-container">' + svg_content + '</div>',
            dynamic_content='',
            javascript_code=data_js,
            module_types={},
            total_frames=len(delta_frames),
            is_streaming=False
        )

        html_path = out_dir / filename
        html_path.write_text(html)
        logger.info(f"Exported SVG delta routing visualization to {html_path}")
        return html_path

    def clear_frames(self):
        """Clear all captured frames."""
        self.frames = []


def create_visualizer_from_board(board) -> RouteVisualizer:
    """Create a visualizer from a Board object.

    Args:
        board: Board instance from atoplace.board.abstraction

    Returns:
        RouteVisualizer configured for the board
    """
    # Get board bounds from outline or components
    bounds = None

    if board.outline:
        # Try different outline attributes
        if hasattr(board.outline, 'get_bounding_box'):
            bbox = board.outline.get_bounding_box()
            if bbox:
                bounds = bbox
        elif hasattr(board.outline, 'polygon') and board.outline.polygon:
            xs = [p[0] for p in board.outline.polygon]
            ys = [p[1] for p in board.outline.polygon]
            bounds = (min(xs), min(ys), max(xs), max(ys))
        elif hasattr(board.outline, 'points') and board.outline.points:
            xs = [p[0] for p in board.outline.points]
            ys = [p[1] for p in board.outline.points]
            bounds = (min(xs), min(ys), max(xs), max(ys))

    if not bounds:
        # Fall back to component bounds
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for comp in board.components.values():
            bbox = comp.get_bounding_box()
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
        bounds = (min_x - 5, min_y - 5, max_x + 5, max_y + 5)

    return RouteVisualizer(board_bounds=bounds, board=board)
