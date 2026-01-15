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


@dataclass
class Via:
    """A via connecting layers."""
    x: float
    y: float
    drill_diameter: float
    pad_diameter: float
    net_id: Optional[int] = None


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
    """

    def __init__(
        self,
        board_bounds: Tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        output_dir: Optional[Path] = None,
        scale: float = 10.0,  # pixels per mm
        margin: float = 5.0,  # mm margin around board
    ):
        """
        Args:
            board_bounds: Board bounding box (min_x, min_y, max_x, max_y)
            output_dir: Directory for output files
            scale: Rendering scale (pixels per mm)
            margin: Margin around board in mm
        """
        self.bounds = board_bounds
        self.output_dir = output_dir or Path("./route_debug")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self.margin = margin

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
        """Generate interactive HTML report with all frames."""
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

    return RouteVisualizer(board_bounds=bounds)
