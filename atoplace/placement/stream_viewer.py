"""Generate HTML viewer for real-time WebSocket streaming.

Creates a self-contained HTML file that connects to the streaming server
and displays placement visualization in real-time with full UI features.
"""

from pathlib import Path
from typing import Dict, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def generate_stream_viewer_html(
    websocket_url: str = "ws://localhost:8765",
    board_bounds: Tuple[float, float, float, float] = (0, 0, 100, 100),
    static_props: Dict = None,
    module_colors: Dict[str, str] = None,
    output_path: Path = None,
) -> str:
    """Generate HTML viewer for WebSocket streaming with full UI features.

    Features matching placement_debug.html:
    - Real-time WebSocket streaming
    - Pan/zoom with mouse wheel and drag
    - Layer visibility controls (Grid, Ratsnest, Trails, Forces, Pads, Labels, Groups)
    - Module visibility toggles
    - Keyboard shortcuts (G/R/T/F/L/M/P, +/-/0)
    - Collapsible side panel
    - Connection status indicator

    Args:
        websocket_url: WebSocket server URL (e.g., ws://localhost:8765)
        board_bounds: (min_x, min_y, max_x, max_y) in mm
        static_props: Component static properties dict
        module_colors: Module name to color mapping
        output_path: Optional path to save HTML file

    Returns:
        HTML content string
    """
    from .viewer_template import generate_viewer_html_template
    from .viewer_javascript import generate_viewer_javascript, generate_canvas_render_javascript

    min_x, min_y, max_x, max_y = board_bounds
    static_props = static_props or {}
    module_colors = module_colors or {}

    # Calculate canvas dimensions
    board_width = max_x - min_x
    board_height = max_y - min_y
    canvas_scale = min(1000 / board_width, 800 / board_height) if board_width > 0 and board_height > 0 else 10.0
    canvas_width = int(board_width * canvas_scale + 100)
    canvas_height = int(board_height * canvas_scale + 100)

    # Board bounds for JavaScript
    board_bounds_dict = {
        'minX': min_x,
        'minY': min_y,
        'maxX': max_x,
        'maxY': max_y
    }

    # Generate canvases content
    static_content = f'<canvas id="staticCanvas" width="{canvas_width}" height="{canvas_height}"></canvas>'
    dynamic_content = f'<canvas id="dynamicCanvas" width="{canvas_width}" height="{canvas_height}"></canvas>'

    # Generate JavaScript code for streaming
    javascript_code = f'''
// ============================================================================
// Streaming Configuration
// ============================================================================
const WEBSOCKET_URL = "{websocket_url}";
const boardBounds = {json.dumps(board_bounds_dict)};
const moduleColors = {json.dumps(module_colors)};
const staticProps = {json.dumps(static_props)};

// Streaming state
let ws = null;
let totalFrames = 0;
let frames = [];
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 2000;

// ============================================================================
// WebSocket Connection
// ============================================================================
function connect() {{
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    statusIndicator.className = 'status-indicator status-connecting';
    statusText.textContent = 'Connecting...';

    try {{
        ws = new WebSocket(WEBSOCKET_URL);

        ws.onopen = () => {{
            console.log('WebSocket connected');
            statusIndicator.className = 'status-indicator status-connected';
            statusText.textContent = 'Connected';
            reconnectAttempts = 0;
        }};

        ws.onmessage = (event) => {{
            const frame = JSON.parse(event.data);
            handleFrame(frame);
        }};

        ws.onerror = (error) => {{
            console.error('WebSocket error:', error);
            statusIndicator.className = 'status-indicator status-disconnected';
            statusText.textContent = 'Connection error';
        }};

        ws.onclose = () => {{
            console.log('WebSocket closed');
            statusIndicator.className = 'status-indicator status-disconnected';
            statusText.textContent = 'Disconnected';

            // Attempt reconnection
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {{
                reconnectAttempts++;
                statusText.textContent = `Reconnecting (attempt ${{reconnectAttempts}}/${{MAX_RECONNECT_ATTEMPTS}})...`;
                setTimeout(connect, RECONNECT_DELAY);
            }} else {{
                statusText.textContent = 'Connection failed';
            }}
        }};
    }} catch (error) {{
        console.error('Failed to create WebSocket:', error);
        statusIndicator.className = 'status-indicator status-disconnected';
        statusText.textContent = 'Connection failed';
    }}
}}

// ============================================================================
// Frame Handling
// ============================================================================
function handleFrame(frame) {{
    // Store frame
    frames.push(frame);
    totalFrames = frames.length;
    currentFrame = frames.length - 1;

    // Update info bar
    updateInfoBar(frame);

    // Render frame
    if (typeof renderFrame === 'function') {{
        renderFrame(currentFrame);
    }}
}}

// Override showFrame for streaming (no navigation controls active during streaming)
const originalShowFrame = showFrame;
function showFrame(frameIndex) {{
    // In streaming mode, frames might not exist yet
    if (frameIndex >= 0 && frameIndex < frames.length) {{
        originalShowFrame(frameIndex);
    }}
}}

{generate_viewer_javascript(has_energy_graph=False, has_layer_controls=True, is_streaming=True)}

{generate_canvas_render_javascript()}

// ============================================================================
// Initialization
// ============================================================================
// Start connection when page loads
connect();

// Initialize module visibility
initializeModuleVisibility();
'''

    # Generate HTML using template
    html_content = generate_viewer_html_template(
        title="Real-Time Placement Viewer",
        static_content=static_content,
        dynamic_content=dynamic_content,
        javascript_code=javascript_code,
        module_types=module_colors,
        total_frames=None,  # Streaming, so no fixed total
        is_streaming=True
    )

    # Save to file if path provided
    if output_path:
        output_path.write_text(html_content)
        logger.info(f"Generated stream viewer HTML at {output_path}")

    return html_content
