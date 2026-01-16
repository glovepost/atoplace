"""Shared HTML template for placement visualizers.

This module provides the common UI structure used by both Canvas export
and Streaming visualizations, matching the layout of placement_debug.html.

CSS styles are loaded from the consolidated visualization module at:
atoplace/visualization/assets/styles.css
"""

from typing import Dict, Optional

from atoplace.visualization import get_styles_css


def generate_viewer_html_template(
    title: str,
    static_content: str,
    dynamic_content: str,
    javascript_code: str,
    module_types: Optional[Dict[str, str]] = None,
    total_frames: Optional[int] = None,
    is_streaming: bool = False
) -> str:
    """Generate HTML with KiCad-style UI matching placement_debug.html.

    Args:
        title: Page title
        static_content: HTML content for static canvas/elements
        dynamic_content: HTML content for dynamic canvas/elements
        javascript_code: JavaScript code block
        module_types: Dict mapping module names to colors
        total_frames: Total number of frames (None for streaming)
        is_streaming: True if this is for streaming viewer

    Returns:
        Complete HTML string
    """
    # Load CSS from external file
    css_content = get_styles_css()

    module_types = module_types or {}

    # Generate module legend items
    module_legend_html = ""
    for module_name, color in module_types.items():
        safe_id = module_name.replace('.', '-').replace(' ', '-')
        module_legend_html += f'''
                <div class="layer-item">
                    <input type="checkbox" class="layer-checkbox" id="show-module-{safe_id}" checked onchange="updateModuleVisibility()">
                    <span class="color-swatch" style="background: {color};"></span>
                    <span class="layer-name">{module_name}</span>
                </div>'''

    # Frame count display
    frame_display = f'<span id="frame-num">0</span> / {total_frames}' if total_frames else '<span id="frame-num">0</span>'
    if is_streaming:
        frame_display = '<span id="frame-num">Live</span>'

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <style>
{css_content}
    </style>
</head>
<body>
    <div class="main-layout">
        <div class="content-area">
            {'<div class="status-bar" id="status-bar"><div class="status-indicator status-connecting" id="status-indicator"></div><span id="status-text">Connecting...</span></div>' if is_streaming else ''}

            <div class="info">
                <div class="info-item">
                    <span class="info-label">Frame</span>
                    <span class="info-value">{frame_display}</span>
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

            <div class="frame-container" id="frame-container">
                {static_content}
                {dynamic_content}
            </div>

            <div class="energy-graph" onclick="seekToFrame(event)" onmousedown="startSeeking(event)" onmousemove="continueSeeking(event)" onmouseup="stopSeeking()" onmouseleave="stopSeeking()">
                <canvas id="energy-canvas"></canvas>
                <div class="graph-labels">
                    <span>Energy (blue) / Wire Length (green) / Nets Routed (orange)</span>
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
                            <input type="checkbox" class="layer-checkbox" id="show-bottom" onchange="updateLayers()">
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
                        <div class="layer-item">
                            <input type="checkbox" class="layer-checkbox" id="show-astar-debug" onchange="updateLayers()">
                            <span class="layer-icon" style="color:#90ee90;">&#9733;</span>
                            <span class="layer-name">A* Debug</span>
                            <span class="layer-key">A</span>
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
                        {module_legend_html if module_legend_html else '<div style="padding: 10px; font-size: 10px; color: #666;">No modules detected</div>'}
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
    {javascript_code}

    // Panel toggle functionality
    function togglePanel() {{
        const panel = document.getElementById('side-panel');
        panel.classList.toggle('collapsed');
    }}

    // Section collapse/expand
    function toggleSection(header) {{
        header.classList.toggle('collapsed');
    }}

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch(e.key.toLowerCase()) {{
            case 'arrowleft': prevFrame(); break;
            case 'arrowright': nextFrame(); break;
            case ' ': e.preventDefault(); togglePlay(); break;
            case 'home': firstFrame(); break;
            case 'end': lastFrame(); break;
            case '1': document.getElementById('show-top').click(); break;
            case '2': document.getElementById('show-bottom').click(); break;
            case 'e': document.getElementById('show-edge').click(); break;
            case 'g': document.getElementById('show-grid').click(); break;
            case 'r': document.getElementById('show-ratsnest').click(); break;
            case 't': document.getElementById('show-trails').click(); break;
            case 'f': document.getElementById('show-forces').click(); break;
            case 'l': document.getElementById('show-labels').click(); break;
            case 'm': document.getElementById('show-groups').click(); break;
            case 'p': document.getElementById('show-pads').click(); break;
            case 'a': document.getElementById('show-astar-debug')?.click(); break;
            case '+': case '=': zoomIn(); break;
            case '-': case '_': zoomOut(); break;
            case '0': resetView(); break;
        }}
    }});
    </script>
</body>
</html>'''

    return html
