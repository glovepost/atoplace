"""SVG Delta Viewer - JavaScript loader for SVG-based delta playback.

Loads JavaScript from the consolidated visualization assets module.
The JavaScript updates SVG DOM elements directly instead of replacing
entire SVG strings, providing:
- Crisp vector quality at any zoom level
- Better performance than full SVG replacement (30-40 FPS vs 6 FPS)
- Smaller file sizes with delta compression
"""

from atoplace.visualization import get_svg_delta_viewer_js


def generate_svg_delta_viewer_js() -> str:
    """Load JavaScript for SVG delta playback viewer.

    This approach updates SVG element attributes directly via DOM manipulation
    instead of replacing the entire SVG on each frame change.

    The JavaScript is loaded from atoplace/visualization/assets/svg-delta-viewer.js
    """
    return get_svg_delta_viewer_js()
