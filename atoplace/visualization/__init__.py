"""
Unified Visualization Module for atoplace.

This module consolidates all visualization functionality:
- SVG delta-based frame playback
- CSS styling
- HTML template generation

Assets are stored in visualization/assets/ and can be loaded either:
- Inline (bundled into single HTML file)
- External (separate .js/.css files)
"""

from pathlib import Path
from typing import Optional

# Asset directory path
ASSETS_DIR = Path(__file__).parent / "assets"


def get_asset_path(filename: str) -> Path:
    """Get the full path to an asset file."""
    return ASSETS_DIR / filename


def load_asset(filename: str) -> str:
    """Load an asset file as a string."""
    asset_path = get_asset_path(filename)
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset not found: {asset_path}")
    return asset_path.read_text()


def get_svg_delta_viewer_js() -> str:
    """Load the SVG delta viewer JavaScript."""
    return load_asset("svg-delta-viewer.js")


def get_styles_css() -> str:
    """Load the visualization CSS styles."""
    return load_asset("styles.css")


__all__ = [
    "ASSETS_DIR",
    "get_asset_path",
    "load_asset",
    "get_svg_delta_viewer_js",
    "get_styles_css",
]
