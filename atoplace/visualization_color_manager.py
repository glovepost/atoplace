"""Visualization color management.

Provides centralized access to visualization colors from configuration,
with fallback to programmatic color generation for undefined module types
and PCB layers.

Based on patterns.py architecture - external configuration for maintainability.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
import hashlib

logger = logging.getLogger(__name__)


class ColorManager:
    """Manages visualization colors from external configuration.

    Loads colors from visualization_colors.yaml and provides methods
    to retrieve colors for modules, forces, routing layers, etc.
    Falls back to programmatic generation for undefined types.
    """

    _instance: Optional["ColorManager"] = None
    _config: Dict = {}

    def __new__(cls):
        """Singleton pattern to avoid reloading config."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load color configuration from YAML file."""
        config_path = Path(__file__).parent / "visualization_colors.yaml"

        try:
            if not config_path.exists():
                logger.warning(
                    f"Visualization colors config not found at {config_path}, "
                    "using defaults"
                )
                self._config = self._get_default_config()
                return

            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}

            logger.debug(f"Loaded visualization colors from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load visualization colors: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default color configuration as fallback."""
        return {
            "module_colors": {
                "power_supply": "#e74c3c",
                "power": "#e74c3c",
                "microcontroller": "#3498db",
                "mcu": "#3498db",
                "rf_frontend": "#9b59b6",
                "rf": "#9b59b6",
                "sensor": "#2ecc71",
                "connector": "#f39c12",
                "crystal": "#1abc9c",
                "led": "#e91e63",
                "memory": "#00bcd4",
                "analog": "#ff5722",
                "digital": "#607d8b",
                "esd_protection": "#795548",
                "level_shifter": "#9c27b0",
                "accel": "#00e676",
                "imu": "#00e676",
                "gps": "#ffeb3b",
                "lte": "#ff9800",
                "default": "#95a5a6",
            },
            "force_colors": {
                "repulsion": "#e74c3c",
                "attraction": "#2ecc71",
                "boundary": "#3498db",
                "constraint": "#f39c12",
                "alignment": "#9b59b6",
            },
            "routing_colors": {
                "board_outline": "#333333",
                "obstacle": "#cccccc",
                "obstacle_stroke": "#999999",
                "layer_0_pad": "#cc0000",
                "layer_0_trace": "#ff6666",
                "layer_1_pad": "#0000cc",
                "layer_1_trace": "#6666ff",
                "via": "#00cc00",
                "explored": "#ffeeee",
                "frontier": "#ffffcc",
                "current_path": "#00ff00",
                "target_pad": "#ff00ff",
            },
            "color_generation": {
                "saturation": 70,
                "lightness": 50,
                "hash_seed": 137.508,
            },
        }

    def get_module_color(self, module_type: str) -> str:
        """Get color for a module type, generating one if not predefined.

        Uses a deterministic hash-based color generation for unknown module types,
        ensuring the same module name always gets the same color.

        Args:
            module_type: Module type name (e.g., "power", "accel", "rf")

        Returns:
            Hex color string (e.g., "#e74c3c")
        """
        if not module_type:
            return self._config.get("module_colors", {}).get("default", "#95a5a6")

        module_colors = self._config.get("module_colors", {})

        # Check for direct match
        if module_type in module_colors:
            return module_colors[module_type]

        # Check for case-insensitive match
        module_lower = module_type.lower()
        if module_lower in module_colors:
            return module_colors[module_lower]

        # Check for partial matches (e.g., "power_regulator" contains "power")
        for key, color in module_colors.items():
            if key != "default" and (key in module_lower or module_lower in key):
                return color

        # Generate a deterministic color based on hash of module name
        return self._generate_color_from_hash(module_type)

    def _generate_color_from_hash(self, text: str) -> str:
        """Generate a deterministic color from text using hash.

        Uses golden ratio to spread hues evenly and MD5 for determinism
        (avoiding Python's randomized hash()).

        Args:
            text: Text to hash (module type, net name, etc.)

        Returns:
            Hex color string
        """
        gen_config = self._config.get("color_generation", {})
        saturation = gen_config.get("saturation", 70)
        lightness = gen_config.get("lightness", 50)
        hash_seed = gen_config.get("hash_seed", 137.508)

        # Use MD5 for deterministic hashing across runs
        hash_bytes = hashlib.md5(text.encode("utf-8")).digest()
        hash_val = int.from_bytes(hash_bytes[:4], byteorder="little")

        # Generate hue from 0-360 using golden angle for good distribution
        hue = (hash_val * hash_seed) % 360

        # Convert HSL to RGB
        c = (1 - abs(2 * lightness / 100 - 1)) * saturation / 100
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = lightness / 100 - c / 2

        if hue < 60:
            r, g, b = c, x, 0
        elif hue < 120:
            r, g, b = x, c, 0
        elif hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        elif hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Convert to hex
        r_hex = int((r + m) * 255)
        g_hex = int((g + m) * 255)
        b_hex = int((b + m) * 255)

        return f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"

    def get_force_color(self, force_type: str) -> str:
        """Get color for a force type.

        Args:
            force_type: Force type ("repulsion", "attraction", "boundary", etc.)

        Returns:
            Hex color string
        """
        force_colors = self._config.get("force_colors", {})
        return force_colors.get(force_type, "#999999")

    def get_routing_color(self, element: str) -> str:
        """Get color for a routing element.

        Args:
            element: Element name (e.g., "board_outline", "obstacle", "via")

        Returns:
            Hex color string
        """
        routing_colors = self._config.get("routing_colors", {})
        return routing_colors.get(element, "#cccccc")

    def get_layer_color(self, layer: int, element_type: str = "pad") -> str:
        """Get color for a specific PCB layer.

        Supports multi-layer boards by programmatically generating colors
        for undefined layers using a spectral colormap.

        Args:
            layer: Layer number (0 = front, 1 = back, 2+ = inner)
            element_type: "pad" or "trace"

        Returns:
            Hex color string
        """
        routing_colors = self._config.get("routing_colors", {})
        key = f"layer_{layer}_{element_type}"

        # Check if explicitly defined
        if key in routing_colors:
            return routing_colors[key]

        # Generate color for undefined layer using spectral distribution
        return self._generate_layer_color(layer, element_type)

    def _generate_layer_color(self, layer: int, element_type: str) -> str:
        """Generate a color for a PCB layer using spectral colormap.

        Uses a rainbow-like distribution for good visual separation between layers.

        Args:
            layer: Layer number
            element_type: "pad" or "trace"

        Returns:
            Hex color string
        """
        # Use layer number to generate hue (0-360)
        # Spread across spectrum: 0° = red, 120° = green, 240° = blue
        hue = (layer * 360 / 10) % 360  # Cycle through spectrum every 10 layers

        # Pads are more saturated, traces are lighter
        if element_type == "pad":
            saturation = 80
            lightness = 40
        else:  # trace
            saturation = 100
            lightness = 70

        # Convert HSL to RGB
        c = (1 - abs(2 * lightness / 100 - 1)) * saturation / 100
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = lightness / 100 - c / 2

        if hue < 60:
            r, g, b = c, x, 0
        elif hue < 120:
            r, g, b = x, c, 0
        elif hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        elif hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Convert to hex
        r_hex = int((r + m) * 255)
        g_hex = int((g + m) * 255)
        b_hex = int((b + m) * 255)

        return f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"

    def get_all_module_colors(self) -> Dict[str, str]:
        """Get all defined module colors.

        Returns:
            Dictionary of module_type -> color
        """
        return self._config.get("module_colors", {}).copy()

    def get_all_force_colors(self) -> Dict[str, str]:
        """Get all defined force colors.

        Returns:
            Dictionary of force_type -> color
        """
        return self._config.get("force_colors", {}).copy()

    def get_all_routing_colors(self) -> Dict[str, str]:
        """Get all defined routing colors.

        Returns:
            Dictionary of element -> color
        """
        return self._config.get("routing_colors", {}).copy()

    def reload(self):
        """Reload configuration from file.

        Useful for testing or runtime config updates.
        """
        self._load_config()


# Global instance
_color_manager = None


def get_color_manager() -> ColorManager:
    """Get the global ColorManager instance.

    Returns:
        ColorManager singleton
    """
    global _color_manager
    if _color_manager is None:
        _color_manager = ColorManager()
    return _color_manager
