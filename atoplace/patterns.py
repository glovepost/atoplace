"""
Component and Net Classification Patterns

Loads and manages classification patterns from configuration file.
Allows users to customize component detection without modifying code.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


class ComponentPatterns:
    """
    Manager for component and net classification patterns.

    Loads patterns from component_patterns.yaml by default, but allows
    users to provide custom configuration files.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pattern manager.

        Args:
            config_path: Optional path to custom patterns YAML file.
                        If None, uses default component_patterns.yaml.
        """
        if config_path is None:
            # Use default config file in same directory as this module
            config_path = Path(__file__).parent / "component_patterns.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Load patterns from YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Pattern configuration file not found: {self.config_path}"
            )

        # Security: Check for symlinks to prevent reading unintended files
        if self.config_path.is_symlink():
            raise ValueError(
                f"Pattern configuration file cannot be a symlink: {self.config_path}"
            )

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Validate required sections
        required_sections = [
            # Module detection patterns
            'microcontroller_patterns',
            'rf_patterns',
            'power_regulator_patterns',
            'sensor_patterns',
            'esd_patterns',
            'connector_patterns',
            'crystal_patterns',
            # Constraint parser patterns
            'analog_components',
            'digital_components',
            'high_speed_ics',
            'medium_speed_ics',
            'high_speed_nets',
            'differential_pair_suffixes',
            'decoupling_distances',
        ]

        missing = [s for s in required_sections if s not in self._config]
        if missing:
            raise ValueError(
                f"Configuration file missing required sections: {missing}"
            )

    # ==========================================================================
    # Module Detection Patterns (used by module_detector.py)
    # ==========================================================================

    @property
    def microcontroller_patterns(self) -> List[str]:
        """Get microcontroller IC patterns."""
        return self._config.get('microcontroller_patterns', [])

    @property
    def rf_patterns(self) -> List[str]:
        """Get RF/wireless component patterns."""
        return self._config.get('rf_patterns', [])

    @property
    def power_regulator_patterns(self) -> List[str]:
        """Get power regulator IC patterns."""
        return self._config.get('power_regulator_patterns', [])

    @property
    def sensor_patterns(self) -> List[str]:
        """Get sensor IC patterns."""
        return self._config.get('sensor_patterns', [])

    @property
    def esd_patterns(self) -> List[str]:
        """Get ESD protection device patterns."""
        return self._config.get('esd_patterns', [])

    @property
    def connector_patterns(self) -> List[str]:
        """Get connector component patterns."""
        return self._config.get('connector_patterns', [])

    @property
    def crystal_patterns(self) -> List[str]:
        """Get crystal/oscillator patterns."""
        return self._config.get('crystal_patterns', [])

    @property
    def opamp_patterns(self) -> List[str]:
        """Get op-amp patterns (subset of analog_components)."""
        # Op-amp patterns are the first 5 entries in analog_components
        return self._config.get('analog_components', [])[:5]

    def get_module_patterns(self) -> dict:
        """
        Get all module detection patterns as a dictionary.

        Returns:
            Dictionary mapping module type to list of regex patterns.
            Keys: 'microcontroller', 'rf', 'power_regulator', 'sensor',
                  'esd', 'opamp', 'connector', 'crystal'
        """
        return {
            'microcontroller': self.microcontroller_patterns,
            'rf': self.rf_patterns,
            'power_regulator': self.power_regulator_patterns,
            'sensor': self.sensor_patterns,
            'esd': self.esd_patterns,
            'opamp': self.opamp_patterns,
            'connector': self.connector_patterns,
            'crystal': self.crystal_patterns,
        }

    # ==========================================================================
    # Constraint Parser Patterns (used by constraint_parser.py)
    # ==========================================================================

    @property
    def analog_patterns(self) -> List[str]:
        """Get analog component patterns."""
        return self._config.get('analog_components', [])

    @property
    def digital_patterns(self) -> List[str]:
        """Get digital component patterns."""
        return self._config.get('digital_components', [])

    @property
    def high_speed_ic_patterns(self) -> List[str]:
        """Get high-speed IC patterns."""
        return self._config.get('high_speed_ics', [])

    @property
    def medium_speed_ic_patterns(self) -> List[str]:
        """Get medium-speed IC patterns."""
        return self._config.get('medium_speed_ics', [])

    @property
    def high_speed_net_patterns(self) -> List[str]:
        """Get high-speed net patterns."""
        return self._config.get('high_speed_nets', [])

    @property
    def differential_pair_suffixes(self) -> List[str]:
        """Get differential pair suffixes."""
        return self._config.get('differential_pair_suffixes', [])

    def get_decoupling_distances(self, speed_class: str) -> Dict[str, float]:
        """
        Get decoupling distance thresholds for a speed class.

        Args:
            speed_class: One of 'high_speed', 'medium_speed', 'standard'

        Returns:
            Dictionary with 'critical', 'warning', 'info' distance values (mm)

        Raises:
            ValueError: If speed_class is not recognized
        """
        distances = self._config.get('decoupling_distances', {})

        if speed_class not in distances:
            raise ValueError(
                f"Unknown speed class: {speed_class}. "
                f"Valid options: {list(distances.keys())}"
            )

        return distances[speed_class]

    def reload(self):
        """Reload configuration from file (useful during development)."""
        self._load_config()


# Global instance for convenience
_default_patterns: Optional[ComponentPatterns] = None


def get_patterns(config_path: Optional[str] = None) -> ComponentPatterns:
    """
    Get component patterns instance.

    Args:
        config_path: Optional path to custom patterns file.
                    If None, uses cached default instance.

    Returns:
        ComponentPatterns instance
    """
    global _default_patterns

    if config_path is not None:
        # Custom config path - create new instance
        return ComponentPatterns(config_path)

    # Use cached default instance
    if _default_patterns is None:
        _default_patterns = ComponentPatterns()

    return _default_patterns


def reload_patterns():
    """Reload default patterns from configuration file."""
    global _default_patterns
    if _default_patterns is not None:
        _default_patterns.reload()
