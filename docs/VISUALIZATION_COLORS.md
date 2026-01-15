# Visualization Color Configuration

AtoPlace's visualization system supports customizable colors for better accessibility and personal preference.

## Overview

Colors for placement and routing visualizations are defined in `atoplace/visualization_colors.yaml`. This allows you to:

- Customize colors for color blindness or personal preference
- Define colors for specific module types (power, RF, sensors, etc.)
- Configure colors for multi-layer PCBs (4, 6, 8+ layers)
- Adjust force vector colors for placement debugging

## Configuration File

The configuration file is located at `atoplace/visualization_colors.yaml` and contains four main sections:

### 1. Module Colors

Used in placement visualization to color-code functional groups of components:

```yaml
module_colors:
  power_supply: "#e74c3c"      # Red
  microcontroller: "#3498db"   # Blue
  rf_frontend: "#9b59b6"       # Purple
  sensor: "#2ecc71"            # Green
  # ... more types
```

**Unknown module types** are automatically assigned a deterministic color using a hash-based algorithm, ensuring consistency across runs.

### 2. Force Colors

Used in placement visualization to show different types of forces:

```yaml
force_colors:
  repulsion: "#e74c3c"    # Red - components pushing apart
  attraction: "#2ecc71"   # Green - components pulling together
  boundary: "#3498db"     # Blue - board edge forces
  constraint: "#f39c12"   # Orange - user constraints
  alignment: "#9b59b6"    # Purple - grid alignment forces
```

### 3. Routing Colors

Used in routing visualization for traces, pads, and debugging:

```yaml
routing_colors:
  board_outline: "#333333"
  obstacle: "#cccccc"
  via: "#00cc00"

  # Layer colors (expandable for N-layer boards)
  layer_0_pad: "#cc0000"     # Front copper pads
  layer_0_trace: "#ff6666"   # Front copper traces
  layer_1_pad: "#0000cc"     # Back copper pads
  layer_1_trace: "#6666ff"   # Back copper traces

  # Inner layers (2-9) with spectral colormap
  layer_2_pad: "#00cc00"     # Green
  layer_2_trace: "#66ff66"   # Light green
  # ... up to layer 9
```

**Multi-layer support**: Layers beyond those defined in the config file are automatically assigned colors using a spectral colormap that distributes hues evenly across the rainbow.

### 4. Color Generation Settings

Controls automatic color generation for undefined types:

```yaml
color_generation:
  saturation: 70      # 0-100
  lightness: 50       # 0-100
  hash_seed: 137.508  # Golden angle for even distribution
```

## Customization

### Adding New Module Types

To add a color for a new module type:

1. Edit `atoplace/visualization_colors.yaml`
2. Add an entry under `module_colors`:

```yaml
module_colors:
  my_custom_module: "#ff0000"  # Your color
```

3. Save the file - changes take effect immediately

### Supporting More PCB Layers

To define explicit colors for additional layers:

1. Edit `atoplace/visualization_colors.yaml`
2. Add entries under `routing_colors`:

```yaml
routing_colors:
  layer_10_pad: "#your_color"
  layer_10_trace: "#your_lighter_color"
```

**Note**: Undefined layers automatically get colors from the spectral colormap, so explicit definition is optional.

### Color Blindness Support

For color blindness accessibility, consider these palettes:

**Deuteranopia-friendly** (red-green color blindness):
- Use blue/orange instead of red/green
- Example: `repulsion: "#0066cc"` (blue), `attraction: "#ff9900"` (orange)

**Protanopia-friendly**:
- Similar to deuteranopia adjustments
- Avoid red hues, use blue/yellow/orange

**Monochromacy-friendly**:
- Use different saturations/brightness levels
- Combine with patterns (dashed lines, different shapes)

## Programmatic Access

### Python API

```python
from atoplace.visualization_color_manager import get_color_manager

# Get color manager
cm = get_color_manager()

# Module colors
color = cm.get_module_color("power")
color = cm.get_module_color("custom_type")  # Auto-generates if undefined

# Force colors
color = cm.get_force_color("repulsion")

# Routing colors
color = cm.get_routing_color("board_outline")
color = cm.get_routing_color("via")

# Layer colors (supports N layers)
color = cm.get_layer_color(0, "pad")     # Front pad
color = cm.get_layer_color(5, "trace")   # Inner layer 5 trace
color = cm.get_layer_color(15, "pad")    # Auto-generated for layer 15

# Reload config (useful for testing)
cm.reload()
```

### Helper Functions

Convenience functions are available in visualizer modules:

```python
# In placement/visualizer.py
from atoplace.placement.visualizer import get_module_color, get_force_color

color = get_module_color("rf")
color = get_force_color("attraction")

# In routing/visualizer.py
from atoplace.routing.visualizer import get_routing_color, get_layer_color

color = get_routing_color("via")
color = get_layer_color(3, "trace")
```

## Technical Details

### Color Generation Algorithm

For undefined module types and layers, colors are generated using:

1. **MD5 hashing** of the type name for deterministic color assignment
2. **Golden angle (137.508°)** for even hue distribution
3. **HSL color space** for consistent saturation and lightness
4. **Automatic conversion** to hex RGB format

This ensures:
- Same type always gets the same color
- Different types get visually distinct colors
- Colors are vibrant and easy to distinguish

### Layer Color Distribution

Multi-layer boards use a spectral colormap:
- **0° (red)** → **360° (red)** cycling through rainbow
- Each layer offset by 36° for visual separation
- Pads are more saturated (darker) than traces

### Backward Compatibility

Legacy code using the old `MODULE_COLORS` and `FORCE_COLORS` dictionaries continues to work through deprecated property accessors that load from the YAML config.

## Examples

### High-Contrast Theme

For better visibility on bright screens:

```yaml
module_colors:
  power: "#cc0000"           # Darker red
  microcontroller: "#0000cc" # Darker blue
  sensor: "#00cc00"          # Darker green

routing_colors:
  board_outline: "#000000"   # Black outline
  obstacle: "#999999"        # Darker grey
```

### Pastel Theme

For reduced eye strain:

```yaml
module_colors:
  power: "#ffb3b3"           # Light red
  microcontroller: "#b3d9ff" # Light blue
  sensor: "#b3ffb3"          # Light green

routing_colors:
  layer_0_pad: "#ffcccc"     # Light red pads
  layer_0_trace: "#ffe6e6"   # Very light red traces
```

### Dark Mode

For dark backgrounds:

```yaml
routing_colors:
  board_outline: "#ffffff"   # White outline
  obstacle: "#444444"        # Dark grey obstacles
  layer_0_trace: "#ff9999"   # Brighter traces
```

## Troubleshooting

### Colors Not Updating

If color changes don't appear:

1. Check that `atoplace/visualization_colors.yaml` exists
2. Verify YAML syntax (use a YAML validator)
3. Restart your Python process to clear cached config
4. Check logs for "Failed to load visualization colors" errors

### Missing Colors

If a color shows as grey or wrong:

1. Check spelling of module/force/element names
2. Ensure hex colors are in `"#RRGGBB"` format (quoted)
3. Verify YAML indentation (2 spaces, not tabs)

### Performance Issues

The color manager is a singleton - config is loaded once per process. If you modify the config file at runtime, call `cm.reload()` to refresh.

## Migration from Hardcoded Colors

Previous versions had colors hardcoded in Python files. The new system:

- **Moves colors to YAML** for easy customization
- **Preserves all original colors** as defaults
- **Maintains backward compatibility** for existing code
- **Adds new features** like N-layer support and color generation

No code changes are required for existing usage - the new system is a drop-in replacement.
