# Custom Component Patterns

AtoPlace uses pattern matching to classify components and nets for intelligent placement and validation. By default, patterns are loaded from `atoplace/component_patterns.yaml`, but you can provide your own custom patterns to match your specific components and naming conventions.

## Why Customize Patterns?

The default patterns cover common components (STM32 MCUs, USB controllers, etc.), but your designs may use:
- Different component manufacturers or part numbers
- Custom naming conventions
- Specialized components not in the default list

By customizing patterns, you ensure AtoPlace correctly:
- Groups analog vs. digital components for separation constraints
- Identifies high-speed ICs requiring close decoupling
- Detects high-speed nets needing impedance control
- Applies appropriate validation rules

## Pattern Categories

### Analog Components
Components typically sensitive to noise (op-amps, ADCs, DACs, voltage references).
Used for "group analog components" and "separate analog from digital" constraints.

### Digital Components
Microcontrollers, logic ICs, programmable logic, memory.
Used for digital component grouping and separation.

### High-Speed ICs
Components requiring very close decoupling (<2mm recommended).
Examples: USB, Ethernet, RF, high-speed memory interfaces.

### Medium-Speed ICs
Standard digital ICs requiring close decoupling (<5mm recommended).
Examples: MCUs, SPI/I2C controllers.

### High-Speed Nets
Net names indicating signals requiring special routing (impedance control, length matching).
Examples: USB data lines, differential pairs, high-speed clocks.

### Differential Pair Suffixes
Suffixes indicating differential signals (e.g., `_P`, `_N`, `+`, `-`).
Used to identify differential pairs that need matched routing.

## Creating a Custom Pattern File

1. Copy the default pattern file as a starting point:
```bash
cp atoplace/component_patterns.yaml my_patterns.yaml
```

2. Edit `my_patterns.yaml` to add your patterns:
```yaml
# Add your custom MCU to medium-speed patterns
medium_speed_ics:
  - 'STM32'
  - 'ESP32'
  - 'MY_CUSTOM_MCU'  # Your custom MCU

# Add custom analog component patterns
analog_components:
  - 'OPA'
  - 'LM358'
  - 'MY_ANALOG_IC'  # Your custom analog IC
```

3. Use your custom patterns in code:

```python
from atoplace.nlp.constraint_parser import ConstraintParser
from atoplace.validation.confidence import ConfidenceScorer

# Use custom patterns with parser
parser = ConstraintParser(board, patterns_config="my_patterns.yaml")

# Use custom patterns with confidence scorer
scorer = ConfidenceScorer(patterns_config="my_patterns.yaml")
```

## Pattern Matching Rules

Patterns are matched using Python regex against:
- Component **value** field (e.g., "STM32F405")
- Component **footprint** field (e.g., "LQFP-64")
- Net **names** (for high-speed net detection)

Matching is **case-insensitive** and uses **substring matching**, so:
- Pattern `'USB'` matches: `USB_DP`, `USB2_DN`, `USB_VBUS`
- Pattern `'STM32'` matches: `STM32F405`, `STM32L4`, etc.

For differential pairs, suffixes must appear **at the end** of the net name.

## Decoupling Distance Thresholds

You can customize distance requirements for decoupling capacitors:

```yaml
decoupling_distances:
  high_speed:
    critical: 2.0   # mm - high-speed needs very close decoupling
    warning: 3.0
    info: 5.0

  medium_speed:
    critical: 5.0   # mm - standard digital
    warning: 7.0
    info: 10.0

  standard:
    critical: 10.0  # mm - low-speed/unknown
    warning: 15.0
    info: 20.0
```

These thresholds are based on trace inductance (~1nH per mm) and how it affects PDN performance at different frequencies.

## Example: Adding a Custom RF Module

Suppose you have a custom RF module `ACME_RF2400`:

```yaml
# Add to high-speed ICs (requires close decoupling)
high_speed_ics:
  - 'RF'
  - 'WIFI'
  - 'ACME_RF2400'  # Your custom RF module

# Add related net patterns
high_speed_nets:
  - 'RF'
  - 'ANT'  # Antenna traces
  - 'ACME_RF'  # Your RF signal prefix
```

Now:
- AtoPlace will require decoupling caps within 2mm of your RF module
- Nets containing `ACME_RF` will be flagged for impedance control
- "Group RF components" commands will find your module

## Default Pattern File Location

The default patterns are located at:
```
atoplace/component_patterns.yaml
```

You can modify this file directly if you want to change defaults for all projects.

## Validating Your Pattern File

Your custom YAML file must include all required sections:
- `analog_components`
- `digital_components`
- `high_speed_ics`
- `medium_speed_ics`
- `high_speed_nets`
- `differential_pair_suffixes`
- `decoupling_distances`

If any section is missing, AtoPlace will raise a `ValueError` with details about what's missing.

## Examples of Good Patterns

### Manufacturer-Specific
```yaml
analog_components:
  - 'AD8'    # Analog Devices op-amps (AD8605, AD8672, etc.)
  - 'LTC'    # Linear Technology analog ICs
  - 'MAX'    # Maxim analog ICs
```

### Function-Specific
```yaml
high_speed_nets:
  - 'CLK'    # All clock signals
  - 'MCLK'   # Master clocks
  - 'QSPI'   # Quad SPI signals
```

### Project-Specific Naming
If your project uses custom prefixes like `PROJ_USB_DP`:
```yaml
high_speed_nets:
  - 'PROJ_USB'   # Will match PROJ_USB_DP, PROJ_USB_DN, etc.
```

## Tips

1. **Be specific**: Pattern `'USB'` is better than `'U'` to avoid false positives
2. **Test incrementally**: Add one pattern, test, verify it works
3. **Document your patterns**: Add comments explaining why each pattern exists
4. **Share patterns**: Contribute commonly-used patterns back to the project
5. **Use regex sparingly**: Simple substring matching is usually sufficient and more maintainable

## Contributing Patterns

If you create patterns for common components, please contribute them! Open a PR adding them to the default `component_patterns.yaml` with:
- Clear comments explaining each pattern
- Examples of components that match
- Links to datasheets or manufacturer pages where helpful
