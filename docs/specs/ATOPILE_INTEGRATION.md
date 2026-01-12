# Atopile Integration Specification

## Overview

[atopile](https://atopile.io/) is a declarative language for electronics design that compiles `.ato` source files directly to KiCad board files. AtoPlace integrates with atopile to provide intelligent placement optimization within the atopile build workflow.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATOPILE BUILD PIPELINE                       │
│                                                                 │
│  .ato files  ──►  ato build  ──►  .kicad_pcb (unplaced)        │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                               │
│              │    AtoPlace     │                               │
│              │  Placement Hook │                               │
│              └────────┬────────┘                               │
│                       │                                         │
│                       ▼                                         │
│              .kicad_pcb (optimized placement)                   │
│                       │                                         │
│                       ▼                                         │
│              Continue build (routing, outputs)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Approach: Post-Build File Processing

Rather than embedding into atopile's compiler, AtoPlace operates on atopile's output files:

**Rationale:**
1. **Decoupled development** - No dependency on atopile internal APIs
2. **Version resilience** - Works with any atopile version that produces KiCad files
3. **Standalone operation** - Same tool works for pure KiCad projects
4. **Simpler maintenance** - Single adapter (KiCad) instead of two

**Data Sources:**
| Data | Source File | Purpose |
|------|-------------|---------|
| Components | `.kicad_pcb` | Footprints, pads, initial positions |
| Netlist | `.kicad_pcb` | Connectivity (nets already embedded) |
| Board outline | `.kicad_pcb` | Boundary constraints |
| Component values | `ato-lock.yaml` | Selected parts for module detection |
| Hierarchy | `.ato` files (optional) | Module grouping hints |

## Workflow Integration

### Option A: CLI Integration (MVP)
```bash
# Standard atopile build
ato build

# Run AtoPlace on the generated board
atoplace place elec/layout/default/project.kicad_pcb \
  --constraints "USB connector on left edge" \
  --dfm jlcpcb

# Continue with manual routing or Freerouting
```

### Option B: Build Hook (Future)
Configure in `ato.yaml`:
```yaml
builds:
  default:
    entry: elec/src/project.ato:MainModule
    hooks:
      post-netlist: atoplace place --auto
```

### Option C: MCP Server Integration (Future)
Claude interacts with both atopile and AtoPlace:
```
User: "Build my ESP32 board and optimize the placement"
Claude: [calls ato build] -> [calls atoplace place] -> [returns confidence report]
```

## Atopile-Specific Features

### Module Detection Enhancement
Atopile's hierarchical structure provides explicit module boundaries:

```python
# From .ato file structure, infer:
# - power_supply module -> group power components
# - sensor_interface module -> group sensor components
# - mcu_core module -> central MCU placement
```

**Implementation:**
```python
class AtopileModuleParser:
    """Parse .ato files to extract module hierarchy for grouping."""

    def parse_hierarchy(self, ato_file: Path) -> Dict[str, List[str]]:
        """Extract module->component mappings from .ato source."""
        # Parse module definitions
        # Map component instances to their parent modules
        # Return hierarchy for GroupingConstraint generation
```

### Component Value Extraction
The `ato-lock.yaml` contains selected component values:

```yaml
# Example ato-lock.yaml structure
components:
  C1:
    mpn: "GRM155R71C104KA88D"
    value: "100nF"
    package: "0402"
  U1:
    mpn: "ESP32-S3-WROOM-1"
    package: "QFN-56"
```

**Use for:**
- Decoupling capacitor identification (for proximity to ICs)
- Power component detection (inductors, bulk caps)
- Footprint size awareness for spacing

### Constraint Inference from Atopile
Certain atopile patterns imply placement constraints:

| Atopile Pattern | Inferred Constraint |
|-----------------|---------------------|
| `decoupling_cap ~ ic.vdd` | ProximityConstraint(cap, ic) |
| `crystal ~ mcu.xtal` | ProximityConstraint(crystal, mcu) + short trace |
| `usb_connector.dp ~ esd.io` | ProximityConstraint(esd, connector) |
| `power_input -> DCDCConverter` | Edge placement preference |

## File Handling

### Input Processing
```python
class AtopileProjectLoader:
    """Load an atopile project for placement optimization."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.ato_yaml = self._load_ato_yaml()
        self.lock_file = self._load_lock_file()

    def get_board_path(self, build_name: str = "default") -> Path:
        """Get path to generated KiCad board file."""
        entry = self.ato_yaml["builds"][build_name]["entry"]
        # elec/src/project.ato:Module -> elec/layout/default/project.kicad_pcb
        return self._entry_to_board_path(entry)

    def load_board(self, build_name: str = "default") -> Board:
        """Load board through KiCad adapter."""
        board_path = self.get_board_path(build_name)
        board = Board.from_kicad(board_path)

        # Enhance with atopile metadata
        self._apply_module_hierarchy(board)
        self._apply_component_values(board)

        return board
```

### Output Handling
Placement results write back to the same `.kicad_pcb` file:

```python
def save_placement(self, board: Board, build_name: str = "default"):
    """Save optimized placement back to KiCad file."""
    board_path = self.get_board_path(build_name)
    board.to_kicad(board_path)

    # Optionally update ato-lock.yaml with placement positions
    # (if atopile supports position persistence)
```

## CLI Extensions for Atopile

```bash
# Auto-detect atopile project
atoplace place .  # Detects ato.yaml, finds board

# Specify build target
atoplace place . --build my-variant

# Use atopile hierarchy for grouping
atoplace place . --use-ato-modules

# Full atopile workflow
atoplace ato-workflow . \
  --constraints "power on bottom edge" \
  --dfm jlcpcb \
  --route  # Also run Freerouting
```

## Implementation Plan

### Phase 2A: Basic Atopile Support
1. **AtopileProjectLoader** - Detect and parse `ato.yaml`
2. **Board path resolution** - Find `.kicad_pcb` from entry point
3. **CLI integration** - `atoplace place <ato-project-dir>`

### Phase 2B: Enhanced Metadata
1. **Lock file parser** - Extract component values from `ato-lock.yaml`
2. **Module hierarchy parser** - Parse `.ato` files for grouping
3. **Constraint inference** - Auto-generate constraints from patterns

### Phase 2C: Workflow Integration
1. **Build hook support** - Post-build placement trigger
2. **Position persistence** - Write positions to lock file (if supported)
3. **MCP integration** - Combined atopile+AtoPlace operations

## Testing Strategy

```python
# tests/test_atopile_integration.py

def test_project_detection():
    """Verify ato.yaml detection and parsing."""

def test_board_path_resolution():
    """Verify correct .kicad_pcb path from entry point."""

def test_module_hierarchy_extraction():
    """Verify .ato parsing for module grouping."""

def test_lock_file_parsing():
    """Verify component value extraction."""

def test_constraint_inference():
    """Verify auto-constraints from atopile patterns."""
```

## Example: ESP32 Sensor Board

```
# Project structure
my-sensor/
├── ato.yaml
├── ato-lock.yaml
├── elec/
│   ├── src/
│   │   ├── sensor.ato      # Main module
│   │   └── parts/
│   └── layout/
│       └── default/
│           └── sensor.kicad_pcb
```

```bash
# Workflow
cd my-sensor
ato build                    # Generate initial board
atoplace place . \
  --constraints "USB-C on left, sensors on top" \
  --use-ato-modules \
  --dfm jlcpcb
# Opens interactive session or applies and saves
```

