# AtoPlace

AI-Powered PCB Placement and Routing Tool

## Overview

AtoPlace is an intelligent PCB layout tool that uses natural language understanding and physics-based optimization to automate component placement and routing. It integrates with KiCad and atopile to accelerate professional EE workflows.

## Features

- **Natural Language Constraints**: Describe placement requirements in plain English
  - "Keep C1 close to U1"
  - "USB connector on left edge"
  - "Separate analog and digital sections"

- **Intelligent Placement**: Force-directed algorithm with:
  - Module detection (power, RF, digital, analog)
  - Connectivity-aware optimization
  - Constraint satisfaction

- **Confidence Scoring**: Automatic quality assessment
  - Placement validation
  - DFM rule checking
  - Human review flagging

- **Multiple DFM Profiles**: Pre-configured rules for:
  - JLCPCB Standard/Advanced
  - OSH Park
  - PCBWay

## Installation

```bash
pip install atoplace
```

**Note:** AtoPlace requires KiCad's Python API (pcbnew). Run with KiCad's Python:

```bash
# macOS
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install atoplace

# Then run
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m atoplace place board.kicad_pcb
```

## Quick Start

### Command Line

```bash
# Run placement optimization
atoplace place board.kicad_pcb

# With constraints
atoplace place board.kicad_pcb --constraints "USB on left edge, antenna in corner"

# Validate existing placement
atoplace validate board.kicad_pcb --dfm jlcpcb_standard

# Interactive session
atoplace interactive board.kicad_pcb
```

### Python API

```python
from atoplace.board import Board
from atoplace.placement import ForceDirectedRefiner
from atoplace.validation import ConfidenceScorer
from atoplace.nlp import ConstraintParser

# Load board
board = Board.from_kicad("my_board.kicad_pcb")

# Parse natural language constraints
parser = ConstraintParser(board)
constraints, summary = parser.parse_interactive(
    "Keep decoupling caps close to MCU, USB on left edge"
)
print(summary)

# Run placement refinement
refiner = ForceDirectedRefiner(board)
for constraint in constraints:
    refiner.add_constraint(constraint)
result = refiner.refine()

# Validate result
scorer = ConfidenceScorer()
report = scorer.assess(board)
print(report.summary())

# Save
board.to_kicad("my_board_placed.kicad_pcb")
```

## Architecture

```
atoplace/
├── board/          # Board abstraction layer
├── placement/      # Placement algorithms
│   ├── force_directed.py   # Physics-based optimization
│   ├── module_detector.py  # Functional module detection
│   └── constraints.py      # Constraint definitions
├── routing/        # Routing integration (Freerouting)
├── validation/     # Quality checks
│   ├── confidence.py       # Confidence scoring
│   ├── pre_route.py        # Pre-routing validation
│   └── drc.py              # DRC checking
├── dfm/            # Design for Manufacturing
│   └── profiles.py         # Fab-specific rules
├── nlp/            # Natural language processing
│   └── constraint_parser.py
├── output/         # Manufacturing outputs
└── cli.py          # Command-line interface
```

## Constraint Types

| Type | Example | Description |
|------|---------|-------------|
| Proximity | "Keep C1 close to U1" | Minimize distance between components |
| Edge | "J1 on left edge" | Place component on board edge |
| Zone | "Analog section in top-left" | Restrict components to area |
| Grouping | "Group all capacitors" | Keep components together |
| Separation | "Separate analog and digital" | Maintain distance between groups |
| Fixed | "U1 at (50, 30)" | Lock component position |

## DFM Profiles

```python
from atoplace.dfm import get_profile, list_profiles

# List available profiles
print(list_profiles())
# ['jlcpcb_standard', 'jlcpcb_standard_4layer', 'jlcpcb_advanced', ...]

# Get specific profile
profile = get_profile("jlcpcb_standard")
print(f"Min trace: {profile.min_trace_width}mm")
```

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/atoplace/atoplace.git
cd atoplace
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black atoplace/
ruff check atoplace/
```

## Roadmap

- [x] Force-directed placement refinement
- [x] Natural language constraint parsing
- [x] Confidence scoring system
- [x] DFM profile support
- [ ] Freerouting integration
- [ ] OrthoRoute cloud integration
- [ ] Signal integrity checks
- [ ] MCP server for Claude integration

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [atopile](https://atopile.io) - Declarative hardware description
- [KiCad](https://kicad.org) - Open source EDA suite
- [Freerouting](https://github.com/freerouting/freerouting) - Open source autorouter
