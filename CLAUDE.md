# CLAUDE.md

This is the AtoPlace project - an AI-powered PCB placement and routing tool.

## Project Overview

AtoPlace uses natural language understanding and physics-based algorithms to automate PCB layout. It integrates with KiCad and atopile.

## Key Concepts

### Board Abstraction (`atoplace/board/`)
- `Board` - Unified representation of PCB board
- `Component` - Component with position, rotation, pads
- `Net` - Electrical connection between pads
- `kicad_adapter.py` - Load/save KiCad files

### Placement (`atoplace/placement/`)
- `ForceDirectedRefiner` - Physics simulation for placement optimization
- `ModuleDetector` - Identifies functional modules (power, RF, digital)
- `PlacementConstraint` - Base class for constraints
- Constraint types: Proximity, Edge, Zone, Grouping, Separation, Fixed

### Validation (`atoplace/validation/`)
- `ConfidenceScorer` - Assess design quality
- `ConfidenceReport` - Detailed assessment with flags
- `PreRouteValidator` - Pre-routing checks
- `DRCChecker` - Design rule checking

### DFM Profiles (`atoplace/dfm/`)
- `DFMProfile` - Fab-specific design rules
- Pre-defined: JLCPCB, OSH Park, PCBWay

### NLP (`atoplace/nlp/`)
- `ConstraintParser` - Parse natural language to constraints
- `ModificationHandler` - Handle placement modifications

## Running the Tool

Requires KiCad's Python (pcbnew):

```bash
# macOS
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m atoplace place board.kicad_pcb

# With constraints
atoplace place board.kicad_pcb --constraints "USB on left edge"
```

## Development Workflow

1. Load board with `Board.from_kicad(path)`
2. Parse constraints with `ConstraintParser`
3. Run `ForceDirectedRefiner.refine()`
4. Assess with `ConfidenceScorer.assess()`
5. Save with `board.to_kicad(path)`

## Testing

```bash
pytest tests/
```

## Key Files

- `atoplace/cli.py` - Command-line interface
- `atoplace/placement/force_directed.py` - Core placement algorithm
- `atoplace/validation/confidence.py` - Quality scoring
- `atoplace/nlp/constraint_parser.py` - NL parsing
