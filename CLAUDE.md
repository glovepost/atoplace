# CLAUDE.md

This is the AtoPlace project - an AI-powered PCB placement and routing tool.

## Session Start Checklist

At the start of every session:

1. **Review key documents:**
   - `docs/PRODUCT_PLAN.md` - Current development roadmap and priorities
   - `ISSUES.md` - Active bugs, code review findings, and fix history

2. **Work on resolving all open issues** in `ISSUES.md`, prioritizing High before Medium before Low

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

## Issue Tracking

Issues are tracked in `ISSUES.md`. When fixing issues:

1. Work through issues in priority order (High before Medium before Low)
2. After fixing a batch of related issues, create a git commit and push
3. Update `ISSUES.md` to mark resolved issues with `~~strikethrough~~` and **FIXED** notes

## Development Scratchpad

After completing a major chunk of work, **always update `SCRATCHPAD.md`** with:

1. **Session header**: Date and task summary
2. **What was done**: Issues fixed, features implemented
3. **Files modified/created**: Table of changes
4. **Key algorithms/classes**: Document new classes and important logic
5. **Current status**: Updated project phase status table
6. **Next steps**: What should be tackled next

This creates a running log of development decisions and progress that persists across sessions.
