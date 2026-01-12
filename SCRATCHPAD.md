# AtoPlace Development Scratchpad

## Session Summary - January 11, 2026

This document captures everything covered in the initial development session for AtoPlace, an AI-powered PCB placement and routing tool.

---

## 1. Project Origin

### Background
- Originated from work on the `aertactx-dogtracker` project
- Previous work included fixing build warnings, creating ARCHITECTURE.md, module builds, smart placement, and autorouting
- Issues existed with component placement being reset on rebuild and components being placed too far apart
- Existing Python tools: `smart_placer.py`, `module_autorouter.py`, `module_placer.py`, `autoplace_v2.py`

### Initial Request
User requested a comprehensive product development plan for:
- State-of-the-art autorouter and placement tool using generative AI
- Generate KiCad board files that are electronically correct and manufacturable
- Workflow starting with LLM conversation about what user wants to create

---

## 2. Research Documents Reviewed

### layout_rules_research.md
Comprehensive PCB layout design handbook covering:
- **Golden Rules**: Return current paths, modularity, inductance minimization
- **Stack-up & Materials**: RF/high-speed materials, layer configurations
- **Power Delivery Network (PDN)**: Decoupling capacitors, loop inductance, ferrite beads
- **Switching Regulators**: Hot loop minimization, feedback node isolation
- **EMI/EMC**: Edge plating, guard rings, stitching vias
- **Thermal Management**: Thermal vias, heatsink grounding
- **DfM Rules**: Fiducials, acid traps, solder mask dams, annular rings
- **Deep Dive Scenarios**: Ground planes (solid vs split), crystal layout, differential pairs

### CLAUDE.md (atopile documentation)
- Atopile DSL syntax and grammar
- Library modules/interfaces API (ElectricPower, I2C, Resistor, Capacitor, etc.)
- Package creation guide
- LLM development process for electronics design

---

## 3. Clarifying Questions Asked

I asked 8 categories of questions:

1. **Target Users & Use Cases** - Who is the primary user? What complexity?
2. **AI Integration Depth** - How much should AI own the design process?
3. **Technical Integration** - atopile vs standalone KiCad?
4. **Manufacturability Requirements** - Target fabs? Success criteria?
5. **Correctness Verification** - What level of electrical verification?
6. **Training & Knowledge Base** - What knowledge sources?
7. **Output & Iteration** - Output formats? Iteration approach?
8. **Scope & MVP** - What's the minimum viable first version?

---

## 4. User's Answers & Design Decisions

### Target Users
- **Primary**: Professional EEs wanting to accelerate workflow
- **Secondary**: Hobbyist accessibility
- **Complexity**: Medium (MCU + peripherals, 50-150 components)

### AI Integration
- **Approach**: Collaborative human-AI with engineer validation
- **Rationale**: "AI's real value lies in complementing human judgment, not replacing it"
- **Scope**: Technical specifications + layout preferences (not high-level requirements)

### Technical Integration
- Support both atopile and standalone KiCad
- Build on existing tools (smart_placer.py, module_autorouter.py)
- Abstract board access for KiCad version compatibility

### Manufacturability
- **Default**: JLCPCB-specific
- **Also support**: Generic DFM profiles
- **Success criteria**:
  1. Passes DRC with fab-specific rules
  2. BOM matches available inventory
  3. Assembly outputs generated

### Verification
- **MVP**: Netlist connectivity + confidence scoring
- **Defer**: SI/PI analysis to post-MVP (KiCad 10 adds time-domain calculations)
- **Approach**: Flag areas needing human review

### Knowledge Base
Hierarchical:
1. Core rules (layout_rules_research.md, physics constraints) - immutable
2. Component-specific (datasheets, reference designs) - parsed
3. Project-specific (user input) - configurable

**Do NOT rely on**: Community designs or LLM training data for layout decisions

### Output
- **MVP**: KiCad native only (.kicad_pcb, .kicad_sch)
- **Iteration**: Natural language with structured modifications
- **Avoid**: Bidirectional KiCad sync (too complex for MVP)

### MVP Scope
- **Focus**: Placement + Freerouting for routing
- **Board type**: IoT sensors and MCU breakouts
- **Not included**: Custom routing algorithm, SI/PI analysis, multi-format export

---

## 5. Architecture Design

```
USER INTERFACE LAYER
├── CLI (ato cmd)
├── MCP Server (Claude integration)
└── Natural Language Parser

ORCHESTRATION LAYER
└── AI Design Agent
    ├── Interprets user intent
    ├── Manages workflow
    └── Generates confidence reports

ENGINE LAYER
├── Placement Engine
│   ├── smart_placer.py (enhanced)
│   ├── Module detector
│   ├── Force-directed refinement
│   └── Constraint satisfaction
├── Routing Engine
│   ├── Freerouting (local JAR)
│   ├── Net classes
│   └── Diff pairs
└── Validation Engine
    ├── DRC Checker
    ├── DFM Validator
    └── Confidence Scorer

BOARD ABSTRACTION LAYER
├── Unified Board Model
├── KiCad Adapter (pcbnew API)
└── atopile Integration
```

---

## 6. Implementation Phases

### Phase 1: Foundation (COMPLETED)
- Force-directed refinement
- Pre-routing validation pipeline
- Confidence scoring framework
- DFM profile system

### Phase 2: Natural Language Interface (COMPLETED)
- Constraint parser (regex patterns)
- Modification handler
- MCP server integration (skeleton)

### Phase 3: Routing Integration (PLANNED)
- Freerouting Python client
- Net class assignment logic
- Differential pair detection
- Post-route DRC integration

### Phase 4: Production Ready (PLANNED)
- Manufacturing output generation
- JLCPCB BOM matching
- Comprehensive test suite
- Documentation

---

## 7. Files Created

### Package Structure
```
/Users/glovepost/Projects/atoplace/
├── atoplace/
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── board/
│   │   ├── __init__.py
│   │   ├── abstraction.py        # Board, Component, Net, Pad classes
│   │   └── kicad_adapter.py      # KiCad load/save
│   ├── placement/
│   │   ├── __init__.py
│   │   ├── force_directed.py     # Physics-based optimization
│   │   ├── module_detector.py    # Functional module detection
│   │   └── constraints.py        # Constraint types & solver
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── confidence.py         # Quality scoring
│   │   ├── pre_route.py          # Pre-routing checks
│   │   └── drc.py                # DRC checking
│   ├── dfm/
│   │   ├── __init__.py
│   │   └── profiles.py           # JLCPCB, OSH Park, PCBWay
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── constraint_parser.py  # NL -> constraints
│   │   └── modification.py       # Handle modifications
│   ├── routing/
│   │   └── __init__.py           # (pending)
│   ├── output/
│   │   └── __init__.py           # (pending)
│   └── mcp/
│       └── __init__.py           # (pending)
├── tests/
│   ├── __init__.py
│   ├── test_constraints.py
│   └── test_nlp.py
├── docs/
│   └── PRODUCT_PLAN.md
├── research/
│   └── layout_rules_research.md
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── LICENSE
└── .gitignore
```

---

## 8. Key Classes Implemented

### Board Abstraction
- `Board` - Main board representation with components, nets, outline
- `Component` - Position, rotation, footprint, pads, bounding box
- `Net` - Connections, net class, power/ground flags, differential pairs
- `Pad` - Position, size, shape, net connection
- `Layer` - PCB layer enumeration
- `BoardOutline` - Board geometry

### Placement
- `ForceDirectedRefiner` - Physics simulation with forces:
  - Repulsion (prevent overlap)
  - Attraction (minimize wire length)
  - Boundary (keep on board)
  - Constraint (user requirements)
  - Alignment (grid snapping)
- `ModuleDetector` - Identifies: power_supply, microcontroller, rf_frontend, sensor, connector, crystal, led
- `PlacementConstraint` - Base class
- Constraint types: Proximity, Edge, Zone, Grouping, Separation, Fixed
- `ConstraintSolver` - Evaluate and calculate forces

### Validation
- `ConfidenceScorer` - Assess design quality
- `ConfidenceReport` - Overall score, flags, sub-scores
- `DesignFlag` - Severity, category, location, message, suggested action
- `PreRouteValidator` - Check unconnected pads, single-pad nets, overlapping pads
- `DRCChecker` - Clearance, minimum sizes, edge clearance

### DFM Profiles
- `DFMProfile` - Trace/space, via, hole, mask, silk rules
- Pre-defined: JLCPCB_STANDARD, JLCPCB_STANDARD_4LAYER, JLCPCB_ADVANCED, OSHPARK_2LAYER, PCBWay_STANDARD

### NLP
- `ConstraintParser` - Regex patterns for common constraints
- `ModificationHandler` - Move, rotate, swap, flip operations
- `ParsedConstraint` - Constraint with confidence level

### CLI Commands
- `atoplace place <board>` - Run placement optimization
- `atoplace validate <board>` - Validate placement
- `atoplace report <board>` - Generate detailed report
- `atoplace interactive <board>` - Interactive constraint session

---

## 9. Constraint Patterns Supported

| Pattern | Example | Constraint Type |
|---------|---------|-----------------|
| Proximity | "keep C1 close to U1" | ProximityConstraint |
| Edge | "J1 on left edge" | EdgeConstraint |
| Rotation | "rotate J1 90 degrees" | FixedConstraint (with rotation) |
| Grouping | "group all capacitors together" | GroupingConstraint |
| Separation | "separate analog and digital" | SeparationConstraint |
| Fixed | "fix U1 at (50, 30)" | FixedConstraint |

---

## 10. Success Metrics Defined

| Metric | Target |
|--------|--------|
| Placement Quality | 90% of designs pass DRC without manual adjustment |
| Routing Success | 95% of nets routed automatically |
| Time Savings | 60% reduction vs manual layout |
| Confidence Accuracy | 85% correlation between flags and actual problems |
| User Satisfaction | NPS > 40 |

---

## 11. Risk Mitigations

| Risk | Mitigation |
|------|------------|
| LLM hallucination | Validate all parsed constraints against netlist; require confirmation |
| Freerouting fails | Fall back to partial routing + flags for manual completion |
| DFM rules outdated | Version DFM profiles; automated checks against fab documentation |
| Performance on large boards | Limit MVP to <200 components; optimize later |
| KiCad API changes | Abstract board access; support multiple KiCad versions |

---

## 12. Next Steps

1. **Test with real board** - Run on a KiCad PCB from the dogtracker project
2. **Implement Freerouting integration** - Python client for routing
3. **Add MCP server** - Claude integration for conversational workflow
4. **Manufacturing outputs** - Gerber, drill, BOM generation
5. **JLCPCB API integration** - Stock checking, cost estimation

---

## 13. Commands to Test

```bash
# Navigate to project
cd /Users/glovepost/Projects/atoplace

# Install with KiCad's Python
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install -e .

# Run placement
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m atoplace place /path/to/board.kicad_pcb

# Interactive mode
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m atoplace interactive /path/to/board.kicad_pcb
```

---

## 14. Key Insights from Research

### From layout_rules_research.md
1. **Return Current Paths**: For AC (>100kHz), current flows directly underneath the signal trace
2. **Ground Planes**: Solid ground plane wins over split planes
3. **Decoupling**: Place capacitors CLOSEST to IC power pin, smallest value closest
4. **Hot Loop**: Minimize the switching current loop area above all else
5. **Stitching Vias**: Max spacing 5mm or λ/10 of highest frequency

### From Existing Tools
1. `smart_placer.py` had simulated annealing but lacked force-directed refinement
2. Module detection was partial - needed comprehensive functional grouping
3. Constraint system was missing - now fully implemented

---

## 15. Design Decisions Made

1. **Board abstraction first** - Decouple from KiCad API for testability
2. **Regex before LLM** - Use patterns for common constraints, LLM for complex
3. **Confidence over correctness** - Flag uncertainty rather than guess
4. **DFM profiles as data** - Not hardcoded, easily extended
5. **Force-directed over annealing** - More intuitive tuning, better incremental updates

---

*End of session scratchpad - January 11, 2026*

---

## Session Summary - January 12, 2026

### Task: Codebase Assessment & Product Plan Validation

Reviewed the Product Plan (`docs/PRODUCT_PLAN.md`) and validated it against actual code implementation.

---

## Key Finding: Product Plan is Outdated

The Product Plan marks several features as "Planned" that are actually **fully implemented**. The atopile adapter (`board/atopile_adapter.py`) is 607 lines of production code but is untracked in git, explaining why the plan wasn't updated.

---

## Actual Implementation Status

### Phase 1: Foundation ✅ Complete (Plan Accurate)

| Module | Lines | Status |
|--------|-------|--------|
| `placement/force_directed.py` | ~640 | Full implementation |
| `placement/module_detector.py` | ~470 | 12 module types |
| `placement/constraints.py` | ~510 | 6 constraint types + solver |
| `validation/confidence.py` | ~530 | Scoring + reports |
| `validation/pre_route.py` | ~220 | 5 validation checks |
| `validation/drc.py` | ~160 | Clearance/edge checks |
| `dfm/profiles.py` | ~260 | 5 fab profiles |
| `board/abstraction.py` | ~370 | Unified board model |
| `board/kicad_adapter.py` | ~410 | KiCad load/save |

### Phase 2: NLP + Atopile ⚠️ Plan Says "Planned" but Mostly Complete

**NLP ✅ Complete** (accurately documented):
- `nlp/constraint_parser.py` (~640 lines) - Full regex parser + ModificationHandler
- Interactive CLI mode in `cli.py`

**Atopile 2A - Basic Support ✅ COMPLETE** (plan says "Planned"):
- `AtopileProjectLoader` class - fully implemented
- `get_board_path()` - resolves `.kicad_pcb` from entry point
- CLI auto-detection - `load_board_from_path()` in cli.py
- `detect_board_source()` - auto-detects KiCad vs atopile

**Atopile 2B - Enhanced Metadata ✅ COMPLETE** (plan says "Planned"):
- `_load_lock_file()` - parses `ato-lock.yaml`
- `_apply_component_metadata()` - enriches board with MPN/value/package
- `AtopileModuleParser` class - parses `.ato` files for hierarchy
- `infer_constraints()` - auto-generates proximity constraints

**Atopile 2C - Workflow Integration ⏳ Not Started**:
- Build hook support - not implemented
- Position persistence - not implemented
- MCP integration - stub only

### Phase 3: Routing ⏳ Stub Only
- `routing/__init__.py` - lazy imports that raise ImportError
- No actual implementation files exist

### Phase 4: Production ⏳ Stub Only
- `output/__init__.py` - stub
- `mcp/__init__.py` - stub

---

## Files Reviewed

```
✅ Full Implementation:
- atoplace/board/abstraction.py
- atoplace/board/kicad_adapter.py
- atoplace/board/atopile_adapter.py  ← UNTRACKED IN GIT
- atoplace/placement/force_directed.py
- atoplace/placement/module_detector.py
- atoplace/placement/constraints.py
- atoplace/validation/confidence.py
- atoplace/validation/pre_route.py
- atoplace/validation/drc.py
- atoplace/dfm/profiles.py
- atoplace/nlp/constraint_parser.py
- atoplace/cli.py

⏳ Stub Only:
- atoplace/routing/__init__.py
- atoplace/mcp/__init__.py
- atoplace/output/__init__.py
```

---

## Recommended Actions

1. **Update Product Plan** - Mark Phase 2A and 2B as complete
2. **Commit atopile_adapter.py** - Currently untracked, needs to be added to git
3. **Prioritize Next Phase**:
   - Option A: Phase 2C (MCP server) for Claude integration
   - Option B: Phase 3A (Freerouting) for routing capability

---

## Corrected Status Summary

| Phase | Product Plan Says | Actual Status |
|-------|-------------------|---------------|
| 1: Foundation | Complete | ✅ Complete |
| 2A: Basic Atopile | Planned | ✅ **Complete** |
| 2B: Enhanced Metadata | Planned | ✅ **Complete** |
| 2C: Workflow Integration | Planned | ⏳ Not started |
| 3: Routing | Planned | ⏳ Stub only |
| 4: Production | Planned | ⏳ Stub only |

---

*End of session - January 12, 2026*
