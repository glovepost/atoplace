# AtoPlace Product Development Plan

## Executive Summary

**Product Name:** AtoPlace

**Vision:** An LLM-augmented PCB layout tool that accelerates professional EE workflows by automating placement optimization and routing for medium-complexity boards, while maintaining human oversight for critical decisions.

**Core Value Proposition:** Reduce PCB layout time by 60-80% for IoT/MCU boards through intelligent placement, automated routing, and natural language iteration - without sacrificing design quality or manufacturability.

---

## Target Users

### Primary: Professional EEs Wanting to Accelerate Workflow
- Engineers working on medium-complexity boards (50-150 components)
- Teams doing iterative product development
- Those familiar with KiCad/atopile workflows

### Secondary: Hobbyist Accessibility
- Makers with limited PCB layout experience
- Students learning electronics design
- Rapid prototyping needs

---

## Design Philosophy

### Collaborative Human-AI
The AI complements human judgment rather than replacing it:
- AI handles repetitive placement optimization
- AI suggests and validates, human approves
- RF and high-speed signals flagged for manual attention
- Confidence scoring indicates where review is needed

### Knowledge-Based Approach
Rules come from verified sources, not LLM training data:
- Core rules from `layout_rules_research.md`
- Component-specific rules from datasheets
- DFM rules from fab house specifications

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                        │
│  CLI  │  MCP Server (Claude)  │  Natural Language Parser        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                         │
│                    AI Design Agent                              │
│  - Interprets user intent                                       │
│  - Manages design workflow                                      │
│  - Generates confidence reports                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────┬───────────────┬───────────────┐
│  PLACEMENT  │   ROUTING     │  VALIDATION   │
│  ENGINE     │   ENGINE      │  ENGINE       │
│             │               │               │
│ Module      │ Freerouting   │ DRC Checker   │
│ Detector    │ (local JAR)   │ DFM Validator │
│             │               │               │
│ Force-      │ Net Classes   │ Confidence    │
│ Directed    │ Diff Pairs    │ Scorer        │
│ Refiner     │               │               │
└─────────────┴───────────────┴───────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 BOARD ABSTRACTION LAYER                         │
│  Unified Board Model  │  KiCad Adapter  │  atopile Integration  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Workflow

### Phase 1: Design Input
User provides:
- Schematic/netlist (atopile or KiCad)
- Board outline (optional)
- Initial constraints via conversation

### Phase 2: Intelligent Placement
Automated steps:
1. Module detection (power, RF, MCU, sensors)
2. Topology analysis
3. Initial placement using smart placement algorithm
4. Force-directed refinement
5. Constraint satisfaction verification

### Phase 3: Iterative Refinement
User can request changes via natural language:
- "Move C3 closer to U1"
- "Rotate the power connector 180 degrees"
- "Swap C1 and C2"

### Phase 4: Automated Routing
1. Net class assignment
2. Differential pair identification
3. Route with Freerouting
4. Post-route DRC check
5. Flag issues for user

### Phase 5: Output Generation
- KiCad native files
- Manufacturing outputs (Gerbers, drill, BOM, PnP)
- Confidence report with flagged items

---

## Implementation Phases

### Phase 1: Foundation (Complete)
- [x] Force-directed refinement
- [x] Pre-routing validation pipeline
- [x] Confidence scoring framework
- [x] DFM profile system

### Phase 2: Natural Language Interface (Complete)
- [x] Constraint parser (regex patterns)
- [x] Modification handler
- [x] Interactive CLI mode

### Phase 3: Routing Integration (Planned)
- [ ] Freerouting Python client
- [ ] Net class assignment logic
- [ ] Differential pair detection
- [ ] Post-route DRC integration

### Phase 4: Production Ready (Planned)
- [ ] Manufacturing output generation
- [ ] JLCPCB BOM matching
- [ ] Comprehensive test suite
- [ ] Documentation

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Placement Quality | 90% pass DRC | Automated testing |
| Routing Success | 95% nets routed | Track unrouted |
| Time Savings | 60% reduction | User surveys |
| Confidence Accuracy | 85% correlation | Compare to fab issues |

---

## DFM Profiles

Pre-configured for popular fabs:
- **JLCPCB Standard** - 1-2 layer boards
- **JLCPCB Standard 4-Layer** - 4 layer boards
- **JLCPCB Advanced** - Tighter tolerances
- **OSH Park** - 2-layer purple boards
- **PCBWay Standard** - General manufacturing

---

## Constraint Types

| Type | Example | Description |
|------|---------|-------------|
| Proximity | "Keep C1 close to U1" | Minimize distance |
| Edge | "J1 on left edge" | Board edge placement |
| Zone | "Analog in top-left" | Area restriction |
| Grouping | "Group capacitors" | Keep together |
| Separation | "Separate analog/digital" | Maintain distance |
| Fixed | "U1 at (50, 30)" | Lock position |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM hallucination | Validate against netlist, require confirmation |
| Freerouting fails | Fall back to partial routing + flags |
| DFM rules outdated | Version profiles, check fab docs |
| Performance on large boards | Limit MVP to <200 components |
| KiCad API changes | Abstract board access layer |

---

## File Structure

```
atoplace/
├── atoplace/
│   ├── board/          # Board abstraction
│   ├── placement/      # Placement algorithms
│   ├── routing/        # Routing integration
│   ├── validation/     # Quality checks
│   ├── dfm/            # DFM profiles
│   ├── nlp/            # NL parsing
│   ├── output/         # Manufacturing outputs
│   ├── mcp/            # MCP server
│   └── cli.py          # CLI interface
├── tests/              # Test suite
├── docs/               # Documentation
├── research/           # Research documents
└── examples/           # Example projects
```
