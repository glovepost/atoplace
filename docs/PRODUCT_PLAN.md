# AtoPlace Product Development Plan

## Purpose

This document is a grounded execution roadmap tied to the current codebase.
Marketing and product vision live in `README.md`.

---

## Current Capabilities (as implemented)

- Load/save KiCad boards via `pcbnew` with component/pad extraction.
- Force-directed placement refinement with constraint support.
- Natural language constraint parsing and interactive CLI adjustments.
- Pre-route validation and confidence scoring with DFM profiles.
- Atopile project detection and board path resolution (partial, see gaps).

---

## Known Gaps (track details in `ISSUES.md`)

- Placement physics issues (net attraction scaling, boundary handling, convergence).
- Polygonal board outlines not supported in containment checks.
- Pad rotation is not modeled, skewing overlap/extent checks.
- Atopile YAML fallback parser lacks nested support; PyYAML not declared.
- Differential pair detection marks only the positive net.
- CLI flag `--use-ato-modules` is defined but unused.
- Routing, MCP server, and manufacturing outputs are stubs.

---

## Project Status (2026-01-12)

- **Placement Engine:** Core force-directed solver is implemented but exhibits critical "organic" layout behaviors (lack of grid snapping/alignment) and stability issues (high-degree net collapse, inaccurate circular boundary checks).
- **Atopile Integration:** Basic detection works, but deep integration is broken. The fallback YAML parser fails on nested files, and module mapping (instance path vs. KiCad ref) is non-functional. `ato-lock.yaml` persistence is missing.
- **Routing/MCP:** These modules are currently stubs (`__init__.py` only).
- **Validation:** Pre-route and Confidence Scoring pipelines are functional but lack "Proactive Force" feedback into the physics engine.

---

## Known Gaps (track details in `ISSUES.md`)

- **Placement Quality:** "Organic" vs. "Manhattan" gap. The current solver produces valid but "messy" layouts that require significant manual cleanup (no grid snapping or row alignment).
- **Physics Instability:** $O(N^2)$ attraction scaling causes component collapse on power nets; circular boundary checks allow rectangular components to drift off-board.
- **Polygonal Support:** Complex board shapes are ignored (reduced to bounding box).
- **Atopile Reliability:** Module mapping is broken; dependency management (`PyYAML`) is missing.
- **Routing Strategy:** No distinction between critical signals (USB, RF) and general nets; no fanout generation.

---

## Near-Term Roadmap (2026 Q1)

- Fix placement physics scaling: reduce high-degree net collapse ($O(N)$ Star Model), improve boundary handling (AABB), and add oscillation detection.
- **Implement Legalization Phase:** Add Grid Snapping, Row Alignment, and Overlap Removal (Sweep-line/R-Tree) to solve the "Organic" layout issue.
- Implement polygonal outline support end-to-end (outline extraction + containment checks).
- Harden atopile integration: reliable YAML parsing, module mapping (via `ato-lock.yaml` sidecar logic), and declared PyYAML dependency.
- Wire CLI flags and improve diff-pair detection for both sides of the pair.

---

## Current Architecture (as implemented)

- CLI and Python API drive placement and validation.
- Board abstraction layer wraps KiCad `pcbnew` IO and atopile metadata.
- Placement engine uses force-directed refinement plus constraints.
- Validation includes pre-route checks, DRC heuristics, and confidence scoring.
- Routing, MCP, and manufacturing outputs are not implemented.

---

## Current Workflow (as implemented)

1. Load board from KiCad `.kicad_pcb` or atopile project path.
2. Parse constraints from text (optional).
3. Run force-directed refinement.
4. Run validation and confidence scoring.
5. Save updated placement to `.kicad_pcb`.

---

## Implementation Phases

**Status Legend:** [x] complete, [~] implemented but needs fixes, [ ] planned

### Phase 1: Placement Foundation (Complete)
- [x] Force-directed refinement (Star Model, AABB collision, convergence detection)
- [x] Pre-routing validation pipeline (DFM-driven spacing, rotated pad geometry)
- [~] Confidence scoring framework (routing checks are stubbed)
- [x] DFM profile system
- [x] Legalization pipeline (Grid snapping, PCA row alignment, priority-based overlap removal)

### Phase 2: Natural Language + Atopile Integration (In Progress)

**NLP (In Progress):**
- [~] Constraint parser (regex patterns; missing some target captures)
- [~] Modification handler (move closer/away lacks target parsing)
- [x] Interactive CLI mode

**Atopile Phase 2A - Basic Support (In Progress):**
- [~] AtopileProjectLoader - detect and parse `ato.yaml` (fallback parser needs nested support)
- [~] Board path resolution from entry point (needs more layout variants)
- [~] CLI auto-detection of atopile projects (flag wiring and error flow still rough)

**Atopile Phase 2B - Enhanced Metadata (Planned):**
- [~] Lock file parser (`ato-lock.yaml`) for component values (depends on YAML reliability)
- [~] Module hierarchy parser for grouping hints (instance-to-ref mapping missing)
- [ ] Constraint inference from atopile patterns

**Atopile Phase 2C - Workflow Integration (Planned):**
- [ ] Build hook support for post-build placement
- [ ] Position persistence to lock file
- [ ] MCP integration for combined atopile+AtoPlace operations

### Phase 3: Routing Integration (Planned)

**Phase 3A - Basic Freerouting:**
- [ ] FreeroutingRunner with JAR execution
- [ ] DSN export via pcbnew API
- [ ] SES import back to KiCad
- [ ] CLI command `atoplace route`

**Phase 3B - Smart Routing:**
- [ ] NetClassAssigner - automatic net classification
- [ ] DiffPairDetector - USB, LVDS, Ethernet pairs
- [ ] Pre-route net class assignment
- [ ] Post-route DRC integration

**Phase 3C - Advanced Routing:**
- [ ] Docker execution mode
- [ ] Freerouting JAR bundling and auto-install
- [ ] Progress streaming for long routes
- [ ] `place-and-route` combined workflow

**Phase 3D - Full MCP Integration:**
- [ ] `route_board` MCP tool
- [ ] `modify_placement` MCP tool
- [ ] `export_manufacturing` MCP tool
- [ ] MCP prompts for guided workflows
- [ ] Board state management for multi-turn conversations

### Phase 4: Production Ready (Planned)

**Phase 4A - Manufacturing Outputs:**
- [ ] Gerber generation
- [ ] Drill file generation
- [ ] BOM export (CSV, JLCPCB format)
- [ ] Pick-and-place file generation

**Phase 4B - JLCPCB Integration:**
- [ ] JLCPCB parts library matching
- [ ] Stock availability checking
- [ ] Cost estimation

**Phase 4C - Quality & Documentation:**
- [ ] Comprehensive test suite
- [ ] API documentation
- [ ] User guide
- [ ] Example projects

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Placement Quality | 90% pass DRC | Automated testing |
| Routing Success | 95% nets routed | Track unrouted |
| Time Savings | 60% reduction | User surveys |
| Confidence Accuracy | 85% correlation | Compare to fab issues |

---

## Quality Gates

- Placement: no overlaps at DFM min spacing, all components within outline.
- Validation: pre-route warnings reviewed; DRC has zero errors on core checks.
- NLP: constraints parsed deterministically with logged warnings for rejects.
- Performance: <60 seconds for 150-component boards on a laptop.

---

## Dependencies & Assumptions

- KiCad `pcbnew` available for real IO and outline/pad extraction.
- Freerouting JAR or Docker image available for routing.
- PyYAML optional but recommended; fallback parser needs nested support to be safe.
- Deterministic outputs prioritized over LLM-only behaviors.

---

## Milestones

### Milestone A: Placement Reliability (Target: 2026 Q1)
- Fix attraction scaling for high-degree nets.
- Boundary forces respect component rectangles, not diagonal-only margins.
- Add oscillation/stall detection to convergence.
- Add polygon outline containment and update boundary checks.
- Add pad rotation in geometry model and overlap checks.

### Milestone A+: Professional Polish (Target: 2026 Q1)
- **Legalization Phase:** ~~Implement Grid Snapping, Overlap Removal (R-Tree), and Row Alignment.~~ **DONE** - Implemented in `atoplace/placement/legalizer.py` with PCA-based axis detection, median projection, and priority-based MTV overlap resolution.
- **Critical Path Routing:** Implement Geometric Planner for Tier 1 diff-pairs (Dual-Grid A*).
- **Fanout Generation:** Implement BGA/QFN escape routing strategies.

### Milestone B: Atopile Viability (Target: 2026 Q1-Q2)
- PyYAML dependency declared (or nested fallback parser implemented).
- Module hierarchy mapping to KiCad references.
- CLI flag `--use-ato-modules` wired end-to-end.
- Build-path resolution supports common layout variants.

### Milestone C: Routing MVP (Target: 2026 Q2)
- Freerouting runner + DSN/SES round-trip.
- Net class assignment and diff-pair detection (both + and - nets).
- CLI `atoplace route` command.

### Milestone D: MCP Alpha (Target: 2026 Q2)
- MCP server skeleton with `place_board` and `validate_placement`.
- Basic resources for components and nets.

---

## Backlog (Prioritized)

1. Placement physics stability (high-degree net weights, boundary handling).
2. Polygonal outline support and containment checks.
3. Pad rotation modeling + accurate pad overlap detection.
4. Atopile YAML robustness and module/ref mapping.
5. CLI flag wiring and error flow improvements.
6. Diff-pair detection symmetry and net-class assignment.
7. Routing pipeline scaffolding (DSN/SES).
8. Manufacturing outputs pipeline (Gerber/BOM/PNP).
9. MCP server tools/resources/prompts.

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
| Freerouting fails | Bundle versioned JAR; fall back to partial routing + flags |
| DFM rules outdated | Version profiles, check fab docs |
| Performance on large boards | Limit MVP to <200 components |
| KiCad API changes | Abstract board access layer |
| atopile format changes | Add adapter tests + pin versioned schema |
| atopile build breaks AtoPlace | Post-build integration (no compiler coupling) |
| ato-lock.yaml schema changes | Version detection + graceful degradation |
| Module hierarchy parsing errors | Fall back to KiCad-only component grouping |
| Freerouting JAR not found | Auto-installer + clear error message |
| Freerouting routing fails | Fall back to partial routing + flag unrouted nets |
| DSN export/import errors | KiCad CLI fallback + version compatibility tests |
| MCP protocol changes | Pin MCP SDK version + adapter layer |
| Long routing times | Timeout with partial results + progress streaming |
| Java not installed | Docker fallback + installation guide |

---

## File Structure

```
atoplace/
├── atoplace/
│   ├── board/              # Board abstraction
│   │   ├── abstraction.py
│   │   ├── kicad_adapter.py
│   │   └── atopile_adapter.py    # Atopile project integration
│   ├── placement/          # Placement algorithms
│   │   ├── force_directed.py
│   │   ├── module_detector.py
│   │   └── constraints.py
│   ├── routing/            # Routing integration (stubs only)
│   │   └── __init__.py
│   ├── validation/         # Quality checks
│   │   ├── confidence.py
│   │   ├── pre_route.py
│   │   └── drc.py
│   ├── dfm/                # DFM profiles
│   │   └── profiles.py
│   ├── nlp/                # NL parsing
│   │   ├── constraint_parser.py
│   │   └── modification.py
│   ├── output/             # Manufacturing outputs (stubs only)
│   │   └── __init__.py
│   ├── mcp/                # MCP server (stubs only)
│   │   └── __init__.py
│   └── cli.py              # CLI interface
├── tests/
│   ├── test_atopile_integration.py
│   ├── test_constraints.py
│   ├── test_nlp.py
│   └── ...
├── docs/               # Documentation
├── research/           # Research documents
└── examples/
    ├── atopile-esp32/  # Example atopile project
    └── kicad-sensor/   # Example KiCad project
```

---

## Atopile Integration (Execution Notes)

- Current behavior: auto-detects `ato.yaml`, resolves a board path, and loads via KiCad.
- Gaps: YAML fallback parser does not handle nested structures; module hierarchy uses instance names rather than KiCad references.
- Next steps:
  - Declare PyYAML dependency or implement a nested-safe parser.
  - Map module hierarchy to KiCad reference designators.
  - Wire `--use-ato-modules` and improve build-path resolution variants.

---

## MCP Server Integration (Planned)

- Current status: `atoplace/mcp/__init__.py` is a stub only.
- Planned scope: provide MCP tools for place/validate/modify plus basic resources.
- Dependencies: stable CLI/placement APIs and board IO in place.

---

## Routing Integration (Planned)

- Current status: no routing modules exist beyond a stub `atoplace/routing/__init__.py`.
- Planned scope: DSN export/import, Freerouting runner, net class assignment, diff-pair detection, CLI `atoplace route`.
- Dependencies: stable board IO, outline handling, and DFM-driven constraints.
