# AtoPlace Product Development Plan

## Purpose

This document is a grounded execution roadmap tied to the current codebase.
Marketing and product vision live in `README.md`.

---

## Project Status (2026-01-15)

- **Placement Engine:** Core force-directed solver is implemented with **Star Model** and **Adaptive Damping**.
- **Legalization:** Professional **Manhattan Legalizer** implemented with grid snapping, PCA-based row/column alignment, and priority-based overlap resolution.
- **Routing:** Internal **A* Geometric Router** implemented with Greedy Multiplier, Spatial Hash Indexing, obstacle map builder, and net ordering.
- **BGA Fanout:** **FanoutGenerator** implemented with dogbone/VIP patterns, onion-model layer assignment, and escape routing for high-density BGAs.
- **CLI Routing:** `atoplace route` command is functional with optional visualization.
- **Atopile Integration:** Project detection, `ato.yaml` parsing (fallback parser), and `ato-lock.yaml` module-to-ref mapping are functional.
- **Validation & MCP:** Pre-route and confidence scoring pipelines are functional; MCP exposes context tools/resources and DRC runner.

---

## Known Gaps (track details in `ISSUES.md`)

- ~~**Atopile Persistence:** `atoplace.lock` sidecar logic is planned but not yet implemented.~~ **IMPLEMENTED** - Full sidecar persistence with `atoplace.lock` files.
- ~~**BGA Fanout:** Specialized BGA fanout is not yet implemented (Plan created).~~ **IMPLEMENTED** - Full fanout module with dogbone/VIP patterns, onion-model layer assignment, and escape routing.
- **Routing Advanced:** Differential pair detection is not yet implemented.
- **Routing Fallback:** Freerouting runner is not yet implemented for failed nets.
- **MCP Routing:** No `route_board` MCP tool yet (CLI-only routing). BGA fanout tools are available.

---

## Research-Driven Technical Strategy

Based on deep-dive research (Jan 2026), we are adopting the following technical strategies:

### 1. Placement: "Abacus" Legalization
To solve the "Organic Layout" problem, we have implemented a post-physics legalization pipeline:
1.  **Quantizer:** Snap all components to a user-defined grid (0.5mm/0.1mm) and 90° rotation.
2.  **Beautifier:** Detect clusters of similar components (e.g., bypass caps) and align them into strict Rows or Columns using PCA-based axis detection.
3.  **Abacus Solver:** Remove overlaps using priority-based displacement (Sweep-line/MTV) to minimize movement from the physics-optimized location.

### 2. Physics: Star Model & Spatial Indexing
To solve instability and performance:
1.  **Star Model:** Implemented Star Model attraction for high-degree nets (GND/VCC), preventing "black hole" collapse.
2.  **Spatial Indexing:** Using O(~1) spatial hashing for collision detection.

### 3. Routing: A* + Greedy Multiplier
We have built a deterministic, iterative router:
1.  **Algorithm:** A* with a "Greedy Multiplier" ($w=2.0-3.0$) to balance speed vs. optimality.
2.  **Data Structure:** **Spatial Hash Index** for O(~1) collision detection.
3.  **Visualization:** `RouteVisualizer` (SVG/HTML) for debugging and results report.
4.  **Fallback:** Use Freerouting (Java) only for nets that fail the internal planner.

### 4. LLM Strategy: Multi-Level RAG
To handle 50k+ line board files:
1.  **Level 1 (Executive):** Board stats, critical modules, unrouted net count.
2.  **Level 2 (Schematic):** JSON connectivity graph (Who connects to Who) without coordinates.
3.  **Level 3 (Spatial Microscope):** Tool `focus_region(ref, radius)` returns precise geometry *only* for the requested area.

---

## Near-Term Roadmap (2026 Q1)

- **Milestone A: Placement Reliability**
    - [x] Implement **Star Model** physics for high-degree nets.
    - [x] Implement **Abacus Legalization** (Quantize -> Align -> Shove).
    - [x] Implement **Sidecar Persistence** (`atoplace.lock`) for atopile integration.
    - [x] Add **Polygonal Outline** support (ray-casting containment).

- **Milestone B: Routing Foundation**
    - [x] Build **RouteVisualizer** (SVG/HTML export).
    - [x] Implement **SpatialHashIndex** for obstacles.
    - [x] Implement **ObstacleMapBuilder** (pads, keepouts, existing traces).

- **Milestone C: Basic Routing**
    - [x] Implement **A* Router** with greedy multiplier.
    - [x] Implement **Net Orderer** (hardest first).
    - [x] CLI `atoplace route` command.

- **Milestone D: Advanced Routing**
    - [x] Implement **BGA Fanout Generator** (Dogbone/VIP, Escape Routing).
    - [ ] Implement **Differential Pair Router** (Dual-grid geometric planner).
    - [ ] Implement **Freerouting Fallback**.

---

## Implementation Phases

**Status Legend:** [x] complete, [~] implemented but needs fixes, [ ] planned

### Phase 1: Placement Engine (Refining)
- [x] Force-directed solver (Star Model + adaptive damping)
- [x] Pre-routing validation pipeline
- [x] **Star Model** implementation for nets
- [x] **Legalization Pipeline:**
    - [x] Grid Snapping (Quantizer)
    - [x] Row/Col Alignment (PCA-based)
    - [x] Overlap Removal (Abacus/Sweep-line)
- [x] **Sidecar Persistence:**
    - [x] `atoplace.lock` parser/writer
    - [x] Merge logic (Lock > Physics)

### Phase 2: Routing Integration (New)
**Implementation Reference:** `research/routing_implementation_plan.md`

**Phase 2A - Foundation:**
- [x] `atoplace.routing.visualizer`: SVG/HTML rendering of routing state.
- [x] `atoplace.routing.spatial_index`: Spatial Hash implementation.
- [x] `atoplace.routing.obstacle_map`: Board to Obstacle converter.

**Phase 2B - The Router:**
- [x] `atoplace.routing.astar`: A* implementation with `w` multiplier.
- [x] `atoplace.routing.net_orderer`: Difficulty scorer.
- [ ] `atoplace.routing.cache`: Pattern caching system.

**Phase 2C - Advanced & Fallback:**
- [ ] `atoplace.routing.diff_pair`: Dual-grid geometric planner for diff pairs.
- [ ] `atoplace.routing.freerouting`: Java runner for fallback.

**Phase 2D - BGA Fanout (New):**
- [x] `atoplace.routing.fanout.patterns`: Dogbone/VIP geometry generators.
- [x] `atoplace.routing.fanout.layer_assigner`: Onion-model ring analysis.
- [x] `atoplace.routing.fanout.escape_router`: Spoke routing to clear space.
- [x] `atoplace.routing.fanout.generator`: Main FanoutGenerator class.
- [x] CLI `atoplace fanout` command.
- [x] MCP fanout tools (detect_bga_components, fanout_component, fanout_all_bgas, get_fanout_preview).

### Phase 3: Atopile & NLP
- [x] Atopile project detection
- [x] `ato.yaml` nested parser (or PyYAML dependency)
- [x] `atoplace.lock` integration for module positions
- [x] LLM "Microscope" tool implementation (`inspect_region`)

### Phase 4: MCP Server
- [ ] `place_board` tool
- [ ] `route_board` tool
- [x] `run_drc` tool (atoplace + KiCad)
- [x] BGA fanout tools (detect_bga_components, fanout_component, fanout_all_bgas, get_fanout_preview)
- [x] Resource: Board Summary (Level 1)
- [~] Resource: Connectivity Graph (Level 2)

---

## Quality Gates

- **Placement:**
    - No overlaps at DFM min spacing.
    - All components on 0.1mm grid (unless 0201/fine-pitch).
    - Passives aligned to rows/cols where possible.
- **Routing:**
    - 100% connectivity (or explicit failure report).
    - No DRC violations (clearance, width).
    - Maximize 45° routing; minimize vias.
- **Performance:**
    - Placement < 30s for 200 components.
    - Routing < 60s for 100 nets.

---

## Dependencies & Assumptions

- **KiCad 8+**: Required for `pcbnew` Python API.
- **Freerouting**: Optional fallback (jar file).
- **Python 3.10+**: For type hinting and performance.
- **System**: 8GB+ RAM recommended for routing large boards.

---

## File Structure Update

```
atoplace/
├── placement/
│   ├── force_directed.py   # Physics (Star Model)
│   ├── legalizer.py        # Abacus/Grid Snapping
│   └── ...
├── routing/
│   ├── visualizer.py       # Debug viz
│   ├── spatial_index.py    # O(~1) collision
│   ├── obstacle_map.py     # World builder
│   ├── astar_router.py     # Core logic
│   ├── net_orderer.py      # Difficulty sorting
│   └── fanout/             # BGA Fanout module
│       ├── generator.py    # FanoutGenerator main class
│       ├── patterns.py     # Dogbone/VIP patterns
│       ├── layer_assigner.py  # Onion model layer assignment
│       └── escape_router.py   # Spoke escape routing
├── board/
│   ├── lock_file.py        # atoplace.lock handler
│   └── ...
└── ...
```
