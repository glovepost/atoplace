# Product Requirements Document: AtoPlace (v2.0)

**Date:** 2026-01-13
**Status:** Living Document
**Target Audience:** Development Team, Product Stakeholders

---

## Index

- 1. Product Vision
- 2. Core Value Pillars
- 3. System Architecture
- 4. Functional Requirements
- 5. Roadmap & Phasing
- 6. Technical Constraints & Standards
- Appendix A: Atopile Integration (Future Spec)
- Appendix B: MCP Server Integration (Future Spec)
- Appendix C: Freerouting Integration (Future Spec)

---

## 1. Product Vision
To provide Professional Electrical Engineers with an "AI Pair Designer" that automates the tedious 80% of PCB layout (placement optimization, non-critical routing, validation) while strictly adhering to "Manhattan" design aesthetics, DFM constraints, and Signal Integrity (SI) best practices. Unlike "Black Box" autorouters, AtoPlace produces human-readable, professional-grade layouts that an engineer would be proud to claim as their own.

---

## 2. Core Value Pillars

### 2.1. "Human-Grade" Aesthetics (The "Turing Test" of Layout)
*   **Grid Compliance:** Components must not float; they must snap to user-defined grids (e.g., 0.1mm, 0.5mm, 1mm).
*   **Alignment:** Components, especially passives (R/C/L), must form strict rows or columns.
*   **Manhattan Routing:** Traces must follow 45/90-degree routing rules. No "any-angle" spaghetti unless specifically requested (e.g., flex PCBs).

### 2.2. Physics-Driven Signal Integrity
*   **Impedance First:** Critical nets (USB, Ethernet, RF) are prioritized and routed with impedance/length matching constraints.
*   **Power Plane Integrity:** Component placement prioritizes continuous ground return paths.
*   **Hierarchy-Aware:** Placement respects the logical signal flow (Input → Processing → Output).

### 2.3. Transparent Interactivity
*   **No Lock-in:** Native KiCad files are the source of truth. The user can take over manually at any moment.
*   **Explainable Actions:** The system can explain *why* a component was placed in a specific location (e.g., "Placed C1 near U1 to minimize loop inductance").
*   **Natural Language Control:** Users can direct high-level moves ("Move the USB connector to the bottom edge") without dragging individual pads.

---

## 3. System Architecture

AtoPlace operates as an orchestration layer over existing, proven EDA tools (KiCad, Freerouting) with its own internal geometry engines.

```mermaid
graph TD
    User((User)) -->|Natural Language| NLParser[NLP & Intent Engine]
    User -->|KiCad/Atopile Files| Loader[Project Loader]
    
    subgraph "Phase 1: Intelligent Placement"
        Loader --> Physics[Force-Directed Physics (Star Model)]
        Physics -->|Raw Cloud| Quantizer[Grid Snapping]
        Quantizer --> Beautifier[Row/Col Alignment]
        Beautifier --> Solver[Abacus Legalization]
        Solver -->|Manhattan Layout| Valid[Validation Engine]
    end
    
    subgraph "Phase 2: Critical Routing"
        Valid --> Fanout[BGA/QFN Fanout Generator]
        Fanout --> CritRoute[A* Geometric Planner]
    end
    
    subgraph "Phase 3: Completion"
        CritRoute -->|Locked Critical Nets| Fallback[Freerouting Fallback]
        Fallback --> DFM[DFM & DRC Checker]
    end
    
    DFM -->|Final Board| Output[Output Generator]
    Valid -->|Feedback| Physics
```

---

## 4. Functional Requirements

### 4.1. Placement Engine (The "Brain")
*   **REQ-P-01 (Clustering):** System MUST group components by logical module (Power, Analog, Digital, RF) using schema hierarchy or netlist topology.
*   **REQ-P-02 (Physics):** System MUST use force-directed annealing for global optimization ($O(N)$ **Star Model** for large nets like GND/VCC).
    *   *Constraint:* Repulsion forces MUST use **Spatial Hashing** (O(~1)) for performance.
*   **REQ-P-03 (Legalization - CRITICAL):** System MUST apply a 3-stage post-physics pipeline:
    *   *Quantizer:* Snap component centroids to the user grid (0.5mm/0.1mm) and 90° rotation.
    *   *Beautifier:* Align adjacent same-size components into shared-axis rows/columns using PCA.
    *   *Solver:* Remove overlaps using the **Abacus** algorithm (dynamic programming) to minimize displacement.
*   **REQ-P-04 (Flow):** System MUST attempt to place components in logical signal flow order.

### 4.2. Routing Engine (The "Nervous System")
*   **REQ-R-01 (Fanout):** System MUST generate escape patterns (dogbone, via-in-pad) for high-density components (BGA/QFN) *before* general routing.
*   **REQ-R-02 (Core Algorithm):** System MUST implement an **A* Router** with a "Greedy Multiplier" ($w=2.0-3.0$) for internal routing of Tier 1/2 nets.
    *   *Data Structure:* Must use **Spatial Hash Index** for obstacle detection.
    *   *Visualization:* Must include a `RouteVisualizer` (SVG/HTML) for debugging.
*   **REQ-R-03 (Priority):** System MUST support a "Priority Queue" for nets:
    *   *Tier 1:* Diff-pairs, RF, Clocks (Geometric Planner).
    *   *Tier 2:* Power/Ground (Route second, huge widths/planes).
    *   *Tier 3:* General Signals (Fallback to Freerouting if internal router fails).

### 4.3. Validation & Physics Feedback
*   **REQ-V-01 (Proactive Forces):** Validation rules (e.g., "Decoupling caps < 2mm") MUST be projected into the physics engine as high-strength attractive forces.
*   **REQ-V-02 (Confidence):** System MUST emit a "Confidence Score" (0-100%) based on routability, SI metrics, and DFM compliance.

### 4.4. Integration & UX
*   **REQ-I-01 (Persistence):** System MUST support `atoplace.lock` (sidecar pattern) to persist placement data for Atopile projects without modifying source code.
*   **REQ-I-02 (CLI):** System MUST provide a CLI for CI/CD integration (`atoplace check board.kicad_pcb`).
*   **REQ-I-03 (MCP):** System MUST expose an MCP Server interface for LLM agents.
*   **REQ-I-04 (LLM Context):** System MUST implement a **Multi-Level RAG** strategy:
    *   *Level 1:* Executive Summary (Stats, Modules).
    *   *Level 2:* Structured Netlist (Connectivity Graph, no coords).
    *   *Level 3:* Spatial Microscope (`focus_region` tool) for precise geometry.

---

## 5. Roadmap & Phasing

### Phase 1: The "Solid Foundation" (Current Focus)
*   **Goal:** A tool that places components logically and legally (Manhattan style).
*   **Key Deliverables:**
    *   Physics Engine upgrade (Star Model, Spatial Hash).
    *   Legalization Pipeline (Quantizer, Beautifier, Abacus).
    *   Sidecar Persistence (`atoplace.lock`).
    *   Polygonal Outline support.

### Phase 2: The "Routing Assistant"
*   **Goal:** A tool that can safely route a board without ruining signal integrity.
*   **Key Deliverables:**
    *   Routing Visualization System.
    *   Obstacle Map Builder & Spatial Index.
    *   A* Router (Greedy Multiplier).
    *   Fanout Generator for QFN/BGA.

### Phase 3: The "Professional Agent"
*   **Goal:** An autonomous agent capable of passing a Senior Engineer's design review.
*   **Key Deliverables:**
    *   MCP Server for full conversational design.
    *   Deep Signal Integrity checks.
    *   Automated DFM output generation.

---

## 6. Technical Constraints & Standards
*   **Language:** Python 3.10+
*   **Input Format:** KiCad 8+ (`.kicad_pcb`), Atopile (`.ato`, `.yaml`)
*   **Routing Backend:** Internal A* (Primary), Freerouting (Fallback).
*   **License:** MIT (Open Source Core).

---

## Appendix A: Atopile Integration (Future Spec)

See [ATOPILE_INTEGRATION.md](specs/ATOPILE_INTEGRATION.md) for full details.

## Appendix B: MCP Server Integration (Future Spec)

See [MCP_SERVER.md](specs/MCP_SERVER.md) for full details.

## Appendix C: Freerouting Integration (Future Spec)

See [ROUTING_STRATEGY.md](specs/ROUTING_STRATEGY.md) for full details.
