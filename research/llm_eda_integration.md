# LLM and AI in EDA: Integration Strategies

**Applicability:** Phase 3 (Professional Agent)
**Focus:** Context management, RAG strategies, and Agentic workflows for PCB Design.

## 1. Introduction
Integrating Large Language Models (LLMs) into EDA (Electronic Design Automation) faces a unique challenge: **Context Window limits vs. Design Complexity**. A PCB file is massive (coordinates, netlists, rules). An LLM cannot "see" the whole board at once. This document outlines strategies to bridge this gap.

---

## 2. Context Management: The "LOD" Strategy

Video games use **Level of Detail (LOD)** to render vast worlds. We apply this to EDA context.

### 2.1 Multi-Level RAG (Retrieval-Augmented Generation)

**Level 0: Executive Summary (The "Metadata" View)**
*   **Tokens:** ~500
*   **Content:** Board dimensions, layer count, stackup, critical stats (unrouted nets, DRC violations), list of major functional blocks ("Power", "MCU", "Radio").
*   **Use Case:** High-level planning, status checks.

**Level 1: The Logical Graph (The "Netlist" View)**
*   **Tokens:** 2k - 5k
*   **Content:**
    *   `U1` (MCU) connects to `J1` (USB) via `USB_D+`.
    *   `C1, C2` are decoupling for `U1`.
    *   *Crucial:* NO coordinate data. Only topological relationships.
*   **Use Case:** Understanding circuit intent, grouping components, planning placement strategy.

**Level 2: The Spatial Microscope (The "Region" View)**
*   **Tokens:** Variable (windowed)
*   **Content:** Precise X/Y/Rotation coordinates, trace segments, and pads *only for a specific bounding box*.
*   **Tool:** `get_region_context(center_ref="U1", radius="20mm")`.
*   **Use Case:** Detailed placement adjustment, resolving local DRC errors, "Move C1 to the left of U1".

---

## 3. Agentic Workflows

Instead of a single "Chat with PCB" prompt, we use specialized sub-agents.

### 3.1 The "Architect" Agent
*   **Role:** High-level planning.
*   **Input:** User prompt ("Place the power supply").
*   **Tools:** `list_modules()`, `get_board_stats()`.
*   **Output:** Delegation plan ("1. Select power components. 2. Group them. 3. Hand off to Placement Agent").

### 3.2 The "Placement" Agent
*   **Role:** Geometric optimization.
*   **Input:** List of components to place.
*   **Tools:** `move_component()`, `rotate_component()`, `check_overlaps()`.
*   **Loop:** Propose Move -> Validation Check -> Refine Move.

### 3.3 The "Routing" Agent
*   **Role:** Pathfinding strategy.
*   **Input:** Net to route.
*   **Tools:** `route_net_astar()`, `rip_up_net()`, `get_congestion_map()`.
*   **Strategy:** Doesn't route segment-by-segment. Calls the algorithmic router (A*) and reviews the result/failure log.

---

## 4. ChipNeMo Insights (Applied to PCB)

Nvidia's **ChipNeMo** paper demonstrated that domain-adaptive pretraining is powerful, but RAG is often sufficient for CAD tool control.

**Key Takeaways for AtoPlace:**
1.  **Tool-Use is King:** The LLM should be a *caller of algorithms*, not the algorithm itself. It shouldn't calculate `(x, y)` coordinates; it should call `align_row([c1, c2, c3])`.
2.  **Schema Enforcement:** Strict JSON schemas for tool outputs prevent "hallucinated" coordinates.
3.  **Code Generation vs. API Calls:** Generating Python scripts (to be executed by KiCad's API) is often more robust than emitting single actions. It allows loops, math, and logic to happen in Python, not the LLM.

---

## 5. Implementation Roadmap (MCP Server)

**Model Context Protocol (MCP)** is the standard for connecting LLMs to external data/tools.

### 5.1 Resources (Passive Data)
*   `board://summary`: Level 0 context.
*   `board://netlist`: Level 1 context.
*   `board://drc_report`: Textual report of errors.

### 5.2 Tools (Active Control)
*   `place_component(ref, x, y, rot)`: Atomic move.
*   `route_net(net_name, strategy)`: Invoke algorithmic router.
*   `run_legalization()`: Invoke the Manhattan Legalizer.
*   `inspect_area(x, y, w, h)`: Return Level 2 context (ASCII art or JSON).

### 5.3 Prompts (Guided Workflows)
*   "Optimize Placement": A structured prompt template that guides the LLM to inspect modules, plan grouping, and execute moves.
*   "Fix DRC": Fetches the DRC report and iterates on fixes.

---

## 6. References
*   **ChipNeMo:** Nvidia Research (2023), "Domain-Adapted LLMs for Chip Design".
*   **AutoCAD / Copilot:** Industry trends in CAD assistants.
*   **Model Context Protocol:** Anthropic open standard.
