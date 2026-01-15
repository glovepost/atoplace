# Competitor Analysis & Integration Strategy

**Date:** 2026-01-13
**Subject:** Existing LLM-for-PCB solutions and AtoPlace's unique positioning.

## 1. Landscape Analysis

### 1.1 mixelpixx/KiCAD-MCP-Server
*   **Approach:** A direct MCP implementation for KiCad.
*   **Strengths:** Broad toolset, JLCPCB integration, IPC-based control.
*   **Weakness:** Focuses on *automation primitives* (raw API access) rather than *intelligent aesthetics*. It lets the LLM "do anything," which often leads to messy layouts because LLMs are bad at global optimization.
*   **AtoPlace Differentiator:** We are not just an API; we are an **Optimization Engine**. Our MCP tools invoke high-level solvers (Abacus, Star Model), ensuring the result is professional even if the LLM's spatial reasoning is weak.

### 1.2 Flux.ai Copilot
*   **Approach:** Deeply integrated commercial SaaS agent.
*   **Strengths:** Multimodal vision ("See this diagram"), part selection, schematic-to-layout continuity.
*   **Weakness:** Closed ecosystem (Flux only). Cloud-based (privacy concerns).
*   **AtoPlace Differentiator:** **Open & Local**. We bring Flux-like intelligence to the industry-standard KiCad, running entirely on the user's machine (Phase 3).

### 1.3 Kicad-LLM-Plugin (Jasiek)
*   **Approach:** Inspection/Review plugin.
*   **Strengths:** Great for finding flaws in schematics.
*   **Weakness:** Read-only analysis. Doesn't "do" the layout.
*   **AtoPlace Differentiator:** **Action-Oriented**. We move components and route traces.

---

## 2. Strategic Positioning

**"The AI Pair Designer for KiCad"**

We do not aim to replace the designer. We aim to replace the *tedium* of layout.

*   **Competitors:** "Generate a board from this prompt." (Low success rate, messy).
*   **AtoPlace:** "Here is a messy pile of components. Organize them into a professional layout." (High success rate, clean).

## 3. Integration Learnings

1.  **Token Efficiency is Key:** Projects like `kicad-netlist-tool` prove that raw files are too big. Our **Multi-Level RAG** (Macro/Micro) is the correct architectural choice.
2.  **Tool Granularity:** Exposing raw `set_position(x,y)` is dangerous. Successful agents use "Semantic Actions" (`place_next_to`, `align`). Our **Layout DSL** (`atoplace.api`) aligns perfectly with this best practice.
3.  **Visual Grounding:** Flux.ai uses vision. We will approximate this with our "Microscope" (JSON geometry) and potentially SVG snapshots in the future.

## 4. Revised Tool Definition (MCP)

Based on this analysis, our MCP server must expose **Solvers**, not just **Setters**.

*   ❌ `move_component(ref, x, y)` -> Too low level. LLM fails math.
*   ✅ `legalize_placement(refs)` -> Invokes our Manhattan Legalizer.
*   ✅ `cluster_components(refs)` -> Invokes Force-Directed physics.
*   ✅ `route_critical_net(net)` -> Invokes A* Planner.

This "Engine-First" approach is our moat.
