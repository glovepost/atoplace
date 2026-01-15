# Natural Language Layout Control: Research & Strategy

**Applicability:** Phase 3 (Professional Agent)
**Focus:** Implementing "Text-to-PCB" capabilities using LLMs.

## 1. Core Challenge: Spatial Illiteracy
Research (e.g., *PCB-Bench*, *CAD-LLM*) consistently shows that LLMs are "spatially illiterate." They struggle to output precise floating-point coordinates (e.g., `(12.45, 33.12)`) that result in valid, non-overlapping placements.

**Common Failure Modes:**
*   **Hallucination:** Inventing coordinates that place components off-board.
*   **Drift:** "Move left" results in `x - 100` instead of `x - 5`.
*   **Overlap:** Placing components on top of each other because it cannot "see" the collision.

## 2. The Solution: "Code as Action" (Text-to-Code)
Instead of asking the LLM to *perform* the placement (outputting coordinates), we ask it to *write code* that performs the placement.

**Why this works:**
1.  **Logic over Arithmetic:** LLMs excel at logic (`if`, `for`, `function calls`) but fail at arithmetic.
2.  **Constraint Enforcement:** The underlying Python API (`atoplace.placement`) enforces the physics/legalization rules. The LLM just sets the *intent*.
3.  **Verifiability:** Generated code can be linted and dry-run before execution.

### 2.1 The "Layout DSL" (Domain Specific Language)
We expose a simplified Python API to the LLM, effectively a DSL for layout.

**Bad Prompt (Direct Coordinate):**
> "Place U1 at (10, 10) and C1 at (12, 10)."

**Good Prompt (API Call):**
> "Place U1 at the top-left anchor. Place C1 2mm to the right of U1."

**Python Translation:**
```python
u1 = board.get("U1")
c1 = board.get("C1")
actions.place_relative(u1, anchor="top_left", margin=(10, 10))
actions.place_relative(c1, target=u1, direction="right", distance=2.0)
```

## 3. Spatial Grounding Techniques

To give the LLM "sight," we must abstract geometry into tokens it understands.

### 3.1 The "Grid Compass" Abstraction
Map the continuous board space into discrete semantic regions.

*   **9-Zone Grid:** `Top-Left`, `Top-Center`, `Top-Right`, `Center-Left`, etc.
*   **Semantic Anchors:** `Near <Ref>`, `Board Edge`, `Connector Zone`.

**Context Injection:**
When prompting the LLM, we provide a "Spatial Summary" instead of raw coordinates:
```json
{
  "U1": "Center-Left (clustered with C1, C2)",
  "J1": "Bottom-Edge",
  "R1": "Unplaced"
}
```

### 3.2 Visual Feedback (Multimodal)
If using a Multimodal LLM (GPT-4o, Claude 3.5 Sonnet), we generate a lightweight SVG/PNG of the current layout.
*   **Render Style:** High-contrast blocks. No traces. Label component Refs clearly.
*   **Prompt:** "Here is the current layout. The user wants J1 on the right edge. Generate the move command."

## 4. Feedback Loops (ReAct Pattern)

We implement a **Reasoning-Action (ReAct)** loop to handle failures.

**Cycle:**
1.  **Thought:** User wants to move J1. J1 is currently at (0,0).
2.  **Action:** `move_component("J1", 100, 50)`
3.  **Observation (System):** "Error: Collision with U3 at (98, 48)."
4.  **Refined Thought:** I hit U3. I need to move J1 slightly lower to avoid it.
5.  **Refined Action:** `move_component("J1", 100, 60)`

## 5. Proposed Architecture: "The Foreman & The Workers"

### 5.1 The Foreman (Orchestrator LLM)
*   **Input:** Natural language request.
*   **Context:** Board stats, module list.
*   **Output:** Delegation to specialized sub-agents.

### 5.2 The Worker Agents (Specialized LLMs)
1.  **The Placer:** Specializes in `place_relative`, `align`, `group`. Knows DFM rules for spacing.
2.  **The Router:** Specializes in `route_net`. Knows signal integrity (diff pairs, shielding).
3.  **The Inspector:** Read-only agent. Answers "Where is C1?" or "Are there overlapping parts?".

## 6. Implementation Plan (Phase 3)

1.  **API Layer:** Build `atoplace.api.llm_interface` exposing high-level "atomic actions" (Move, Align, Group).
2.  **Context Generator:** Build `SpatialSummarizer` that converts `board.kicad_pcb` to the "Grid Compass" JSON format.
3.  **MCP Server:** Wrap the above in a Model Context Protocol server.
4.  **Agent Loop:** Implement the ReAct loop in a Python script (`atoplace/mcp/agent.py`).

## 7. References
*   *PCB-Bench*: "Benchmarking LLMs for PCB Placement and Routing" (2023).
*   *LaySPA*: "Latent Spatial Reasoning for Design" (2024).
*   *ChipNeMo*: Domain adaptation for EDA (Nvidia).
