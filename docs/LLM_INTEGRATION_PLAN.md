# LLM Integration Implementation Plan

**Objective:** Build the "Harness" that allows an LLM to reliably perceive and manipulate PCB layouts with professional precision.

**Strategy:** "Code-as-Action" via Model Context Protocol (MCP). The LLM does not perform math; it calls high-level Python tools (DSL) which enforce geometric constraints.

---

## Phase 1: The Layout DSL (Core API)

We need a high-level, safe Python API that acts as the "hands" of the LLM.

### 1.1 Atomic Actions Module (`atoplace.api.actions`)
Create a stateless function library that performs atomic geometric operations.
*   **Movements:**
    *   `move_absolute(ref, x, y, rotation=None)`
    *   `move_relative(ref, dx, dy)`
    *   `rotate(ref, angle)`
*   **Relative Placement (The "Solver" functions):**
    *   `place_next_to(ref, target_ref, side="right", clearance=0.5, align="center")`
    *   `align_components(refs, axis="x", anchor="first")`
    *   `stack_components(refs, direction="down", spacing=0.5)`
    *   `distribute_evenly(refs, start_ref, end_ref)`
*   **Grouping:**
    *   `group_components(refs, group_name)`
    *   `lock_components(refs)`

### 1.2 State Management (`atoplace.api.session`)
*   Manage `Board` state persistence (load/save/undo).
*   Track "Dirty" state for re-running legalization/routing.

---

## Phase 2: The Microscope (Context Generator)

We need logic to give the LLM "sight" at two levels of detail.

### 2.1 Macro Context (`atoplace.mcp.context.macro`)
*   **Executive Summary:** Board stats, critical unrouted nets.
*   **Semantic Grid:** Divide board into 3x3 zones (Top-Left, Center, etc.). Return a JSON mapping components to zones.
*   **Module Map:** Hierarchical tree of functional blocks (Power, MCU).

### 2.2 Micro Context (`atoplace.mcp.context.micro`)
*   **`inspect_region(refs=[...], padding=5.0)`**:
    *   Calculates bounding box of targets.
    *   Returns high-precision JSON: exact centroids, pad locations, relative gaps.
    *   *Crucial:* Includes "Gap Analysis" - explicitly calculating distance between nearest edges of requested objects.

### 2.3 Visualizer Bridge (`atoplace.mcp.context.vision`)
*   Generate lightweight PNG/SVG of the *Microscope Viewport* for multimodal models.
*   Annotate with simple dimensions (arrows showing gap sizes).

---

## Phase 3: MCP Server Implementation

Implement the Model Context Protocol server to expose the DSL and Microscope to the LLM.

### 3.1 Server Scaffold (`atoplace.mcp.server`)
*   Initialize MCP server instance.
*   Define Resources: `board://summary`, `board://netlist`.
*   Define Tools: `place`, `route`, `inspect`, `validate`.

### 3.2 Tool Schemas
Define strict JSON schemas for every tool to prevent hallucinated parameters.
*   *Example:* `place_next_to` schema must enforce `side` enum ("left", "right", "top", "bottom").

---

## Phase 4: The Agent Loop (ReAct)

The "Foreman" agent that handles high-level intent.

### 4.1 Prompt Engineering (`atoplace.mcp.prompts`)
*   **System Prompt:** Define the persona ("Senior PCB Designer"). Enforce "Think before acting".
*   **Error Handling:** If an action fails (e.g., collision), feed the error back into the context so the LLM can retry.

### 4.2 Feedback Mechanism
*   After every `place_*` action, auto-run a lightweight `check_overlap`.
*   If overlap detected, return failure to the LLM: *"Action failed: C1 overlaps U1 by 0.2mm. Retry with larger spacing?"*

---

## Implementation Roadmap

### Step 1: Layout DSL (Days 1-2)
- [ ] Create `atoplace/api/` directory.
- [ ] Implement `atoplace.api.actions` with core movement logic.
- [ ] Add unit tests for relative placement math.

### Step 2: Microscope (Days 3-4)
- [ ] Implement `atoplace.mcp.context.micro`.
- [ ] Test JSON output for accuracy on sample board.

### Step 3: MCP Server Wiring (Days 5-6)
- [ ] Install `mcp` python package.
- [ ] Implement `atoplace.mcp.server` with basic tool registration.
- [ ] Wire CLI `atoplace serve` command.

### Step 4: Integration Test (Day 7)
- [ ] Test with Claude Desktop or similar MCP client.
- [ ] Verify "Move U1 next to J1" workflow.
