# Implementation Strategies for AtoPlace v2.0

This document outlines specific algorithms and technical strategies to implement the "Human-Grade" features defined in the Product Requirements Document (PRD).

## 1. Placement Legalization Strategy

To move from "Organic" force-directed placement to "Manhattan" layouts, we will implement a multi-stage legalization pipeline.

### 1.1. Grid Snapping (The "Quantizer")
**Goal:** Ensure all components are on a user-defined grid (e.g., 0.1mm, 0.5mm).
**Algorithm:**
1.  **Input:** Dictionary of `{ref: (x, y, rotation)}` from Physics Engine.
2.  **Process:**
    *   Round `x` and `y` to nearest `grid_pitch`.
    *   Round `rotation` to nearest 90 degrees (unless component is specifically marked for 45°).
3.  **Output:** Quantized positions.

### 1.2. Row/Column Alignment (The "Beautifier")
**Goal:** Force groups of similar components (e.g., decoupling caps, resistor dividers) into strict lines.
**Algorithm:**
1.  **Detection:**
    *   Find clusters of "similar" components (same footprint, connected to same net or bus) within a spatial radius (e.g., 10mm).
    *   *Example:* 5x 0.1uF capacitors near a large MCU.
2.  **Principal Component Analysis (PCA):**
    *   Calculate the principal axis of the cluster's positions.
    *   If axis is < 45° from Horizontal: Target a **Row**.
    *   If axis is > 45° from Horizontal: Target a **Column**.
3.  **Projection:**
    *   **Row Mode:** Calculate median `Y` of the cluster. Snap all components' `Y` to this median. Sort components by `X`.
    *   **Column Mode:** Calculate median `X`. Snap all `X` to median. Sort by `Y`.
4.  **Spacing Enforcement:**
    *   Re-distribute the sorted components along the line with fixed spacing: `width + min_clearance + margin`.

### 1.3. Overlap Removal (Abacus Algorithm)
**Goal:** Resolve collisions while minimizing displacement from the global placement target.
**Algorithm:** **Abacus (Adaptive Block Alignment for CircUit Systems)**
While simple "Shove" works for small clusters, Abacus is the industry standard for minimizing total wirelength disruption.

1.  **Initialization:**
    *   Sort all components by their X-coordinate (from global placement).
2.  **Row Selection:**
    *   For each component `C`, define a "Cost Function" `Cost(C, Row_i)` based on:
        *   Vertical displacement `|C.y - Row_i.y|`.
        *   Horizontal displacement (if `C` must shift to avoid overlap in `Row_i`).
    *   Assign `C` to the `Row_i` that minimizes this cost.
3.  **Cluster Placement (Dynamic Programming):**
    *   Within each row, maintain "clusters" of contiguous components.
    *   When a new component `C` is added to `Row_i`:
        *   Place it at its preferred `x`.
        *   If it overlaps with the last cluster `Q`:
            *   Merge `C` into `Q`.
            *   Solve the quadratic optimization problem to find the new optimal `x` for the entire merged cluster `Q'` (minimizing squared displacement).
            *   Check if `Q'` overlaps with the *previous* cluster; merge recursively if needed.
4.  **Finalize:**
    *   The result is a strictly legal, non-overlapping layout that respects the "magnetic pull" of the original physics simulation.

---

## 2. Differential Pair Routing (Critical Path)

Since KiCad's Python API lacks a direct "Route Diff Pair" command, we will implement a **Geometric Planner** for Tier 1 nets.

### 2.1. Path Planning
**Algorithm: A* on a Dual-Grid**
1.  **Grid:** Create a routing grid where nodes represent *pairs* of coordinates `((x1, y1), (x2, y2))` separated by `diff_pair_gap`.
2.  **Cost Function:**
    *   `dist(start, current) + dist(current, end)` (Standard A*)
    *   Penalty for corners (45° preferred).
    *   Penalty for uncoupling (split around via/obstacle).
3.  **Output:** List of center-line points.

### 2.2. Track Generation
1.  **Trace Expansion:** From the center-line path, generate two parallel paths offset by `±(gap + width)/2`.
2.  **KiCad Object Creation:**
    *   Instantiate `pcbnew.PCB_TRACK` for each segment.
    *   Set `net` property for Positive and Negative signals.
    *   Assign `net_class` for width/clearance enforcement.

---

## 3. Fanout Generator Strategies

### 3.1. BGA Dogbone (Pitch >= 0.5mm)
**Algorithm:**
1.  **Quadrant Mapping:** Divide BGA into 4 quadrants (NW, NE, SE, SW).
2.  **Direction:** Escape traces outward away from the center.
3.  **Via Placement:**
    *   Place vias at `(pad_x + direction_x * pitch/2, pad_y + direction_y * pitch/2)`.
    *   This creates the classic "dogbone" shape.
4.  **Layer Assignment:** Assign outer rings to top layer, inner rings to progressive internal layers.

### 3.2. Via-in-Pad (Pitch < 0.5mm)
**Algorithm:**
1.  **Identification:** Trigger this mode automatically if `component.pitch < 0.5mm`.
2.  **Placement:** Place a microvia (or through-via if drilled plugged) *directly* at `pad.center`.
3.  **Escape:** Route on the *opposite* layer (or internal layers) immediately.
4.  **Cap Check:** Ensure the via definition in KiCad is set to "Tented" or "Filled" to satisfy DFM.

---

## 4. LLM Context Strategy (MCP Optimization)

To allow an LLM to "see" a board without exceeding token limits (which can easily happen with thousands of coordinates), we use a **Multi-Level Representation**.

### 4.1. Level 1: Functional Summary (The "Executive Brief")
*   **Content:** Board stats (size, layer count), list of critical modules ("Power", "MCU", "USB"), and confidence score.
*   **Usage:** Initial conversational context.

### 4.2. Level 2: Structured Netlist (The "Schematic View")
*   **Content:** `JSON` list of components with *key metadata only* (Ref, Footprint, Value, Parent Module) and connectivity graph (who connects to who).
*   **Omission:** No exact X/Y coordinates or pad geometry.
*   **Usage:** For the LLM to understand *logical* relationships ("C1 is the decoupling cap for U1").

### 4.3. Level 3: On-Demand Spatial Focus (The "Microscope")
*   **Content:** Precise X/Y coordinates, pad locations, and tracks *only* for a requested region.
*   **Mechanism:**
    *   LLM calls tool `focus_region(center_ref="U1", radius="10mm")`.
    *   System returns detailed geometry for that specific 10mm circle.
*   **Usage:** When the LLM needs to make specific move commands ("Move C1 slightly left to clear the via").

---

## 5. Atopile Integration (The "Source of Truth")

To support persistent placement without native atopile lock files, we will use a **Sidecar Pattern**.

### 5.1. `atoplace.lock` Schema
We will create a custom JSON/YAML file to store layout intent that shouldn't be overwritten by a fresh compile.

```yaml
# atoplace.lock
version: 1.0
placement:
  root.mcu.u1:
    x: 105.5
    y: 50.0
    rotation: 90
    locked: true
  root.power.c1:
    x: 102.0
    y: 50.0
    rotation: 90
    relative_to: root.mcu.u1  # Rigid group
routing:
  critical_nets:
    - usb_dp
    - usb_dn
```

### 5.2. Sync Workflow
1.  **Pre-Place:** Read `.kicad_pcb` (current state) and `atoplace.lock`.
2.  **Merge:**
    *   If component in `lock` matches `kicad`: Use `lock` position (enforce user intent).
    *   If component new in `kicad`: Place using Physics Engine.
3.  **Post-Place:** Update `atoplace.lock` with new positions of critical/user-moved components.