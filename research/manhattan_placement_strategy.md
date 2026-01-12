# Manhattan Placement & Legalization Strategy

This document details the algorithmic approach for converting "organic" force-directed placements into professional "Manhattan" layouts (aligned, grid-snapped, and Design Rule compliant).

## 1. The Core Problem
Force-directed algorithms (like the one currently implemented) treat components as point masses connected by springs. This produces "cloud-like" layouts where components float at arbitrary angles and distances.
Professional PCB layouts are **Manhattan**:
*   **Grid Aligned:** Components sit on specific grids (e.g., 0.1mm, 0.5mm, 1mm).
*   **Orthogonal:** Rotations are restricted to 0°, 90°, 180°, 270°.
*   **Row/Column Structured:** Passives (decoupling caps, pull-up resistors) form strict linear arrays.
*   **Design Rule Compliant:** No overlaps, specific clearance rules respected.

## 2. The Legalization Pipeline
To bridge this gap, we implement a post-physics **Legalization Pipeline**. This pipeline runs *after* the force-directed solver converges but *before* the final placement is committed.

### Phase 1: Coarse Grid Snapping (The "Quantizer")
**Objective:** Eliminate arbitrary floating-point coordinates.
1.  **Input:** Dictionary of `{ref: (x, y, rotation)}` from the physics engine.
2.  **Grid Hierarchy:**
    *   **Primary Grid:** 0.5mm (Standard for most ICs/Connectors).
    *   **Secondary Grid:** 0.1mm (Fine pitch adjustments).
    *   **Rotation Grid:** 90° (Standard), 45° (Optional, specific components).
3.  **Algorithm:**
    *   For each component:
        *   Round `rotation` to nearest 90°.
        *   Round `x` and `y` to nearest `Primary Grid`.
        *   *Exception:* If component is "fine pitch" (e.g., 0201 caps), use `Secondary Grid`.

### Phase 2: Row/Column Alignment (The "Beautifier")
**Objective:** Detect and enforce linear structures for grouped passives.
1.  **Cluster Detection:**
    *   Identify clusters of "similar" components (same footprint, connected to same net or bus) within a spatial radius (e.g., 10mm).
    *   *Example:* 5x 0.1uF capacitors near a large MCU.
2.  **Principal Component Analysis (PCA):**
    *   Calculate the principal axis of the cluster's positions.
    *   If axis is closer to Horizontal (< 45°): Target a **Row**.
    *   If axis is closer to Vertical (> 45°): Target a **Column**.
3.  **Projection:**
    *   **Row Mode:** Calculate median `Y` of the cluster. Snap all components' `Y` to this median. Sort components by `X`.
    *   **Column Mode:** Calculate median `X`. Snap all `X` to median. Sort by `Y`.
4.  **Spacing Enforcement:**
    *   Re-distribute the sorted components along the line with fixed spacing: `width + min_clearance + margin`.

### Phase 3: Overlap Removal (The "Shove")
**Objective:** Resolve collisions created by Snapping and Alignment.
**Algorithm:** **Scanline Sweep with Separation Force**
1.  **Data Structure:** Insert all component bounding boxes (inflated by `min_clearance`) into a spatial index (R-Tree or sorted lists).
2.  **Sort:** Order components by "Priority" (Locked > Large ICs > Connectors > Passives).
3.  **Iterative Shove:**
    *   Iterate through sorted list.
    *   If `Component A` overlaps `Component B`:
        *   Calculate the **Minimum Translation Vector (MTV)** to separate them.
        *   If `B` is lower priority (or movable): Push `B` by the MTV.
        *   If `B` is locked: Push `A` by -MTV.
        *   *Constrain:* Push direction must be Manhattan (X or Y axis only) to preserve alignment.
4.  **Loop:** Repeat until 0 overlaps or `max_iterations` reached.

## 3. Implementation Details

### 3.1. Mixed-Integer Linear Programming (MILP) - *Optional Advanced Path*
For complex constraints (e.g., "Keep these 4 caps aligned AND near U1 AND don't overlap"), a MILP solver (like `CBC` or `Gurobi`) is the gold standard.
*   **Variables:** `x_i`, `y_i` for each component.
*   **Constraints:**
    *   Non-overlap: `|x_i - x_j| >= (w_i + w_j)/2` OR `|y_i - y_j| >= (h_i + h_j)/2`.
    *   Alignment: `y_i = y_j` for all `i, j` in a row group.
    *   Boundary: `x_min <= x_i <= x_max`.
*   **Objective:** Minimize `Σ (distance_moved)`.
*   *Note:* MILP is computationally expensive. We will implement the Heuristic Pipeline (Phases 1-3) first, reserving MILP for specific "high-density" zones if needed.

### 3.2. Data Structures
*   **`ManhattanGrid`**: A helper class to manage coordinate quantization.
*   **`AlignmentGroup`**: A class to track sets of components that *must* move together.
*   **`SpatialIndex`**: A lightweight R-Tree wrapper for fast collision queries.

## 4. Summary of Improvements
This strategy transforms the "Placement" engine from a simple physics simulation into a structured design tool.
*   **Before:** Components float 1.234mm apart at 3.5° rotation.
*   **After:** Components sit 0.5mm apart, aligned in a perfect row, rotated 0°.

This directly addresses the "Amateur vs. Professional" quality gap identified in the project logic review.
