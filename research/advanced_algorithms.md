# Advanced Placement & Routing Research (2026 Update)

This document contains deep-dive research into specific algorithms required for the "Professional Polish" milestone (Abacus Legalization, Fanout Strategies, and LLM Context Optimization).

## 1. Abacus Legalization Algorithm
*Adaptive Block Alignment for CircUit Systems*

**Concept:** Abacus is a detailed placement algorithm that legalizes a global placement by moving cells as little as possible. It works row-by-row but uses dynamic programming to optimize the entire row at once, rather than placing cells greedily.

### 1.1 Algorithm Pseudocode
```python
def abacus_legalize(cells, rows):
    # Step 1: Global Sort
    # Sort all cells by their X-coordinate from global placement
    sorted_cells = sorted(cells, key=lambda c: c.global_x)

    # Step 2: Row Assignment (Projection)
    for cell in sorted_cells:
        best_row = None
        min_cost = infinity
        
        for row in rows:
            # Estimate cost: vertical displacement + approximate horizontal displacement
            # cost = |cell.global_y - row.y| + horizontal_penalty(cell, row)
            cost = calculate_projection_cost(cell, row)
            if cost < min_cost:
                min_cost = cost
                best_row = row
        
        assign_cell_to_row(cell, best_row)

    # Step 3: Detailed Placement (Dynamic Programming)
    for row in rows:
        clusters = [] # List of Cluster objects
        
        for cell in row.assigned_cells:
            # Create a new cluster with just this cell
            new_cluster = Cluster(cell)
            clusters.append(new_cluster)
            
            # Collapse clusters if they overlap
            while len(clusters) > 1:
                last = clusters[-1]
                prev = clusters[-2]
                
                # Check for overlap
                if prev.end_x > last.start_x:
                    # Merge last into prev
                    prev.merge(last)
                    clusters.pop() # Remove last
                    
                    # Solve quadratic optimization for the new merged cluster
                    # to find the optimal 'x' that minimizes total displacement
                    # for all cells in the cluster, subject to width constraints.
                    prev.optimize_position()
                else:
                    break
        
        # Finalize positions
        for cluster in clusters:
            cluster.apply_positions()
```

### 1.2 "Cluster" Optimization Logic
The core magic of Abacus is `prev.optimize_position()`.
For a cluster of `n` cells, we want to find position `x` of the *first* cell such that:
`minimize Σ (x_i - global_x_i)^2`
Subject to: `x_{i+1} = x_i + width_i` (cells are contiguous)

This reduces to a simple closed-form solution:
`optimal_x = (Σ global_x_i - Σ offset_i) / n`
Where `offset_i` is the relative position of cell `i` in the cluster.

---

## 2. BGA Fanout Strategies

### 2.1 Dogbone Fanout (Pitch >= 0.5mm)
**Usage:** Standard BGAs where a trace can physically pass between two pads.
**Geometry:**
*   **Via:** Placed at center of 4 pads.
*   **Trace:** Short segment from Pad to Via.
*   **Clearance Check:** `diagonal_distance(pad, via) - pad_radius - via_radius >= min_spacing`

### 2.2 Via-in-Pad (VIP) (Pitch < 0.5mm)
**Usage:** Fine-pitch CSP/BGA where no trace can fit between pads.
**Geometry:**
*   **Via:** Placed *exactly* at Pad Center (`dx=0, dy=0`).
*   **Drill:** Must be small (e.g., 0.15mm - 0.2mm).
*   **DFM Requirement:** Must specify "Plugged/Capped" in Fab Notes to prevent solder wicking (the "thieving" effect).
*   **Escape:** Signal goes immediately to inner layer; no top-layer trace exists.

---

## 3. LLM Context Optimization
**Problem:** A `kicad_pcb` file can be 50,000+ lines. An LLM context window is finite (and expensive).

### 3.1 Multi-Level Representation Strategy
**Level 1: Functional Summary (Context: < 500 tokens)**
*   Board Stats: `100x80mm`, `4 layers`.
*   Modules: `["Power", "ESP32", "Sensors"]`.
*   Critical Nets: `["USB_D+", "USB_D-", "ANTENNA"]`.

**Level 2: Structured Netlist (Context: 2k - 5k tokens)**
*   **Format:** JSON/YAML.
*   **Content:** Component List (Ref, Footprint, Value) + Connectivity (Adjacency List).
*   **Optimization:** Omit all coordinate data. Omit passive values unless critical (e.g., `0.1uF` is relevant, exact part number is not).

**Level 3: On-Demand Spatial RAG (Context: Variable)**
*   **Action:** LLM calls `get_region(ref="U1", radius="10mm")`.
*   **Response:** Returns detailed X/Y/Rotation for *only* the objects in that circle.
*   **Usage:** "I see a collision between C1 and R5. Move C1 to (105, 50)."

### 3.2 Token Efficiency Tactics
*   **Symbolic Refs:** Use `C1` instead of `d34a-55b...` UUIDs.
*   **Delta Coordinates:** Store positions relative to Module Centroid, not Board Origin (smaller numbers).
*   **Filtering:** Strip `Graphics` and `Text` layers from the LLM view unless specifically requested.
