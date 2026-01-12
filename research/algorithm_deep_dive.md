# Algorithm Deep Dive: Routing, Physics, and AI

This document details advanced algorithms for the "Professional Polish" and future AI milestones, synthesized from deep-dive research.

## 1. A* Cost Functions for PCB Routing

To achieve professional-quality routing, the A* cost function `f(n) = g(n) + h(n)` must be specifically tuned for PCB physics, not just shortest path.

### 1.1 The Cost Function `g(n)`
The accumulated cost `g(n)` should be a weighted sum:

`g(n) = w_len * length + w_via * vias + w_layer * layer_penalty + w_bend * bend_penalty + w_wrong * wrong_way`

*   **Layer Penalty (`w_layer`):**
    *   Assign specific costs to each layer based on stackup preference.
    *   *Example:* Top/Bottom = 1.0 (preferred), Inner1/Inner2 = 5.0 (avoid if possible).
    *   *Dynamic:* If a net is marked "High Speed", inner stripline layers might actually be *preferred* (lower cost).

*   **Via Penalty (`w_via`):**
    *   High base cost (e.g., equivalent to 10-20mm of trace) to discourage layer hopping.
    *   *Stacked Via Penalty:* Additional cost if vias are stacked (expensive to mfg) vs staggered.

*   **Wrong-Way Routing (`w_wrong`):**
    *   Define a "Preferred Direction" for each layer (e.g., Layer 1: Horizontal, Layer 2: Vertical).
    *   If a segment travels perpendicular to the preferred direction, multiply its length cost by `factor` (e.g., 2.5x).
    *   *Result:* This naturally forms "Manhattan" routing grids and reduces blocking.

*   **Bend Penalty (`w_bend`):**
    *   Cost for changing direction.
    *   *Critical:* 45-degree bends = small cost. 90-degree bends = high cost (acid traps, reflections).

### 1.2 The Heuristic `h(n)`
*   **Manhattan Distance:** Standard `|dx| + |dy|`.
*   **Octile Distance:** Better for PCBs allowing 45-degree routing. `max(|dx|, |dy|) + (sqrt(2)-1) * min(|dx|, |dy|)`.
*   **3D Extension:** `h(n)` must include estimated Z-distance (layer changes) to guide the search towards the target layer.

---

## 2. Force-Directed Physics: The "Star Model"

To solve the $O(N^2)$ explosion for high-degree nets (GND/VCC), we implement the Star Model.

### 2.1 Standard vs. Star
*   **Clique Model (Standard):** Every pin connects to every other pin.
    *   Edges: $N(N-1)/2$.
    *   Force: Massive attraction collapsing the net to a point.
*   **Star Model:** Introduce a virtual "Net Centroid" node.
    *   Edges: $N$.
    *   Force: Pins attracted to Centroid; Centroid attracted to Pins.

### 2.2 Weight Scaling (`w_net`)
For a net with $k$ pins, the spring constant $k_{spring}$ must be scaled to prevent large nets from dominating the placement.

`k_spring = base_strength / (k - 1)`

*   **Reasoning:** This normalizes the total force exerted by the net, independent of its pin count. A 2-pin net and a 50-pin net will exert roughly similar total pulls on their components.

### 2.3 Repulsion Optimization (Barnes-Hut / FMM)
For N > 500 components, $O(N^2)$ repulsion is too slow.
*   **Quadtree Decomposition:** Group distant nodes into a single "center of mass".
*   **Cutoff Radius:** If `dist(A, B) > 50mm`, assume repulsion is 0.

---

## 3. Reinforcement Learning (RL) for Routing

Research indicates RL is viable but requires specific state/action modeling.

### 3.1 State Representation
A "Rasterized Image" + "Graph" hybrid works best.
*   **Global Map:** Low-res grid (e.g., 1mm cells) showing congestion/blockages.
*   **Local Window:** High-res grid (e.g., 0.1mm cells) centered on the "Agent" (current trace tip).
*   **Graph Features:** GNN embedding of the netlist to inform "which net is next".

### 3.2 Action Space
*   **Discrete Movement:** Move N/S/E/W/NE/NW/SE/SW.
*   **Layer Change:** Via Up / Via Down.
*   **Action Masking:** invalid moves (DRC violation, off-board) are masked out before softmax.

### 3.3 Reward Function
*   `Step`: -1 (minimize length).
*   `Via`: -10 (minimize vias).
*   `Target Reached`: +1000.
*   `DRC Violation`: -500 (terminate episode).

---

## 4. Topological Routing vs. Maze Routing

*   **Maze Routing (Geometric):**
    *   *Pros:* Simple, DRC-compliant by definition (if grid is correct).
    *   *Cons:* "Blocking" - early routes block later ones. Hard to "shove".
*   **Topological Routing:**
    *   *Concept:* Route represents a "path class" relative to obstacles (e.g., "Pass North of C1, South of C2").
    *   *Pros:* 100% completion rates (mathematically). Flexible geometry (rubber-banding).
    *   *Cons:* Hard to convert back to geometric DRC-compliant traces.
*   **Verdict for AtoPlace:** Geometric Maze (A*) is preferred for the "Critical Path" planner (Diff Pairs) due to strict impedance/coupling rules. Topological is better for the general "autoroute" phase (Freerouting uses a topological-like approach internally).
