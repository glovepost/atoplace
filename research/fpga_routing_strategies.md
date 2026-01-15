# FPGA and High-Density Routing Strategies

**Focus:** Algorithms for complex FPGA-based PCB designs, focusing on pin optimization, negotiation-based routing, and flow-based escape.

## 1. The FPGA Advantage: Pin Swapping
Unlike fixed ICs, FPGAs and MCUs with remappable I/O allow for **Pin Swapping**. This is the single most effective optimization for PCB routing. By untangling the ratsnest *at the source* (the chip), we can eliminate vias and layer changes before routing even begins.

### 1.1 The "Uncrossing" Algorithm
**Goal:** Minimize crossing nets in the airwires (ratsnest) between two components (e.g., FPGA and Connector).

**Algorithm:**
1.  **Bipartite Matching:** Model the connections as a bipartite graph between FPGA banks and target components.
2.  **Cost Function:** Define cost based on "crossing number" or total wirelength.
3.  **Swap:** Iteratively swap pin assignments on the FPGA side to minimize cost.
    *   *Constraint:* Must respect FPGA banking rules (Voltage standards, Diff-pair polarity, Clock pins).

### 1.2 LUT-Aware Swapping
Advanced optimization considers the internal FPGA logic. Swapping two pins might require re-routing internal FPGA logic.
*   **Strategy:** Treat FPGA pins as "soft constraints" during PCB placement. Feed the optimized pinout back to the FPGA toolchain (Vivado/Quartus) via XDC/QSF files.

---

## 2. Negotiation-Based Routing (The Pathfinder Algorithm)
Standard A* creates "blocking" routes—early nets block later ones. FPGAs are too dense for this. **Pathfinder** (originally for internal FPGA routing) is highly effective for high-density PCBs.

### 2.1 The Concept
Allow nets to **overlap** (share the same space/resource) during initial iterations. Gradually increase the cost of sharing until a conflict-free solution emerges.

### 2.2 The Algorithm Loop
1.  **Initialize:** History costs = 0.
2.  **Route All Nets:** Route every net using A* (or Dijkstra). **Ignore blockages**. Allow overlaps.
3.  **Update Costs:**
    *   For every resource (grid cell/via) used by $N$ nets:
    *   $Cost_{current} = BaseCost + (N-1) \times CongestionPenalty$
    *   $Cost_{history} += (N-1) \times HistoryPenalty$
    *   $TotalCost = Cost_{current} + Cost_{history}$
4.  **Rip-Up & Reroute:** Reroute all nets using the new costs.
    *   Nets in high-congestion areas will naturally find alternative paths to avoid the high penalty.
5.  **Converge:** Repeat until $N=1$ for all resources (no overlaps).

### 2.3 Advantages for PCB
*   **100% Completion:** It rarely fails; it just takes longer to converge.
*   **Global Optimization:** Nets "negotiate" for space. Critical nets (with stricter length constraints) can "pay more" to stay on optimal paths.

---

## 3. Flow-Based BGA Escape
Escaping 1000+ pins requires a global view of layer resources.

### 3.1 Network Flow Formulation
Model the BGA escape as a **Max-Flow** problem.
*   **Nodes:** Pins, Vias, Escape Points (board edge).
*   **Edges:** Routing channels between pads.
*   **Capacity:** 1 (each channel can carry 1 trace).

### 3.2 Layer Assignment
Use **Min-Cost Max-Flow** to assign pins to layers.
*   **Cost:** Deeper layers = higher cost (more vias).
*   **Objective:** Route as many pins as possible on Top/Bottom layers. Push remaining pins to inner layers.

---

## 4. Proposed Architecture for AtoPlace FPGA Engine

### 4.1 "The Swapper" (Pin Optimizer)
A tool that runs *before* placement/routing.
*   **Input:** Netlist + FPGA Part Number + Constraints (XDC).
*   **Action:** Reassigns nets to pins to minimize total crossover.
*   **Output:** Updated Netlist + New XDC file.

### 4.2 "The Negotiator" (Pathfinder Router)
Replace the standard "Rip-up and Reroute" fallback with a true Pathfinder engine.
*   **Implementation:** Modify `AStarRouter` to allow collisions but track `congestion_score` map.
*   **Phase:** Run this for the dense "Escape Zone" around the FPGA. Use standard A* for the rest of the board.

### 4.3 "The Planner" (Flow-Based Escape)
A dedicated solver for the BGA region.
*   **Strategy:** Do not use A* for BGA escape. Use a flow-based generator to create the "fanout spokes" first.
