# FPGA & Advanced Routing Implementation Plan

**Objective:** Upgrade AtoPlace from a general-purpose layout tool to an "FPGA-Capable" design assistant.

**Strategy:** Implement three key modules: Pin Swapping, Negotiation-Based Routing (Pathfinder), and Flow-Based Escape.

---

## Phase 1: The Swapper (Pin Optimization)
**Target:** Q2 2026
**Value:** Reduces routing complexity by 50% by untangling the ratsnest at the source.

### 1.1 FPGA Constraint Parser (`atoplace.fpga.parser`)
*   **Support:** Xilinx (XDC) and Lattice (PDC).
*   **Logic:**
    *   Parse constraints to identify "Swap Groups" (pins that can be interchanged).
    *   Example: "Bank 35 DQS pins are fixed, but DQ pins are swappable within byte lane."

### 1.2 Crossing Minimizer (`atoplace.fpga.optimizer`)
*   **Algorithm:** Bipartite Matching (Weighted).
*   **Inputs:** FPGA component, Connected components (e.g., DDR4 RAM).
*   **Process:**
    1.  Project all connected components to the FPGA boundary.
    2.  Calculate ideal "entry vectors" for each net.
    3.  Assign nets to available physical pins that minimize vector deviation.
*   **Output:** Generates a new Netlist and updated XDC file.

---

## Phase 2: The Negotiator (Pathfinder Router)
**Target:** Q3 2026
**Value:** Solves 100% of routes in dense BGA fields where standard A* fails.

### 2.1 Congestion Map (`atoplace.routing.congestion`)
*   **Data Structure:** Spatial grid tracking `usage_count` per cell.
*   **Logic:**
    *   Allow `usage_count > 1` (illegal overlap).
    *   Cost function: `base_cost + (usage_count - 1) * penalty`.

### 2.2 Iterative Router (`atoplace.routing.pathfinder`)
*   **Logic:**
    *   Loop `iteration` from 1 to `max_iter`.
    *   `rip_up_all()`: Clear all route geometries (but keep history costs).
    *   `route_all()`: Route every net using A* with congestion costs.
    *   `update_history()`: Increase penalty for cells that are *still* congested.
    *   Stop when max congestion == 1 (legal layout).

---

## Phase 3: The Planner (BGA Escape)
**Target:** Q3 2026
**Value:** Automates the most tedious part of FPGA layout.

### 3.1 Flow Solver (`atoplace.routing.escape`)
*   **Algorithm:** Max-Flow Min-Cost.
*   **Logic:**
    *   Define "Rings" of pads (onion layers).
    *   Assign outermost ring to Top Layer.
    *   Assign next ring to Layer 2 (via-in-pad).
    *   Generate "Channels" (gaps between vias) as graph edges.
    *   Push flow (signals) from inner pins to board edge.

### 3.2 Fanout Generator Integration
*   Integrate `research/bga_fanout_algorithms.md` findings.
*   Procedurally place vias and dogbones *before* running the flow solver.

---

## Roadmap Integration

### Milestone D: FPGA Alpha
- [ ] XDC Parser for Pin Swapping.
- [ ] Bipartite Matching Optimizer.
- [ ] Pathfinder-based "Rip-up and Reroute" engine.

### Milestone E: High-Density Routing
- [ ] Flow-based BGA escape.
- [ ] Via-in-Pad support in `FanoutGenerator`.
- [ ] Layer assignment strategy ("Onion Layer").
