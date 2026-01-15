# Differential Pair and Bus Routing Algorithms

**Applicability:** Phase 2B (Routing Assistant)
**Focus:** Tier 1 Critical Nets (USB, Ethernet, PCIe, DDR)

## 1. Introduction
Differential pairs and high-speed buses require strict geometric constraints that standard A* pathfinding (which treats nets independently) cannot satisfy. This document details algorithms for **Coupled Path Planning** and **Length Matching**.

---

## 2. Differential Pair Routing

### 2.1 The "Dual-Grid" A* Approach
Standard A* routes a single line. Differential pairs are two coupled lines with a fixed gap ($g$) and width ($w$). 

**Algorithm:**
Instead of routing trace edges, we route the **virtual centerline**.

1.  **Virtual Obstacle Inflation:**
    *   Inflate all obstacles by $w + g/2 + clearance$.
    *   This ensures that if the centerline passes a point, *both* tracks fit.
2.  **Anisotropic Cost Function:**
    *   Movement along the pair's axis is cheap.
    *   Bending is expensive (requires creating a "structure" to maintain phase).
    *   **Uncoupling Penalty:** We allow the pair to split around a small obstacle (like a via), but apply a massive cost for every mm of uncoupled length.
3.  **Structure Generation:**
    *   Once the centerline $P_{center}$ is found, generate left $P_L$ and right $P_R$ paths:
        *   $P_L = P_{center} + normal \times (g + w)/2$
        *   $P_R = P_{center} - normal \times (g + w)/2$
    *   **Phase Matching:** For corners, the inner track is shorter. We must record the accumulated phase error $\Delta \phi$.

### 2.2 The "River Routing" Algorithm (for Buses)
Used for DDR/Parallel buses where N nets must flow together without crossing.

**Concept:**
Treat the bus as a "ribbon" of width $N \times (w + g)$.
1.  **Global Path:** Find a path for the ribbon using A* with inflated radius.
2.  **Track Assignment:**
    *   Assign nets to tracks in the ribbon based on pin ordering at start/end.
    *   **Planar check:** If the pin ordering at Start is `[1, 2, 3]` and End is `[3, 2, 1]`, a layer change (twist) is required.

---

## 3. Length Matching Algorithms (Meandering)

Length matching is a post-routing optimization to equalize propagation delay.

### 3.1 The "Accordion" Algorithm
**Goal:** Add length $L_{add}$ to a trace segment without violating clearance.

1.  **Segment Selection:**
    *   Identify segments of the trace that have empty space perpendicular to the direction of travel.
    *   Use a **Spatial Interval Tree** to query "gap size" on both sides of the segment.
2.  **Amplitude & Pitch Calculation:**
    *   Let $A$ be amplitude (height of bump) and $P$ be pitch (width of bump).
    *   Added length per bump $\approx 2A$.
    *   Max $A$ is limited by neighbor clearance.
    *   Min $P$ is limited by DFM (typically $2w$).
3.  **Geometry Generation:**
    *   Replace straight segment with a specific pattern:
        *   **Serpentine:** Standard U-turns.
        *   **Trombone:** One large loop (good for localized matching).
        *   **Sawtooth:** 45-degree zig-zags (better for impedance).

### 3.2 Phase Matching (Dynamic)
For diff pairs, length matching must happen *locally* (near the bend that caused the skew).

**Algorithm:**
1.  **Walk the Pair:** Iterate through segments $S_i$.
2.  **Accumulate Skew:** $\Delta L += len(Outer_i) - len(Inner_i)$.
3.  **Compensation:**
    *   If $\Delta L > Tolerance$:
        *   Insert a small "bump" on the shorter trace immediately after the bend.
        *   This creates a "structure" (Start Bend -> Straight -> Comp Bump -> End).

---

## 4. Impedance-Controlled Routing

### 4.1 Layer-Dependent Constraints
Impedance ($Z_0$) depends on trace width ($w$) and height ($h$) above reference.
*   **Rule:** The router must support *layer-specific widths*.
*   **Implementation:**
    *   `RouteNode` in A* includes `layer`.
    *   `trace_width(layer)` is a lookup function, not a constant.
    *   When exploring a VIA edge (`layer` -> `layer+1`), the cost function must assume the width changes.

### 4.2 Return Path checking
**Rule:** High-speed traces must not cross split planes.

**Algorithm (Geometric Check):**
1.  **Plane Rasterization:** Rasterize all reference planes (GND/PWR) into a boolean grid or polygon set.
2.  **Split Detection:**
    *   For each routing segment $(p1, p2)$:
    *   Project $(p1, p2)$ onto the reference layer below.
    *   Check if the projection intersects a "void" or "gap" in the reference polygon.
3.  **Cost Penalty:**
    *   If intersection > 0: Apply `InfiniteCost` (soft block) or hard fail.
    *   **Stitching Via:** If a layer change is required, check for a nearby ground via. If none, add cost to "create stitching via".

---

## 5. Implementation Strategy for AtoPlace

### Phase 1: Diff Pair Primitive
*   Implement `DiffPairRouter` class inheriting from `AStarRouter`.
*   Override `_get_neighbors` to generate dual-node steps.
*   Input: `(start_p, start_n)`, `(end_p, end_n)`.

### Phase 2: Simple Tuning
*   Implement `AccordionTuner` class.
*   Input: `RouteSegment`, `target_length`.
*   Output: List of new points replacing the segment.

### Phase 3: Validation
*   Add `ImpedanceChecker` to the validation engine.
*   Simple geometric check: `Width / Dielectric_Height` ratio verification.
