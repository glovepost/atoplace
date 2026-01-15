# Analytical Placement Methods for PCB Design

**Applicability:** Phase 2 (Placement Upgrade)
**Comparison:** vs. Force-Directed (current) and Simulated Annealing

## 1. Introduction
While AtoPlace currently uses **Force-Directed (FD)** placement (physics simulation), modern VLSI (chip design) tools have moved to **Analytical Placement**. This document explores adapting algorithms like **ePlace**, **RePlAce**, and **SimPL** for PCB design.

**Key Difference:**
*   **Force-Directed:** Iterative local moves based on springs. Can get stuck in local minima. "Organic" look.
*   **Analytical:** Formulates placement as a mathematical optimization problem (minimizing a differentiable cost function). Solves for global optimum.

---

## 2. Electrostatics Analogy (ePlace/RePlAce)

### 2.1 The Concept
Treat components as positively charged particles. We want to spread them out (repulsion) while keeping connected ones close (attraction).
Instead of calculating $N^2$ pair-wise forces, we solve the **Poisson Equation** for the electric potential.

### 2.2 Mathematical Formulation
**Objective:** Minimize $W(\mathbf{x}, \mathbf{y})$ (Wirelength) s.t. Density $D(\mathbf{x}, \mathbf{y}) \le D_{target}$.

1.  **Wirelength Model (Waas):**
    *   Approximates Half-Perimeter Wirelength (HPWL) with a smooth, differentiable function (Weighted Average).
    *   $\text{HPWL} \approx \text{LSE}$ (Log-Sum-Exp).

2.  **Density Model (Electrostatics):**
    *   Discretize board into a grid (bins).
    *   Component area = "Charge".
    *   Solve for "Electric Potential" $\psi$ using DCT (Discrete Cosine Transform) or FFT (Fast Fourier Transform).
    *   **Gradient:** The gradient of potential $\nabla \psi$ gives the repulsion force direction.

### 2.3 Advantages for PCB
*   **Global View:** Sees the entire board density at once.
*   **No "Knots":** Force-directed often creates tangled knots of components. Analytical tends to "unfold" the design naturally.
*   **Speed:** FFT is $O(N \log N)$, faster than $O(N^2)$ repulsion.

---

## 3. SimPL (Simultaneous Place-and-Legalize)

**SimPL** is potentially more applicable to PCBs than pure electrostatics because PCBs have discrete, rigid blocks (large ICs) mixed with tiny dust (passives).

### 3.1 The Algorithm Loop
1.  **Global Optimization:**
    *   Solve unconstrained wirelength minimization (Quadratic Programming).
    *   Result: Components clumped together in the middle (illegal).
2.  **Legalization Step (Look-ahead):**
    *   Project the clumped components onto a legal grid/layout.
    *   *Key Idea:* Don't move them fully. Just find *where they would go*.
3.  **Force Anchoring:**
    *   Add "Anchor Springs" connecting the global position to the legal position.
    *   Repeat Step 1. The springs pull the optimization toward a spread-out solution.
4.  **Convergence:**
    *   Stiffen the springs over iterations until components settle in legal spots.

---

## 4. Adapting to PCB Realities

VLSI placement assumes millions of tiny, identical-height cells. PCBs have:
*   Huge size variance (0402 vs FPGA).
*   Rotational freedom (0/90/45).
*   Two layers (Top/Bottom).

### 4.1 2.5D Placement Strategy
Instead of true 3D, treat Top and Bottom as two separate potential fields that interact.
*   **Attraction:** Works across layers (via cost).
*   **Density:** Calculated independently for Top and Bottom layers.
*   **Layer Swapping:** Heuristic pass to flip components between layers to minimize total potential energy.

### 4.2 Handling Heterogeneity
Large components (Connectors, BGAs) act as "Blockages" in the density map.
*   **Fixed Macros:** Place connectors/critical ICs first. Mark their area as max density.
*   **Soft Logic:** Place passives/small ICs around them using the potential field gradient.

---

## 5. Implementation Roadmap for AtoPlace

Moving fully to Analytical Placement is a major rewrite. We recommend a hybrid approach:

### Step 1: "SimPL-Lite" for Global Spread
*   Current: Random/Center init -> Force Directed.
*   New: Quadratic Solve (Star Model) -> Simple Spreading -> Force Directed.
*   *Benefit:* Starts the physics engine with an untangled, globally sensible layout.

### Step 2: FFT-Based Density (Replacing Repulsion)
*   Replace the $O(N^2)$ or Spatial Hash repulsion with a density grid.
*   Compute density map -> FFT -> Potential -> Gradient -> Repulsion Force.
*   *Benefit:* Much smoother spreading, natural "void filling".

### Step 3: Mixed-Size Binning
*   Bin the board into e.g., $1mm \times 1mm$ cells.
*   Large ICs occupy many bins. Small passives share bins.
*   Optimization targets uniform bin utilization.

---

## 6. References
*   **ePlace:** Lu et al., "ePlace: Electrostatics-based Placement using Fast Fourier Transform and Nesterov's Method", ACM TODAES 2015.
*   **SimPL:** Kim et al., "SimPL: An Effective Placement Algorithm", ICCAD 2010.
*   **RePlAce:** "Advancing Solution Quality and Routability Validation in Global Placement", IEEE TCAD 2019.
