# BGA and High-Density Fanout Algorithms

**Applicability:** Phase 2B (Routing Assistant)
**Focus:** Escape routing for BGA, QFN, and fine-pitch connectors.

## 1. Introduction
Autorouters often fail at the very start: escaping the dense pin grid of a BGA (Ball Grid Array). "Fanout" is the procedural generation of short traces and vias to break signals out from the component body to the main routing field.

---

## 2. Fanout Patterns

### 2.1 Dogbone (Standard)
For pitch $\ge 0.65mm$.
*   **Geometry:**
    *   Pad at $(x, y)$.
    *   Via at $(x \pm d, y \pm d)$.
    *   Trace connecting Pad to Via.
*   **Algorithm:**
    *   For each pin, determine "Escape Quadrant" (NW, NE, SE, SW).
    *   Place via in the center of 4 pads.
    *   Check DRC (clearance to adjacent pads).

### 2.2 Via-in-Pad (VIP)
For pitch $\le 0.5mm$.
*   **Geometry:**
    *   Via placed *directly* in pad center.
    *   Requires "Plugged & Capped" manufacturing (POFV).
*   **Algorithm:**
    *   Simple substitution: Pad $\rightarrow$ Pad + Via.
    *   Routing happens entirely on inner/bottom layers.

### 2.3 Channel Routing (Microwave/RF)
For high-speed edges.
*   **Pattern:** Ground vias surrounding signal pads to create a pseudo-coaxial structure.

---

## 3. Layer Assignment Algorithms

### 3.1 The "Onion Layer" Strategy
For large BGAs (e.g., 20x20 grid), not all pins can escape on the top layer.
*   **Ring 1 (Outermost):** Route on Top Layer.
*   **Ring 2:** Route on Bottom Layer (via).
*   **Ring 3:** Route on Inner Layer 1 (via).
*   **Ring N (Center):** Deepest layers.

**Algorithm:**
1.  **Ring Analysis:** Identify which "ring" (distance from edge) a pin belongs to.
    *   $R = \min(row, col, max\_row-row, max\_col-col)$.
2.  **Layer Mapping:** Assign rings to layers based on stackup.
    *   $Layer = f(R)$.
3.  **Spiral Escape:**
    *   For pins in Ring $R$ on Layer $L$:
    *   Route outward using a "spiral" or "channel" pattern to avoid blocking inner pins.

---

## 4. Escape Routing Algorithms

### 4.1 Flow-Based Routing
Treat the BGA escape as a network flow problem.
*   **Source:** Component pins.
*   **Sink:** Board edges or perimeter of the BGA courtyard.
*   **Capacity:** Number of wires that can fit between two vias.

### 4.2 Ordered Escape (Monotonic)
1.  **Sort Pins:** Sort pins by angle from center.
2.  **Sweep:** Iterate radially.
3.  **Route:** Project a ray from pin to boundary. If clear, convert to trace.

---

## 5. Implementation Strategy for AtoPlace

### Phase 1: Procedural Fanout Generator
Implement a `FanoutGenerator` class that applies templated patterns.

```python
class FanoutGenerator:
    def generate(self, component, strategy="auto"):
        if strategy == "auto":
            if component.pitch <= 0.5:
                self.apply_via_in_pad(component)
            else:
                self.apply_dogbone(component)

    def apply_dogbone(self, comp):
        # 1. Calculate quadrant for each pin
        # 2. Place via
        # 3. Add short trace
```

### Phase 2: Escape Router
A dedicated mini-router that runs *before* the main A* router.
*   **Input:** BGA Component.
*   **Output:** Traces extending 2-5mm outside the BGA boundary.
*   **Logic:**
    *   Route outermost rows first.
    *   Push traces "away" from center.
    *   Stop once "clear air" is reached.

### Phase 3: Diff Pair Fanout
Special handling for diff pairs in BGAs.
*   **Rule:** Vias must be symmetrical.
*   **Pattern:**
    *   Standard Dogbone adds skew (vias are diagonal).
    *   **Flag Fanout:** Vias placed perpendicular to pads to maintain P/N symmetry.

---

## 6. References
*   **"Escape routing for dense BGA components"**, various EDA whitepapers.
*   **"Ordered Escape Routing"**, ICCAD papers on monotonic routing.
