# The Comprehensive PCB Layout Design Handbook

## 1. Introduction & Core Philosophy
This handbook serves as a definitive guide for high-speed, RF, and mixed-signal PCB design. It aggregates industry best practices, expert consensus (from TI, ADI, NXP, EEVblog), and practical "war stories" to prevent common pitfalls.

### The Golden Rules
1.  **Return Current Paths**: Current flows in the path of least impedance. For AC (>100kHz), this is *directly underneath* the signal trace. Never route over splits.
2.  **Modularity**: Partition the board logically (Analog, Digital, RF, Power). Keep signals within their partitions.
3.  **Inductance is the Enemy**: Minimize loop areas everywhere—power, ground, and signal.

---

## 2. Stack-up & Material Selection
*   **Material**:
    *   **RF/High-Speed**: Use Rogers RO4000/RO3000 or Panasonic Megtron 6 for signals >1GHz. Standard FR-4 (Isola 370HR) is acceptable for digital <1GHz if traces are short.
    *   **Dielectric Constant (Dk)**: Lower is better for speed (propagation delay $\approx \sqrt{Dk}$). Stable Dk over frequency/temp is critical for RF.
*   **Layer Stack**:
    *   **Ground Reference**: Every signal layer must have an adjacent solid ground reference plane.
    *   **Symmetry**: Balance copper weights and dielectric thicknesses from the center out to prevent warping.
    *   **4-Layer Standard**: Signal / GND / PWR / Signal (Good) OR Signal+GND / GND / GND / Signal+GND (Better for EMI).

---

## 3. Power Delivery Network (PDN)
* Goal: Provide a low-impedance voltage source from DC up to the switching frequency of the IC.*

### Decoupling Capacitors
*   **Placement**: Place **CLOSEST** to the IC power pin. Priority: Smallest value cap closest to pin.
*   **Loop Inductance**: This is the killer.
    *   **Mounting**: Use "Side-Mount" vias (vias next to pads) rather than "End-Mount" (traces to vias).
    *   **Via Placement**: Locate vias as close to the capacitor leads as manufacturing allows.
    *   **Opposing Vias**: Place power and ground vias close together to allow magnetic field cancellation.
*   **Selection**: Use a mix of decades (e.g., 10uF, 100nF, 1nF) only if verified by simulation to avoid anti-resonance peaks. Otherwise, multiple caps of the same optimal value (e.g., 100nF or 1uF) in parallel is often safer and lower inductance.
*   **Ferrite Beads**:
    *   **Warning**: Do not use on digital power rails blindly. They can cause ringing with decoupling caps (LC resonance).
    *   **Usage**: Effective for isolating clean PLL/Analog rails. Always damp with a series resistor or bulk capacitor if there's a risk of resonance.

---

## 4. Switching Regulators (Buck/Boost)
*Ideally, use modules with integrated inductors for best EMI. If laying out discrete:*

### The "Hot Loop"
*   **Definition**: The path with high $di/dt$ switching current.
    *   *Buck*: Input Capacitor $\rightarrow$ High-Side FET $\rightarrow$ Low-Side FET $\rightarrow$ Ground $\rightarrow$ Input Capacitor.
    *   *Boost*: Low-Side FET $\rightarrow$ Diode $\rightarrow$ Output Capacitor $\rightarrow$ Ground $\rightarrow$ Low-Side FET.
*   **Rule**: **Minimize this loop area above all else.** Shave millimeters.
*   **Placement**: Place the input/output capacitor *immediately* adjacent to the FET switches.
*   **Grounding**: The ground of the Input Cap, Output Cap, and Sync-FET must return to a dedicated "Power Ground" island before joining the main ground plane.

### Feedback Nodes
*   **Isolation**: The feedback (FB) trace is high-impedance and sensitive. Keep it away from the switch node and inductor.
*   **Routing**: Route on a different layer shielding it with ground if possible.

---

## 5. EMI/EMC Design Rules
*   **Edge Plating & Shielding**:
    *   **Edge Plating**: Plate the PCB edges and stitch to Ground to create a Faraday cage, preventing internal layers from radiating out the sides.
    *   **Guard Rings**: For RF, use a ground ring of vias around the circuit block. Spacing $<\lambda/20$.
*   **Stitching Vias**: Stitch ground planes together liberally.
    *   **Rule of Thumb**: Max spacing 5mm (conservative) or $\lambda/10$ of highest frequency.
    *   **Edges**: Stitch board edges every 3-5mm to prevent "patch antenna" effects.
*   **Connectors**: Connector shells *must* be grounded to the chassis/enclosure ground immediately at entry (360-degree bonding best).

---

## 6. Thermal Management
*   **Thermal Vias**:
    *   **Placement**: In the exposed pad (ePad) of the IC.
    *   **Size**: 0.3mm (12 mil) drill is standard. Smaller can be filled/capped.
    *   **Quantity**: More is better. Pitch ~1.0mm-1.2mm grid.
    *   **Connection**: Direct connect to internal Ground planes (no thermal reliefs on thermal vias).
*   **Heatsinks**:
    *   **Grounding**: Ground metal heatsinks to the PCB ground plane to prevent them becoming radiating antennas.

---

## 7. Manufacturability (DfM) Rules
*   **Fiducials**:
    *   Three global fiducials (L-configuration) for panel/board alignment.
    *   Local fiducials for fine-pitch BGAs/QFNs.
    *   Size: 1mm copper dot, 2mm mask opening.
*   **Acid Traps**: Avoid acute angles ($<90^\circ$) in traces. Use chamfered corners ($45^\circ$) or curves.
*   **Solder Mask Dams**:
    *   Min width: 4 mil (green), 5 mil (other colors).
    *   Critical for preventing bridges on fine-pitch ICs.
*   **Annular Rings**:
    *   Min: 4-5 mil over drill size (check fab house capabilities).
    *   Use teardrops on trace-to-pad entries to prevent breakout.

---

## 8. Specific "Deep Dive" Scenarios
### The Ground Wars (Solid vs. Split)
*   **Verdict**: **Solid Ground Plane Wins.**
*   Do not split ground planes to separate Analog/Digital. It creates return path dipoles.
*   Instead, **Partition** component placement. Keep digital current in the digital area and analog in the analog area.

### Crystal Layout
*   **Placement**: Next to MCU pins.
*   **Grounding**: Solid ground plane underneath is usually preferred (shields noise). *Exception*: Specific low-power RTCs where manual says remove copper to reduce parasitic capacitance.
*   **Guard Ring**: A grounded trace ring around the crystal on the top layer helps contain local noise.

### Differential Pairs
*   **Matching**: Length match within tolerance (e.g., USB 2.0 $\pm$150 mil, PCIe much tighter).
*   **Coupling**: Loosely coupled limits crosstalk. Tightly coupled improves common-mode rejection. Follow the specific impedance target (usually $90\Omega$ or $100\Omega$ differential).
