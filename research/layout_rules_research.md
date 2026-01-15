# The Comprehensive PCB Layout Design Handbook

Internal knowledge base for automated layout rules. Each rule is split into
Rule / Applies to / Rationale / Source. Add primary sources before encoding any
rule into automation.

## 1. Introduction & Core Philosophy

### 1.1 Return Current Paths
- Rule: Route AC signals with a continuous reference plane directly beneath the trace; do not cross plane splits.
- Applies to: Signals >100 kHz and any edge-rate-limited digital nets.
- Rationale: Return current follows the path of least impedance, which is the closest reference plane.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 1.2 Functional Partitioning
- Rule: Partition the board into analog, digital, RF, and power regions; keep signals within their region.
- Applies to: Mixed-signal and RF designs.
- Rationale: Limits coupling and makes return paths predictable.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 1.3 Minimize Loop Areas
- Rule: Minimize loop areas for power, ground, and high-speed signals.
- Applies to: All switching, RF, and high-speed nets.
- Rationale: Large loops increase inductance and EMI.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 2. Stack-up & Material Selection

### 2.1 Substrate Choice
- Rule: Use low-loss materials (e.g., RO4000/RO3000, Megtron 6) for >1 GHz; FR-4 is acceptable below 1 GHz for short traces.
- Applies to: RF and high-speed digital.
- Rationale: Lower loss and stable Dk reduce dispersion and attenuation.
- Source: IPC-4101 (Specification for Base Materials for Rigid and Multilayer Printed Boards).

### 2.2 Reference Planes
- Rule: Every signal layer must have an adjacent, solid reference plane.
- Applies to: All multilayer designs.
- Rationale: Controls impedance and ensures return current continuity.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

### 2.3 Stack Symmetry
- Rule: Balance copper weights and dielectric thicknesses around the core.
- Applies to: Multilayer boards.
- Rationale: Reduces warp and twist during fabrication.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 2.4 4-Layer Stack Preference
- Rule: Prefer Signal/GND/PWR/Signal or Signal+GND/GND/GND/Signal+GND for EMI.
- Applies to: 4-layer boards.
- Rationale: Improves return paths and shields outer signals.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 3. Power Delivery Network (PDN)

### 3.1 Decoupling Placement
- Rule: Place decoupling capacitors as close as possible to the IC power pins; smallest value closest.
- Applies to: All ICs with local decoupling.
- Rationale: Minimizes loop inductance at high frequency.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 3.2 Via Placement for Decoupling
- Rule: Use side-mount vias and keep power/ground vias adjacent.
- Applies to: High-speed or high-current rails.
- Rationale: Reduces inductance and improves field cancellation.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 3.3 Decoupling Value Mix
- Rule: Use multi-decade values only if verified by simulation; otherwise, use repeated optimal values in parallel.
- Applies to: PDN design.
- Rationale: Avoids anti-resonance peaks.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 3.4 Ferrite Bead Usage
- Rule: Avoid beads on digital rails unless required; damp with series resistor or bulk cap when used.
- Applies to: Mixed-signal power isolation.
- Rationale: Beads can introduce LC resonance with decoupling.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 4. Switching Regulators (Buck/Boost)

### 4.1 Hot Loop Minimization
- Rule: Minimize the switch-node loop area; place input/output caps adjacent to FETs.
- Applies to: Switching regulators (buck/boost).
- Rationale: The high di/dt loop is the dominant EMI source.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 4.2 Power Ground Island
- Rule: Return input cap, output cap, and sync FET grounds to a dedicated power ground island before joining main ground.
- Applies to: Switching regulators.
- Rationale: Prevents noisy return currents from polluting analog ground.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 4.3 Feedback Trace Isolation
- Rule: Keep FB traces away from switch nodes/inductors; shield with ground when possible.
- Applies to: Switching regulators.
- Rationale: FB nodes are high impedance and noise sensitive.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 5. EMI/EMC Design Rules

### 5.1 Edge Plating and Shielding
- Rule: Use edge plating and ground stitching for RF enclosures when required.
- Applies to: RF or high-emissions boards.
- Rationale: Creates a Faraday cage and reduces edge radiation.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 5.2 Stitching Via Density
- Rule: Stitch ground planes at <=5 mm or <=lambda/10 of highest frequency.
- Applies to: EMI-sensitive designs.
- Rationale: Keeps return paths short and reduces slot antennas.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 5.3 Connector Shield Grounding
- Rule: Ground connector shells to chassis ground at entry.
- Applies to: External connectors (USB, RF, IO).
- Rationale: Provides a controlled return path and reduces ESD/EMI.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 6. Thermal Management

### 6.1 Thermal Via Placement
- Rule: Place thermal vias under exposed pads, direct-connected to internal ground.
- Applies to: Power ICs and thermal pads.
- Rationale: Improves heat spreading into inner planes.
- Source: IPC-2152 (Standard for Determining Current Carrying Capacity in Printed Board Design).

### 6.2 Thermal Via Sizing
- Rule: Use ~0.3 mm drill with 1.0-1.2 mm pitch when fab allows; adjust per fab.
- Applies to: Thermal pads.
- Rationale: Balances thermal conduction and manufacturability.
- Source: IPC-2152 (Standard for Determining Current Carrying Capacity in Printed Board Design).

### 6.3 Heatsink Grounding
- Rule: Ground metal heatsinks to PCB ground when they can radiate.
- Applies to: Metal heatsinks in open enclosures.
- Rationale: Prevents heatsinks from becoming antennas.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 7. Manufacturability (DfM) Rules

### 7.1 Fiducials
- Rule: Use 3 global fiducials and local fiducials for fine-pitch parts; typical 1 mm copper / 2 mm mask.
- Applies to: Assembly-ready boards.
- Rationale: Ensures pick-and-place alignment.
- Source: IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

### 7.2 Acid Traps
- Rule: Avoid acute trace angles (<90 degrees); use 45-degree or curves.
- Applies to: All routing.
- Rationale: Reduces etch defects and stress points.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 7.3 Solder Mask Dams
- Rule: Maintain minimum solder mask dam width per fab (e.g., >=4 mil).
- Applies to: Fine-pitch ICs and dense routing.
- Rationale: Prevents solder bridging.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 7.4 Annular Rings
- Rule: Maintain annular ring >=4-5 mil over drill size; use teardrops at pad transitions.
- Applies to: Through-hole and via pads.
- Rationale: Improves yield and reduces breakout.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 8. Specific Scenarios

### 8.1 Ground Planes (Solid vs. Split)
- Rule: Prefer solid ground planes; partition placement rather than splitting planes.
- Applies to: Mixed-signal boards.
- Rationale: Splits create return discontinuities and dipole antennas.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 8.2 Crystal Layout
- Rule: Place crystals near MCU pins; use ground shield; follow datasheet for copper keepout.
- Applies to: MCU/RTC crystals.
- Rationale: Reduces noise pickup and loading errors.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 8.3 Differential Pairs
- Rule: Match lengths to interface tolerance; set impedance and coupling per standard.
- Applies to: USB, PCIe, LVDS, Ethernet.
- Rationale: Maintains timing and signal integrity.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

## 9. Routing and Signal Integrity

### 9.1 Trace Width and Clearance
- Rule: Use minimum trace width and clearance per board class and voltage; default to fab capabilities if tighter than IPC guidance.
- Applies to: All routed nets.
- Rationale: Prevents shorts and ensures manufacturability.
- Source: IPC-2221B (Generic Standard on Printed Board Design), IPC-2222 (Sectional Design Standard for Rigid Printed Boards).

### 9.2 Controlled Impedance Routing
- Rule: When impedance-controlled, derive trace geometry from stackup and target impedance; do not reuse values across stackups.
- Applies to: High-speed serial, RF, USB, PCIe, HDMI.
- Rationale: Impedance is stackup dependent and affects signal integrity.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

### 9.3 Length Matching
- Rule: Match differential pair lengths and, when required, match critical nets within interface tolerance.
- Applies to: High-speed differential links and DDR buses.
- Rationale: Reduces skew and timing errors.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

### 9.4 Return Path Continuity
- Rule: Do not route high-speed signals over reference plane splits; if a split is unavoidable, add stitching vias at layer transitions.
- Applies to: High-speed nets and clocks.
- Rationale: Maintains continuous return current and reduces EMI.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 9.5 Via Usage
- Rule: Avoid unnecessary vias on high-speed nets; when required, keep via stub length minimal.
- Applies to: High-speed and RF nets.
- Rationale: Vias add inductance and discontinuities.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

## 10. Placement Heuristics

### 10.1 IO Connector Placement
- Rule: Place external connectors on board edges with shortest possible path to relevant circuitry.
- Applies to: USB, RF, power, and IO connectors.
- Rationale: Reduces trace length and improves EMI control.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 10.2 Decoupling Grouping
- Rule: Group decoupling capacitors by IC and place them on the same side as the IC when possible.
- Applies to: Digital ICs and regulators.
- Rationale: Minimizes loop inductance and simplifies routing.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 10.3 Sensitive Analog Placement
- Rule: Place analog front-ends away from high di/dt power stages and high-speed digital clocks.
- Applies to: Mixed-signal boards.
- Rationale: Reduces coupling and noise injection.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 10.4 Keepout Regions
- Rule: Respect keepout zones for antennas, high-voltage, and mechanical features.
- Applies to: RF, HV, and mechanically constrained designs.
- Rationale: Prevents detuning, arcing, or mechanical interference.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 11. Assembly and Test

### 11.1 Component Orientation
- Rule: Align polarized components consistently and minimize orientation variants.
- Applies to: Assembly-ready designs.
- Rationale: Improves pick-and-place reliability and reduces errors.
- Source: IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

### 11.2 Test Points
- Rule: Provide test points for power rails and critical signals with adequate clearance for probes.
- Applies to: Designs requiring validation or production test.
- Rationale: Enables reliable debug and manufacturing test.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 11.3 Solder Mask and Surface Finish
- Rule: Select solder mask and finish compatible with pitch and assembly process; ensure mask expansion per fab.
- Applies to: All assembled boards.
- Rationale: Reduces bridging and improves yield.
- Source: IPC-SM-840 (Qualification and Performance of Permanent Solder Mask), IPC-6012 (Qualification and Performance Specification for Rigid Printed Boards).

## 12. Reliability and Fabrication Details

### 12.1 Via Protection
- Rule: Use via tenting, plugging, or filling based on assembly and routing density.
- Applies to: Dense layouts and fine-pitch components.
- Rationale: Prevents solder wicking and improves yield.
- Source: IPC-4761 (Design Guide for Protection of Printed Board Via Structures).

### 12.2 Annular Ring Requirements
- Rule: Maintain annular ring minimums per board class and drill size.
- Applies to: Through-hole and via pads.
- Rationale: Improves structural integrity and reduces breakout.
- Source: IPC-2221B (Generic Standard on Printed Board Design), IPC-2222 (Sectional Design Standard for Rigid Printed Boards).

## 13. Power Integrity and Decoupling Targets

### 13.1 Target Impedance Planning
- Rule: Define a target impedance for each power rail and ensure the PDN stays below it across operating frequency.
- Applies to: Digital rails and high-current supplies.
- Rationale: Limits supply ripple and noise-induced timing errors.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 13.2 Bulk vs. High-Frequency Decoupling
- Rule: Place bulk capacitors near regulators and high-frequency capacitors at IC pins; do not substitute bulk for local high-frequency decoupling.
- Applies to: All regulated rails.
- Rationale: Different capacitance values cover different frequency ranges.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 13.3 Plane Pairing for PDN
- Rule: When possible, place power and ground planes adjacent to reduce loop inductance.
- Applies to: Multilayer boards with dedicated planes.
- Rationale: Reduces PDN impedance and EMI.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 14. RF and Antenna Layout

### 14.1 RF Keepout
- Rule: Enforce antenna keepout zones per antenna datasheet; no copper, components, or vias in keepout.
- Applies to: RF designs with onboard antennas.
- Rationale: Prevents detuning and gain loss.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

### 14.2 RF Ground Fences
- Rule: Use via fences around RF sections with spacing <= lambda/20 at the highest frequency.
- Applies to: RF front-ends and transmission lines.
- Rationale: Reduces leakage and improves isolation.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 14.3 RF Trace Geometry
- Rule: Maintain impedance-controlled trace geometry and avoid sharp bends; use arcs or mitered corners.
- Applies to: RF and microwave traces.
- Rationale: Prevents impedance discontinuities and radiation.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

## 15. HDI and Advanced Fabrication

### 15.1 Microvia Usage
- Rule: Use microvias only within approved aspect ratios and stack limits; avoid stacked vias unless required.
- Applies to: HDI designs.
- Rationale: Improves yield and reliability.
- Source: IPC-2226 (Sectional Design Standard for High Density Interconnect Printed Boards).

### 15.2 Via-in-Pad
- Rule: Via-in-pad requires filling and planarization when used under fine-pitch BGAs.
- Applies to: Fine-pitch BGA and HDI layouts.
- Rationale: Prevents solder loss and tombstoning.
- Source: IPC-4761 (Design Guide for Protection of Printed Board Via Structures), IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

### 15.3 Trace/Via Density Limits
- Rule: Respect vendor class limits for minimum trace, spacing, and via sizes; encode per fabrication class.
- Applies to: HDI and dense routing.
- Rationale: Prevents yield loss and cost overruns.
- Source: IPC-2226 (Sectional Design Standard for High Density Interconnect Printed Boards).

### 15.4 BGA Fanout Strategy
- Rule: Use Dogbone fanout for BGA pitch >= 0.5 mm; Use Via-in-Pad (VIP) for pitch < 0.5 mm.
- Applies to: BGA components.
- Rationale: Physical clearance constraints for dogbone traces fail below 0.5mm pitch.
- Source: IPC-7351B / Fab Capabilities (Common Standard).

## 16. ESD and EMC Entry Design

### 16.1 ESD Protection Placement
- Rule: Place ESD diodes at the connector entry, before any long trace run.
- Applies to: External connectors (USB, GPIO, RF).
- Rationale: Clamps fast transients before they reach sensitive circuitry.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 16.2 Chassis Grounding
- Rule: Bond connector shields to chassis ground with a short, low-inductance path; avoid long shield traces.
- Applies to: Shielded connectors and metal enclosures.
- Rationale: Provides controlled return for common-mode currents.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 16.3 EMI Filter Placement
- Rule: Place common-mode chokes and filter components at the connector side of the interface.
- Applies to: High-speed I/O (USB, HDMI, Ethernet).
- Rationale: Prevents emissions from entering or leaving the board.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 17. Mechanical and Assembly Constraints

### 17.1 Mounting Holes and Keepouts
- Rule: Reserve keepout zones around mounting holes and fasteners.
- Applies to: Mechanically constrained designs.
- Rationale: Prevents shorts and mechanical interference.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 17.2 Component Height and Courtyard
- Rule: Enforce courtyard clearance for all components; respect max height in enclosure zones.
- Applies to: Assembly-ready designs.
- Rationale: Prevents component collisions and assembly issues.
- Source: IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

### 17.3 Polarized Component Alignment
- Rule: Align polarized parts consistently and mark polarity on silkscreen.
- Applies to: Electrolytics, diodes, LEDs.
- Rationale: Reduces assembly errors and rework.
- Source: IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

## 18. DFM: Solder Paste and Stencil

### 18.1 Paste Aperture Reduction
- Rule: Reduce paste aperture for fine-pitch parts as recommended by footprint standards.
- Applies to: QFN, BGA, fine-pitch ICs.
- Rationale: Prevents bridging and excess solder.
- Source: IPC-7525 (Stencil Design Guidelines), IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

### 18.2 Large Thermal Pads
- Rule: Segment paste apertures on large exposed pads to avoid solder voids and float.
- Applies to: QFN/QFP exposed pads and power ICs.
- Rationale: Improves solder joint reliability.
- Source: IPC-7093 (Design and Assembly Process Implementation for Bottom Termination Components).

### 18.3 Paste Mask Expansion
- Rule: Define paste mask expansion per fab capability and component pitch.
- Applies to: All SMT footprints.
- Rationale: Ensures consistent paste deposition.
- Source: IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard).

## 19. Clocking and Timing-Sensitive Nets

### 19.1 Clock Source Placement
- Rule: Place clocks and oscillators adjacent to their load pins; minimize trace length and avoid vias.
- Applies to: MCU clocks, PLLs, crystal oscillators.
- Rationale: Reduces jitter and coupling.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

### 19.2 Clock Routing Isolation
- Rule: Keep clocks away from high di/dt nets and aggressive edge-rate signals.
- Applies to: High-speed digital designs.
- Rationale: Prevents crosstalk and timing errors.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

## 20. Grounding and Return Paths

### 20.1 Single-Point Analog Ground Tie
- Rule: Connect analog and digital grounds at a single point when separation is required by system architecture.
- Applies to: Mixed-signal systems with sensitive analog.
- Rationale: Controls return current paths and reduces coupling.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 20.2 Stitching at Layer Transitions
- Rule: Add ground stitching vias near signal layer transitions.
- Applies to: High-speed nets crossing layers.
- Rationale: Preserves return path continuity.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 21. Power Conversion Layout Details

### 21.1 Inductor and Switch Node Keepout
- Rule: Keep sensitive signals away from switch nodes and inductor fields.
- Applies to: Switching regulators.
- Rationale: Reduces EMI coupling into analog or RF paths.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 21.2 Sense Line Routing
- Rule: Route sense lines as Kelvin connections directly to load/rail source points.
- Applies to: Regulators with remote sense.
- Rationale: Improves regulation accuracy.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

## 22. Connectors and Edge Interfaces

### 22.1 Edge Connector Keepouts
- Rule: Enforce copper and component keepouts around card-edge connectors.
- Applies to: Edge connectors and mezzanine boards.
- Rationale: Prevents mechanical interference and shorts.
- Source: IPC-2221B (Generic Standard on Printed Board Design).

### 22.2 Differential Pair Entry
- Rule: Route differential pairs to connectors with matched lengths and symmetric geometry.
- Applies to: High-speed serial connectors.
- Rationale: Preserves impedance and reduces skew.
- Source: IPC-2141 (Controlled Impedance Circuit Boards and High-Speed Signal Propagation).

## 23. Automated Placement Rules

### 23.1 Grid Alignment (Quantization)
- Rule: Snap all component centroids to a defined grid hierarchy based on component type.
  - Primary Grid: 0.5mm (ICs, connectors).
  - Secondary Grid: 0.1mm (Fine-pitch passives like 0201/0402).
  - Rotation: 90° increments standard; 45° allowed for specific density requirements.
- Applies to: All components during legalization phase.
- Rationale: Ensures professional, manufacturable layout aesthetics and alignment.
- Source: Internal Research (Manhattan Placement Strategy).

### 23.2 Row/Column Alignment
- Rule: Group similar components (e.g., decoupling caps, pull-up resistors) into linear rows or columns.
  - Row Mode: Align Y coordinates to median; sort by X.
  - Column Mode: Align X coordinates to median; sort by Y.
- Applies to: Passive clusters within 10mm radius.
- Rationale: Improves visual organization and simplifies routing channels.
- Source: Internal Research (Manhattan Placement Strategy).

### 23.3 Overlap Resolution Priority
- Rule: When resolving overlaps, move lower-priority components first.
  - Priority Order: Locked > Large ICs > Connectors > Passives.
- Applies to: Legalization phase (Shove algorithm).
- Rationale: Preserves placement of critical/large components while shifting flexible passives.
- Source: Internal Research (Manhattan Placement Strategy).

## 24. Advanced Routing Rules

### 24.1 Differential Pair Coupling
- Rule: Route differential pairs as a coupled entity using a virtual centerline; maintain constant gap ($g$) and width ($w$).
- Applies to: Differential pairs (USB, PCIe, LVDS).
- Rationale: Maintains constant differential impedance.
- Source: Internal Research (Diff Pair Routing Strategies).

### 24.2 Phase Matching
- Rule: Compensate for skew at corners by adding length to the inner trace immediately after the bend ("structure" generation).
- Applies to: High-speed differential pairs exceeding skew tolerance.
- Rationale: Prevents common-mode conversion and timing errors.
- Source: Internal Research (Diff Pair Routing Strategies).

### 24.3 Uncoupling Penalty
- Rule: Apply a high cost penalty for any segment where the differential pair splits (uncouples) around an obstacle.
- Applies to: Differential pair routing.
- Rationale: Uncoupling disrupts impedance and increases EMI.
- Source: Internal Research (Diff Pair Routing Strategies).

### 24.4 Bus Routing (River Routing)
- Rule: Route parallel buses as a ribbon; ensure start/end pin ordering is planar (e.g., [1,2,3] -> [1,2,3]). If ordering is reversed ([1,2,3] -> [3,2,1]), a layer change/twist is required.
- Applies to: Parallel buses (DDR, SDRAM).
- Rationale: Prevents track crossing and optimizes space.
- Source: Internal Research (Diff Pair Routing Strategies).

## 25. BGA and High-Density Fanout

### 25.1 Fanout Pattern Selection
- Rule: Select fanout geometry based on ball pitch:
  - Pitch >= 0.65mm: Use Dogbone fanout (Via offset from pad).
  - Pitch <= 0.5mm: Use Via-in-Pad (VIP) with plugged/capped vias.
- Applies to: BGA, CSP, QFN packages.
- Rationale: Physical clearance limitations prevent dogbone routing at fine pitches.
- Source: Internal Research (BGA Fanout Algorithms).

### 25.2 Layer Assignment (Onion Strategy)
- Rule: Assign BGA pin rows to layers radially:
  - Outermost rows -> Top Layer.
  - Second ring -> Bottom Layer (via).
  - Third ring -> Inner Layer 1 (via).
  - Center pins -> Deepest layers.
- Applies to: High-pin-count BGAs (>100 pins).
- Rationale: Prevents blocking escape paths for inner pins.
- Source: Internal Research (BGA Fanout Algorithms).

### 25.3 Differential Pair Fanout
- Rule: Maintain symmetry in differential pair fanout; avoid standard diagonal dogbones that induce skew. Use "flag" fanout or perpendicular vias.
- Applies to: High-speed differential pairs under BGAs.
- Rationale: Asymmetric via placement creates uncompensated phase skew.
- Source: Internal Research (BGA Fanout Algorithms).

## 26. FPGA Optimization Rules

### 26.1 Pin Swapping optimization
- Rule: When reconfigurable IO is available (FPGA/MCU), swap pin assignments to minimize total ratsnest crossing number before routing.
- Applies to: FPGA banks, GPIO ports.
- Rationale: Reduces layer transitions and via count by untangling connections at the source.
- Source: Internal Research (FPGA Routing Strategies).

### 26.2 FPGA Decoupling
- Rule: Place high-frequency capacitors (0201/0402) directly on the reverse side of the PCB under the BGA, utilizing the "via forest" for low-inductance connection.
- Applies to: FPGA core voltage rails.
- Rationale: Minimizes loop inductance for high di/dt current demands.
- Source: Internal Research (FPGA Routing Strategies).

## 27. Constraint Definition Language

### 27.1 Grouping Primitives
- **Group:** Force a set of components to move together as a rigid or semi-rigid body.
- **Cluster:** Loose grouping where components attract but can flow around internal obstacles.
- **Pair:** Strict 1:1 association (e.g., Decoupling Cap + IC Pin).

### 27.2 Alignment Primitives
- **Align (Row/Column):** Force centroids to share a common X or Y axis.
- **Flow:** Order components sequentially (A -> B -> C) to minimize total wire length.
- **Stack:** Place components adjacent to each other with minimal clearance (e.g., memory chips).

### 27.3 Anchoring Primitives
- **Anchor:** Lock a component (usually a connector or main IC) to a specific board coordinate.
- **Region:** Restrict a group of components to a bounded box (e.g., "Top-Left", "Analog Zone").
- **Edge:** Constrain component to within `dist` of board outline.
