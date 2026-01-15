# FPGA & BGA Fanout Implementation Plan

**Target:** Automated escape routing for high-density BGA packages (FPGAs, MCUs, SoCs).
**Status:** Planned / Phase 3

## 1. Executive Summary
Routing dense BGAs is the primary bottleneck in modern PCB design. "Escape routing" or "Fanout" is the process of breaking signals out from the BGA pins to the main routing field. This module will automate that process using procedural geometry and flow-based algorithms.

## 2. Core Concepts

### 2.1 Fanout Patterns
*   **Dogbone:** Standard for pitch $\ge$ 0.65mm. Trace + Offset Via.
*   **Via-in-Pad (VIP):** Standard for pitch $\le$ 0.5mm. Via directly in pad center (requires manufacturing support).
*   **Channel Routing:** Routing traces between rows of pins without vias (limited depth).

### 2.2 Layer Assignment (The "Onion" Model)
*   **Ring 1 (Outermost):** Route on Top Layer.
*   **Ring 2:** Route on Bottom Layer (via).
*   **Ring 3:** Route on Inner Layer 1 (via).
*   **Ring N:** Deepest layers.

## 3. Architecture

New module: `atoplace/routing/fanout/`

```
atoplace/routing/fanout/
├── __init__.py
├── generator.py       # Main entry point
├── patterns.py        # Dogbone / VIP geometry generators
├── escape_router.py   # "Spoke" routing from via to open space
└── layer_assigner.py  # Ring analysis and layer mapping
```

### 3.1 `FanoutGenerator` Class
```python
class FanoutGenerator:
    def __init__(self, board: Board, dfm: DFMProfile):
        self.board = board
        self.dfm = dfm

    def fanout_component(self, ref: str, strategy: str = "auto"):
        """
        Generate fanout for a specific component.
        
        Args:
            ref: Component reference (e.g., "U1")
            strategy: "dogbone", "vip", or "auto" (based on pitch)
        """
        comp = self.board.components[ref]
        pitch = self._measure_pitch(comp)
        
        if strategy == "auto":
            strategy = "vip" if pitch <= 0.5 else "dogbone"
            
        if strategy == "dogbone":
            self._apply_dogbone(comp)
        elif strategy == "vip":
            self._apply_vip(comp)
```

### 3.2 `LayerAssigner` Class
Allocates BGA rings to PCB layers.

```python
class LayerAssigner:
    def assign_layers(self, comp: Component, stackup: Stackup):
        """
        Assign escape layers to pins based on ring depth.
        """
        rows, cols = self._get_grid_dimensions(comp)
        
        for pin in comp.pads:
            ring_index = self._calculate_ring_index(pin, rows, cols)
            target_layer = stackup.get_layer_for_ring(ring_index)
            pin.escape_layer = target_layer
```

## 4. Implementation Steps

### Phase 1: Procedural Patterns (Dogbones)
**Goal:** Generate geometric vias and short traces for a BGA.
1.  **Identify Quadrants:** Divide BGA into NE, NW, SE, SW.
2.  **Place Vias:** Place via in the "diagonal" gap toward the corner.
3.  **Add Trace:** Connect Pad -> Via.
4.  **DRC Check:** Verify clearance to adjacent pads.

### Phase 2: Escape Routing (Spokes)
**Goal:** Route from the fanout via to the "edge" of the BGA courtyard.
1.  **Vector Field:** Create a vector field pointing "outward" from the BGA center.
2.  **Ray Casting:** Project rays from vias along the vector field.
3.  **Collision Avoidance:** Stop ray if it hits another via/pad.
4.  **Trace Generation:** Convert valid rays to traces on the assigned layer.

### Phase 3: Pin Swapping (FPGA Only)
**Goal:** Reassign nets to pins to untangle the ratsnest.
1.  **Input:** Netlist + FPGA Bank Rules (voltage, diff pairs).
2.  **Optimization:** Bipartite matching or Min-Cost Max-Flow.
3.  **Output:** Updated Netlist (requires back-annotation to schematic/FPGA tool).

## 5. Integration with Main Router
*   **Pre-Processing:** Fanout runs *before* the main A* router.
*   **Obstacles:** Fanout traces/vias become obstacles.
*   **Start Points:** The main router starts from the *end* of the escape trace (in clear space), not the buried BGA pin.

## 6. References
*   **MonoSAT:** "Scalable, High-Quality, SAT-Based Multi-Layer Escape Routing" (UBC).
*   **kicad-bga-tools:** Open-source Python scripts for dogbone generation.
*   **"Ordered Escape Routing"**: Standard algorithm for monotonic escape.
