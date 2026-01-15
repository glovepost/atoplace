# Precision Visual Grounding for LLM Layout Control

**Focus:** Enabling LLMs to perform high-precision (0.1mm) layout tasks despite "Spatial Illiteracy."

## 1. The Challenge: Token vs. Precision
LLMs are bad at floating-point arithmetic and cannot "see" a 50,000-line KiCad board file. Standard RAG (Retrieval Augmented Generation) loses spatial context.
*   **Problem:** "Move C1 to the right of U1" is easy. "Move C1 to exactly 0.5mm clearance from U1 pin 4" is hard for an LLM to calculate blindly.
*   **Solution:** **Dual-Mode Grounding** + **The Microscope API**.

## 2. Dual-Mode Grounding Strategy

We provide two distinct "views" to the LLM depending on the task phase.

### 2.1 Macro Mode: The Grid Compass (Planning Phase)
*   **Goal:** General arrangement, grouping, flow.
*   **Representation:** Semantic 3x3 Grid + Relative adjacency graph.
*   **Token Cost:** Low (< 1k tokens).
*   **Content:**
    > "U1 is in Center-Left zone. C1-C4 are unplaced. Connector J1 is Bottom-Edge."

### 2.2 Micro Mode: The Precision Microscope (Execution Phase)
*   **Goal:** DRC-compliant placement, fanout, exact alignment.
*   **Representation:** **Local JSON Viewport** centered on a target.
*   **Token Cost:** Medium (1k-2k tokens), but highly focused.
*   **Mechanism:**
    1.  LLM requests: `get_visual_context(targets=["U1", "C1"], padding=5.0)`
    2.  System calculates a bounding box around targets + padding.
    3.  System returns *precise* geometry for objects *only within that box*.

## 3. The Microscope API Specification

The core enabler is a python function exposed to the LLM (via MCP).

**Function:** `inspect_region(center_x, center_y, width, height)` or `inspect_components([refs])`

**Output (JSON):**
```json
{
  "viewport": {
    "center": [105.0, 50.0],
    "size": [10.0, 10.0],
    "units": "mm"
  },
  "grid_aligned": true, // Whether viewport aligns with placement grid
  "objects": [
    {
      "ref": "U1",
      "type": "IC",
      "layer": "Top",
      "location": [105.0, 50.0], // Exact centroid
      "rotation": 0.0,
      "bbox": {
        "min": [102.5, 47.5], // Exact boundaries
        "max": [107.5, 52.5]
      },
      "pads": [ // Key pads only (corners, pin 1)
        {"num": "1", "pos": [103.0, 48.0], "net": "GND"}
      ]
    },
    {
      "ref": "C1",
      "location": [109.0, 50.0],
      "bbox": { "min": [108.5, 49.5], "max": [109.5, 50.5] }
    }
  ],
  "gaps": [
    {
      "between": ["U1", "C1"],
      "distance": 1.0, // Exact clearance
      "vector": [1.0, 0.0] // Direction vector
    }
  ]
}
```

## 4. The "Code-as-Action" Loop

The LLM does NOT calculate the new coordinate. It calculates the **delta** or calls a **solver**.

**Scenario:** "Place C1 0.5mm right of U1."

1.  **Step 1 (Inspect):** LLM calls `inspect_components(["U1", "C1"])`.
2.  **Step 2 (Analyze):**
    *   See U1 bbox.max_x = 107.5.
    *   See C1 width = 1.0 (bbox 108.5 - 109.5).
    *   Target X = 107.5 (U1 edge) + 0.5 (gap) + 0.5 (C1 half-width) = 108.5.
3.  **Step 3 (Act):**
    *   *Preferred (Robust):* Call `place_relative(ref="C1", target="U1", side="right", clearance=0.5)`. Let Python do the math.
    *   *Fallback (Direct):* Call `place_component("C1", 108.5, 50.0)`.

## 5. Visual Feedback (Multimodal)

For models like GPT-4o or Claude 3.5 Sonnet, we can supplement the JSON with an image.

**Rasterizer:**
*   Generate a small (512x512) PNG of the *Microscope Viewport*.
*   **Style:**
    *   High contrast (Black background, White components).
    *   **Annotated Dimensions:** Draw arrow lines showing current gaps/distances.
    *   **Grid Overlay:** 1mm grid lines.
*   **Prompt:** "See image. The capacitor C1 is too far. Move it closer."

## 6. Implementation Plan

1.  **Refactor Visualizer:** Create `atoplace.placement.microscope`.
    *   Method `get_json_view(board, bounds)`: Returns the precision JSON.
    *   Method `render_png_view(board, bounds)`: Returns base64 PNG using Cairo/Pillow (lightweight) or Matplotlib.
2.  **Solver Actions:** Ensure `atoplace.api` has robust relative placement functions:
    *   `align(refs, axis='x')`
    *   `distribute(refs, spacing=0.5)`
    *   `stack(refs, direction='down')`
3.  **Integration:** Hook this into the MCP Server `inspect` tool.
