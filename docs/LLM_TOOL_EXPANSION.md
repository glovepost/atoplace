# LLM Tool Expansion Strategy

**Objective:** Expand the "hands" and "eyes" of the LLM beyond basic relative placement. To truly manipulate a PCB, the agent needs tools for discovery, topology navigation, advanced pattern generation, and routing control.

---

## 1. Discovery & Navigation (The "Where" & "What")

The LLM currently cannot easily find "all 10k resistors" or "unplaced components" without dumping the entire netlist (context heavy).

### Proposed Tools:

*   **`find_components(query: str, filter_by: str)`**
    *   *Usage:* `find_components("10k", filter_by="value")`, `find_components("LED", filter_by="ref")`
    *   *Returns:* List of `{ref, value, footprint, location}`.
    *   *Why:* Essential for "Group all 10k resistors" commands.

*   **`get_unplaced_components()`**
    *   *Usage:* `get_unplaced_components()`
    *   *Returns:* List of refs outside the board outline.
    *   *Why:* The starting point for any placement session.

*   **`get_board_bounds()`**
    *   *Usage:* `get_board_bounds()`
    *   *Returns:* `{min_x, min_y, max_x, max_y, width, height}`.
    *   *Why:* LLM needs to know the canvas size to avoid placing things at `(1000, 1000)`.

---

## 2. Logical Topology (The "Why")

Placement decisions are driven by connectivity. The LLM needs to ask "What is connected to U1?" to perform decoupling or clustering.

### Proposed Tools:

*   **`get_connected_components(ref: str)`**
    *   *Usage:* `get_connected_components("U1")`
    *   *Returns:* `{ "GND": ["C1", "C2"], "VCC": ["C3"], "SPI_MOSI": ["J1"] }`
    *   *Why:* Critical for "Place decoupling capacitors near U1".

*   **`get_critical_nets()`**
    *   *Usage:* `get_critical_nets()`
    *   *Returns:* List of nets tagged as High Speed, RF, or Diff Pairs (from AtoPlace analysis).
    *   *Why:* Prioritizing routing order.

---

## 3. Advanced Placement Patterns (The "How")

Moving one component at a time is tedious. We need procedural pattern generators.

### Proposed Tools:

*   **`arrange_pattern(refs: List[str], pattern: str, ...)`**
    *   *Usage:*
        *   `arrange_pattern(["D1".."D4"], "grid", cols=2, spacing=2.0)`
        *   `arrange_pattern(["R1".."R8"], "circular", radius=5.0)`
    *   *Why:* Creating LED arrays, connector breakouts, or sensor rings.

*   **`cluster_around(anchor_ref: str, target_refs: List[str], side: str)`**
    *   *Usage:* `cluster_around("U1", ["C1", "C2", "C3"], "left")`
    *   *Logic:* Automatically arranges targets in a tight cloud/row on the specified side.
    *   *Why:* Fast "good enough" placement for passives.

*   **`swap_positions(ref1: str, ref2: str)`**
    *   *Usage:* `swap_positions("R1", "R2")`
    *   *Why:* Optimizing crossing nets (untangling the ratsnest).

---

## 4. Routing Control (The "Connect")

The LLM should not draw lines segment-by-segment. It should invoke the autorouter strategies we built.

### Proposed Tools:

*   **`route_net(net_name: str, strategy: str)`**
    *   *Usage:* `route_net("USB_D+", strategy="diff_pair")`
    *   *Action:* Invokes internal A* router.
    *   *Returns:* Success/Fail, via count, length.

*   **`ripup_net(net_name: str)`**
    *   *Usage:* `ripup_net("GND")`
    *   *Why:* Fixing bad routes or clearing space for placement changes.

*   **`fanout_component(ref: str)`**
    *   *Usage:* `fanout_component("U1")`
    *   *Action:* Triggers the BGA/QFN fanout generator (Phase 2B).
    *   *Why:* Prepares dense components for routing.

---

## 5. Validation & Safety (The "Check")

The ReAct loop needs feedback to correct mistakes.

### Proposed Tools:

*   **`check_overlaps(refs: Optional[List[str]])`**
    *   *Usage:* `check_overlaps()` (all) or `check_overlaps(["U1", "C1"])`
    *   *Returns:* List of colliding pairs and overlap depth.
    *   *Why:* Immediate feedback after a `move` command.

*   **`validate_placement()`**
    *   *Usage:* `validate_placement()`
    *   *Returns:* Full DFM report (clearance violations, off-board components).

---

## 6. Implementation Priority

1.  **Topology (`get_connected_components`)**: Essential for smart grouping.
2.  **Navigation (`find_components`, `get_board_bounds`)**: Basic usability.
3.  **Safety (`check_overlaps`)**: Required for the ReAct loop.
4.  **Routing (`route_net`)**: The next major capability unlock.
5.  **Patterns (`arrange_pattern`)**: productivity booster.
