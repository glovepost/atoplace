# Atopile Integration Specification

## 1. Overview
Atopile integration allows AtoPlace to understand the logical structure of a design defined in `.ato` files. This includes:
*   Module grouping (Power, MCU, etc.)
*   Constraint inference
*   Persistence of placement data without modifying source code.

## 2. The `ato-lock.yaml` Strategy
Instead of parsing `.ato` directly (which is complex), we rely on the build artifacts.
*   **Input:** `ato-lock.yaml` provides component MPNs, values, and addresses.
*   **Output:** `atoplace.lock` stores physical layout data.

## 3. Implementation Details
(See `atoplace/board/atopile_adapter.py` for current code)