# Issue Tracker

## Technical Debt & Code Quality

- [ ] **Global Code Formatting**
  - **Issue:** ~1,200 linting warnings reported by `ruff`.
  - **Details:** Mostly whitespace (`W293`), unsorted imports (`I001`), and f-string placeholders (`F541`).
  - **Action:** Run `ruff check --fix` and `black .` across the entire codebase.

- [ ] **CI Pipeline for KiCad Tests**
  - **Issue:** Tests requiring `pcbnew` cannot run in standard python environments.
  - **Action:** Create a Docker container or CI workflow that includes the KiCad runtime/libraries to enable automated testing of `atoplace.board` adapters.

## Feature Implementation Gaps

- [ ] **Differential Pair Routing Integration**
  - **Location:** `atoplace/routing/manager.py`
  - **Issue:** `TODO: Call dp_router.route_pair()` indicates the logic exists in `diff_pairs.py` but is not hooked into the main routing pipeline.
  - **Action:** Wire up the differential pair router in the `RoutingManager.run()` method.

- [ ] **Routing Validation**
  - **Location:** `atoplace/validation/confidence.py`
  - **Issue:** `TODO: Implement routing checks`
  - **Action:** Add confidence scoring metrics for routed traces (e.g., length matching, clearance violations, unrouted nets).

- [ ] **KiCad Adapter Layer Stack**
  - **Location:** `atoplace/board/kicad_adapter.py`
  - **Issue:** `TODO: Set up layer stack`
  - **Action:** Implement proper layer stack configuration when creating new boards from scratch.

## Documentation

- [ ] **Type Hinting**
  - **Action:** Improve type coverage in `atoplace/board/abstraction.py` to prevent regression in the core data model.