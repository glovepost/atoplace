# Issue Tracker

## Security Issues

- [x] ~~**Launcher Log File Permissions**~~ **FIXED**
  - **Location:** `atoplace/mcp/launcher.py:35-43`
  - **Severity:** LOW
  - **Issue:** Launcher defaults to `/tmp/atoplace.log` without restricting permissions.
  - **Impact:** Logs may be world-readable on multi-user systems, exposing board paths or debug output.
  - **Fix:** Use a user-specific log directory or `chmod` the log file to `0o600` after creation.

- [x] ~~**Socket File Permissions Vulnerability**~~ **FIXED**
  - **Location:** `atoplace/mcp/ipc.py:258`
  - **Severity:** HIGH
  - **Issue:** Socket created with world-readable/writable permissions (`0o666`)
  - **Impact:** Any user on the system can connect to the socket and execute KiCad commands; potential privilege escalation or information disclosure
  - **Fix:** Changed to `os.chmod(self.socket_path, 0o600)` to restrict access to owner only

## Resource Leaks

- [x] ~~**Bridge Stdout Pipe Not Drained**~~ **FIXED**
  - **Location:** `atoplace/mcp/launcher.py:175-182`
  - **Severity:** MEDIUM
  - **Issue:** Bridge subprocess is started with `stdout=PIPE` but no reader consumes the stream.
  - **Impact:** If the bridge logs enough output, the pipe buffer can fill and block the bridge process.
  - **Fix:** Stream bridge output to a log file or spawn a background reader thread.

- [x] ~~**Unclosed File Handle in CLI**~~ **FIXED**
  - **Location:** `atoplace/cli.py:118`
  - **Severity:** CRITICAL
  - **Issue:** `_LOG_FILE_HANDLE = log_path.open("a", encoding="utf-8")` opens file but never closes it
  - **Impact:** File descriptor leak causing resource exhaustion if CLI runs multiple times
  - **Fix:** Added atexit handler `_cleanup_log_file()` to ensure file closure on exit

- [x] ~~**Unclosed Subprocess Pipes in RPC Client**~~ **FIXED**
  - **Location:** `atoplace/rpc/client.py:20-27`
  - **Severity:** CRITICAL
  - **Issue:** `subprocess.Popen` pipes not explicitly closed in cleanup
  - **Impact:** File descriptors may leak on abnormal termination
  - **Fix:** Enhanced `close()` method to explicitly close stdin/stdout/stderr pipes and wait for process termination

## Technical Debt & Code Quality

- [ ] **Global Code Formatting**
  - **Issue:** ~1,200 linting warnings reported by `ruff`.
  - **Details:** Mostly whitespace (`W293`), unsorted imports (`I001`), and f-string placeholders (`F541`).
  - **Action:** Run `ruff check --fix` and `black .` across the entire codebase.

- [x] ~~**Bare Exception Handlers**~~ **FIXED**
  - **Locations:** Multiple files (see details below)
  - **Severity:** HIGH
  - **Issue:** Broad `except:` blocks with `pass` mask real errors and make debugging difficult
  - **Files affected:**
    - `atoplace/board/kicad_adapter.py` (lines 463, 499-500, 510, 517, 525, 687, 1028, 1062, 1071, 1077, 1083, 1093, 1156, 1277)
    - `atoplace/visualization/color_manager.py:55` (file no longer exists)
  - **Fix:** Replaced bare `except:` with specific exception types (`AttributeError`, `RuntimeError`, `IndexError`) and added logging where appropriate

- [x] ~~**Overly Broad Exception Handling**~~ **FIXED**
  - **Location:** `atoplace/board/kicad_adapter.py:47`
  - **Issue:** `except (ImportError, RuntimeError, AttributeError, Exception)` - catching `Exception` makes specific types redundant
  - **Fix:** Replaced `Exception` with `TypeError` to handle method signature changes across wx versions

- [ ] **CI Pipeline for KiCad Tests**
  - **Issue:** Tests requiring `pcbnew` cannot run in standard python environments.
  - **Action:** Create a Docker container or CI workflow that includes the KiCad runtime/libraries to enable automated testing of `atoplace.board` adapters.

- [x] ~~**Unchecked List Indexing**~~ **RESOLVED**
  - **Location:** `atoplace/mcp/drc.py:726`
  - **Severity:** MEDIUM
  - **Issue:** `polygon_areas[i + 1][1]` may cause IndexError
  - **Resolution:** Code was refactored in earlier changes; the referenced line no longer exists (file has 649 lines)

- [x] ~~**Temporary File Cleanup**~~ **FIXED**
  - **Location:** `atoplace/mcp/drc.py:215-218, 267`
  - **Severity:** MEDIUM
  - **Issue:** NamedTemporaryFile not explicitly configured for cleanup
  - **Fix:** Added explicit cleanup in finally block with proper OSError handling instead of bare except

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

## Functional Bugs

- [x] ~~**IPC Batch Update Cannot Unlock Locked Components**~~ **FIXED**
  - **Location:** `atoplace/mcp/bridge.py:265-292`
  - **Severity:** HIGH
  - **Issue:** `handle_update_components` rejects any locked component, even when the update explicitly sets `locked=False`.
  - **Impact:** IPC clients cannot unlock components via batch updates; MCP lock/unlock calls in IPC mode silently fail.
  - **Fix:** Allow updates that explicitly unlock (`locked` is `False`) or mirror `handle_update_component` behavior.

- [x] ~~**IPC Session Drops Failed Updates**~~ **FIXED**
  - **Location:** `atoplace/mcp/ipc_session.py:194-215`
  - **Severity:** MEDIUM
  - **Issue:** `_sync_to_bridge` ignores per-component failures from `update_components` and clears `_dirty_refs` regardless.
  - **Impact:** Local board state can diverge from the bridge (e.g., locked/not-found refs), losing pending changes.
  - **Fix:** Inspect response results and keep failed refs dirty or refresh from the bridge on partial failure.

- [x] ~~**Fanout Obstacles Block Same-Net Routing**~~ **FIXED**
  - **Location:** `atoplace/routing/manager.py:193-210`
  - **Severity:** HIGH
  - **Issue:** Fanout traces/vias are added with `net_id=None`, so they block all nets including their own.
  - **Impact:** Nets with fanout can become unroutable because the router treats their own escape geometry as obstacles.
  - **Fix:** Assign `net_id` for fanout traces/vias (e.g., hash of `net_name`) so same-net filtering works.

- [x] ~~**KiPy Lock State Not Synced**~~ **FIXED**
  - **Location:** `atoplace/mcp/kipy_session.py:508-536`
  - **Severity:** MEDIUM
  - **Issue:** `_sync_to_kicad` omits the `locked` field from batch updates.
  - **Impact:** Lock/unlock operations update the local model but never reach KiCad; components stay movable in KiCad UI.
  - **Fix:** Include `locked` in batch updates or apply lock changes via `update_component`.

- [x] ~~**find_components Filter Case Handling**~~ **FIXED**
  - **Location:** `atoplace/api/inspection.py:111-126`
  - **Severity:** LOW
  - **Issue:** `filter_by.lower()` is validated but the original `filter_by` is used, so `Ref`/`FOOTPRINT` yield no matches.
  - **Impact:** Clients using mixed-case filter names get empty search results.
  - **Fix:** Normalize `filter_by = filter_by.lower()` before branching.

- [x] ~~**distribute_evenly Ignores Auto Axis With Explicit Start**~~ **FIXED**
  - **Location:** `atoplace/api/actions.py:192-208`
  - **Severity:** LOW
  - **Issue:** Auto-axis detection only runs when `start_ref` is not provided; with `start_ref` it defaults to `y`.
  - **Impact:** Components may distribute along the wrong axis in common calls.
  - **Fix:** Resolve `axis="auto"` regardless of anchor selection.

- [x] ~~**Lock File Timestamps Lost on Load**~~ **FIXED**
  - **Location:** `atoplace/board/lock_file.py:238-258`
  - **Severity:** LOW
  - **Issue:** PyYAML `safe_load` converts ISO timestamps to `datetime`, but `from_dict` expects strings and drops them.
  - **Impact:** `created`/`modified` fields reset to “now” after load/save cycles.
  - **Fix:** Accept `datetime` objects or stringify before parsing.

- [x] ~~**RPC DRC endpoint crashes**~~ **FIXED**
  - **Location:** `atoplace/rpc/worker.py:212-234`
  - **Severity:** CRITICAL
  - **Issue:** Calls nonexistent `DRCChecker.check_all()` and reads `violation_type/refs` attributes that don't exist on `DRCViolation`, so `run_drc` always raises.
  - **Fix:** Changed to use `DRCChecker.run_checks()` and serialize existing fields (`rule`, `items`, `location`, `severity`, `message`)

- [x] ~~**RPC arrange_pattern TypeError**~~ **FIXED**
  - **Location:** `atoplace/rpc/worker.py:185-188`
  - **Severity:** HIGH
  - **Issue:** Passes `center_x`/`center_y` as extra positional args to `LayoutActions.arrange_pattern`, causing a TypeError before any work is done.
  - **Fix:** Convert center_x, center_y to tuple and pass as `center` parameter

- [x] ~~**RPC validate_placement always fails**~~ **FIXED**
  - **Location:** `atoplace/rpc/worker.py:236-247`
  - **Severity:** HIGH
  - **Issue:** Instantiates `ConfidenceScorer(self.board)` (interprets board as dfm profile) and returns non-existent fields (`category_scores`, `recommendations`), so the call raises AttributeError.
  - **Fix:** Create `ConfidenceScorer(dfm_profile=dfm)` and call `assess(board)`, serialize actual report fields (placement_score, routing_score, dfm_score, electrical_score, flags)

- [x] ~~**Locked components moved by distribute_evenly**~~ **FIXED**
  - **Location:** `atoplace/api/actions.py:171-235`
  - **Severity:** MEDIUM
  - **Issue:** Only skips locked anchors; other locked refs are repositioned, violating user lock semantics.
  - **Fix:** Rewrote function to filter out all locked components before calculating pitch; reports skipped components in return value

- [x] ~~**Overlap check misses rotated/pad extents**~~ **FIXED**
  - **Location:** `atoplace/api/inspection.py:29-75`
  - **Severity:** LOW
  - **Issue:** Uses unrotated width/height AABB without pads, so overlaps on rotated/edge-mounted parts are missed, diverging from DRC/placement checks.
  - **Fix:** Updated to use `get_bounding_box_with_pads()` for proper rotation handling and pad extents; added `include_pads` parameter for flexibility

- [x] ~~**Visualizer overlaps use stale board positions**~~ **FIXED**
  - **Location:** `atoplace/placement/force_directed.py:315-344`
  - **Severity:** MEDIUM
  - **Issue:** `_capture_viz_frame` calls `board.find_overlaps()` (board still at initial positions) instead of the simulated `PlacementState`, so overlap highlights/counts in captured frames are incorrect.
  - **Fix:** Added `_compute_overlaps_from_state()` helper that computes overlaps from `state.positions` and `state.rotations`

- [x] ~~**Pad coordinates double-offset in placement viz**~~ **FIXED**
  - **Location:** `atoplace/placement/force_directed.py:292-309`
  - **Severity:** MEDIUM
  - **Issue:** Pads are offset by `pad.x - comp.origin_offset_x` even though `pad.x` is already centroid-relative, shifting pads in visualization frames.
  - **Fix:** Removed incorrect origin_offset subtraction; pad.x/y are already centroid-relative per abstraction.py

## Code Maintainability

- [ ] **Brittle KiCad API Version Handling**
  - **Location:** `atoplace/board/kicad_adapter.py:1059-1063`
  - **Severity:** LOW
  - **Issue:** Multiple nested try/except blocks to handle API version differences; hard to maintain
  - **Impact:** May break with future KiCad versions
  - **Action:** Create wrapper functions that abstract version differences

- [x] ~~**Path Validation**~~ **FIXED**
  - **Location:** `atoplace/patterns.py:45`
  - **Severity:** LOW
  - **Issue:** File operations without symlink checking
  - **Impact:** Potential symlink attack if attacker controls config path
  - **Fix:** Added symlink check with `is_symlink()` that raises ValueError if config path is a symlink

## Documentation

- [ ] **Type Hinting**
  - **Action:** Improve type coverage in `atoplace/board/abstraction.py` to prevent regression in the core data model.

## Positive Security Findings

✓ No SQL injection vulnerabilities (no SQL usage)
✓ No use of pickle or eval() (avoided dangerous serialization)
✓ No hardcoded secrets/credentials
✓ No shell injection risks (subprocess calls use proper argument lists, not shell=True)
✓ Good context manager usage for most file I/O
✓ Proper logging throughout codebase
✓ Threading safety with locks in IPC/RPC clients

---

## Summary by Severity

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 3 | ✅ All Fixed |
| HIGH | 6 | ✅ All Fixed |
| MEDIUM | 8 | ✅ All Fixed |
| LOW | 7 | 6 Fixed, 1 Remaining |
| UNSPECIFIED | 7 | 1 Fixed, 6 Remaining |

**Total Issues:** 31 tracked issues
**Fixed:** 24 issues (3 CRITICAL, 6 HIGH, 8 MEDIUM, 6 LOW, 1 UNSPECIFIED)
**Remaining:** 7 issues (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW, 6 UNSPECIFIED)
