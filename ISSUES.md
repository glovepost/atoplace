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

- [ ] **RPC Client Stderr Pipe Not Drained**
  - **Location:** `atoplace/rpc/client.py:13-60`
  - **Severity:** MEDIUM
  - **Issue:** Worker process is spawned with `stderr=PIPE`, but stderr is only read when stdout returns empty.
  - **Impact:** If the KiCad worker emits enough stderr output (warnings, tracebacks), the pipe can fill and deadlock the worker, causing CLI/MCP calls to hang.
  - **Fix:** Drain stderr asynchronously (thread/task), redirect to a log file, or set `stderr=subprocess.DEVNULL` if not needed.

- [ ] **RPC Client Read Can Hang Indefinitely**
  - **Location:** `atoplace/rpc/client.py:31-55`
  - **Severity:** MEDIUM
  - **Issue:** `self.process.stdout.readline()` has no timeout or watchdog.
  - **Impact:** If the worker stalls or fails to flush a response, client calls block forever, freezing CLI/MCP usage.
  - **Fix:** Add a timeout/read watchdog (e.g., `select`/thread + timer) and surface a clear error on timeout.

## Technical Debt & Code Quality

- [x] ~~**Global Code Formatting**~~ **FIXED**
  - **Issue:** ~1,200 linting warnings reported by `ruff`.
  - **Details:** Mostly whitespace (`W293`), unsorted imports (`I001`), and f-string placeholders (`F541`).
  - **Fix:** Ran `ruff check --fix` and `black .` across the entire codebase. Fixed critical syntax error in pinswapper.py (unclosed docstring). Added TYPE_CHECKING imports to fix F821 undefined name errors. Updated pyproject.toml to ignore B008 (typer.Option false positive) and E402 (intentional conditional imports). Reduced from ~1,200 to ~53 warnings (96% reduction).

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

- [ ] **CI Pipeline for KiCad Tests** (DEFERRED - Infrastructure)
  - **Issue:** Tests requiring `pcbnew` cannot run in standard python environments.
  - **Status:** Deferred pending DevOps resources. Unit tests in `tests/unit/` can run without KiCad.
  - **Requirements:**
    1. Docker image with KiCad 8+ and Python bindings (base image: `kicad/kicad:*` or custom build)
    2. GitHub Actions workflow that uses the Docker image
    3. Mount test fixtures and run pytest for integration tests
  - **Workaround:** Use `ATOPLACE_BACKEND=direct` for file-based tests without live KiCad process.
  - **References:** KiCad Docker images exist in community (kicad-python-docker) but may need customization.

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

- [x] ~~**Differential Pair Routing Integration**~~ **FIXED**
  - **Location:** `atoplace/routing/manager.py`
  - **Issue:** `TODO: Call dp_router.route_pair()` indicates the logic exists in `diff_pairs.py` but is not hooked into the main routing pipeline.
  - **Fix:** Integrated diff pair routing in `_run_diff_pair_phase()`. Now calls `dp_router.route_pair()` with proper start/end pad tuples, handles success/failure with appropriate logging, adds routed traces/vias to board, and marks nets as routed. Falls back to general router on failure.

- [x] ~~**Routing Validation**~~ **FIXED**
  - **Location:** `atoplace/validation/confidence.py`
  - **Issue:** `TODO: Implement routing checks`
  - **Fix:** Implemented `_check_routing()` with validation for: differential pair configuration (missing/invalid pairs), high-fanout net warnings, trace width vs DFM minimums, clearance vs DFM minimums, single-connection net detection, and routing density estimation based on connections/cm².

- [x] ~~**KiCad Adapter Layer Stack**~~ **FIXED**
  - **Location:** `atoplace/board/kicad_adapter.py`
  - **Issue:** `TODO: Set up layer stack`
  - **Fix:** Implemented `_setup_new_board()` helper that configures: copper layer count (with even-number enforcement), default trace width, default clearance, default via drill/diameter, and board outline. Includes try/except fallbacks for different KiCad API versions.

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

- [ ] **RoutingManager API Mismatch With CLI/MCP**
  - **Locations:** `atoplace/cli.py:2289`, `atoplace/mcp/server.py:1829`, `atoplace/routing/manager.py`
  - **Severity:** CRITICAL
  - **Issue:** CLI/MCP call `RoutingManager.route_all()`, `add_diff_pair()`, `set_critical_nets()`, and `set_progress_callback()` but the current `RoutingManager` implementation does not define these methods and returns a different result shape.
  - **Impact:** Routing commands crash with `AttributeError` or unexpected result fields, blocking routing in CLI and MCP.
  - **Fix:** Align the routing manager API with callers (restore methods/result structure) or update CLI/MCP to use the current `RoutingManager.run()` and result model.

- [ ] **MCP Routing Uses Wrong Constructor Signature**
  - **Location:** `atoplace/mcp/server.py:1819-1822`
  - **Severity:** HIGH
  - **Issue:** `RoutingManager(session.board, config=config)` passes a `RoutingManagerConfig` where a `DFMProfile` is required.
  - **Impact:** Routing attempts will fail at runtime when the manager accesses `dfm_profile` attributes (e.g., `min_trace_width`, `min_spacing`).
  - **Fix:** Pass a real `DFMProfile` (e.g., from `get_profile()` or session config) and pass `config=` as a keyword argument.

- [ ] **RoutingManager State Not Reset Between Runs**
  - **Location:** `atoplace/routing/manager.py:105-174`
  - **Severity:** LOW
  - **Issue:** `run()` appends to `nets_to_route`, `routed_nets`, and `results` without clearing previous state.
  - **Impact:** Reusing a `RoutingManager` instance for multiple runs yields stale or inflated stats and can skip routing nets.
  - **Fix:** Clear `nets_to_route`, `routed_nets`, and `results` at the start of `run()` or make `run()` construct a fresh result object.

- [ ] **Microscope Gap Calculations Ignore Pad Extents**
  - **Location:** `atoplace/mcp/context/micro.py:119-178`
  - **Severity:** LOW
  - **Issue:** Gap analysis uses `Component.get_bounding_box()` (body-only), omitting pad protrusions.
  - **Impact:** Reported gaps can be overly optimistic for connectors or edge-mounted footprints, leading to clearance violations in downstream placement decisions.
  - **Fix:** Use `get_bounding_box_with_pads()` or make pad inclusion configurable.

## Code Maintainability

- [x] ~~**Brittle KiCad API Version Handling**~~ **FIXED**
  - **Location:** `atoplace/board/kicad_adapter.py`
  - **Severity:** LOW
  - **Issue:** Multiple nested try/except blocks to handle API version differences; hard to maintain
  - **Fix:** Added KiCad API Compatibility Layer with version detection (`KICAD_VERSION`, `KICAD_MAJOR`) and wrapper functions: `_kicad_get_text_angle_degrees()`, `_kicad_get_reference_field()`, `_kicad_get_value_field()`, `_kicad_set_layer_count()`, `_kicad_set_track_width()`, `_kicad_set_clearance()`, `_kicad_set_via_drill()`, `_kicad_set_via_size()`. Refactored `_setup_new_board()` and `_extract_ref_des_text()` to use these wrappers.

- [x] ~~**Path Validation**~~ **FIXED**
  - **Location:** `atoplace/patterns.py:45`
  - **Severity:** LOW
  - **Issue:** File operations without symlink checking
  - **Impact:** Potential symlink attack if attacker controls config path
  - **Fix:** Added symlink check with `is_symlink()` that raises ValueError if config path is a symlink

## Documentation

- [x] ~~**Type Hinting**~~ **FIXED**
  - **Action:** Improve type coverage in `atoplace/board/abstraction.py` to prevent regression in the core data model.
  - **Fix:** Added missing `-> None` return types to void methods (`add_component`, `add_net`, `add_connection`, `move_component`). Added `Any` and `Union` to typing imports. Updated `get_stats` to return `Dict[str, Union[int, float]]` instead of bare `Dict`.

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
| MEDIUM | 10 | 8 Fixed, 2 Remaining |
| LOW | 8 | ✅ All Fixed |
| UNSPECIFIED | 10 | 7 Fixed, 3 Remaining (1 Deferred) |

**Total Issues:** 37 tracked issues
**Fixed:** 30 issues (3 CRITICAL, 6 HIGH, 8 MEDIUM, 8 LOW, 5 UNSPECIFIED)
**Remaining:** 7 issues (0 CRITICAL, 0 HIGH, 2 MEDIUM, 0 LOW, 5 UNSPECIFIED - includes 1 deferred infrastructure task)
