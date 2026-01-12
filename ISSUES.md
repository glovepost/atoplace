# Project Issues Log

This file tracks code review findings and risks discovered during development.

## 2026-01-12 - REVIEW FINDINGS

### Medium - RESOLVED
- ~~**CLI Atopile Grouping**: `cmd_place` builds `GroupingConstraint` with `component_refs` and `strength` args that do not exist on `GroupingConstraint` (expects `components` and `max_spread`). This raises `TypeError` when `--use-ato-modules` is set. Files: `atoplace/cli.py`, `atoplace/placement/constraints.py`.~~ **FIXED**: Changed `component_refs` to `components` and `strength` to `max_spread=15.0` in CLI.
- ~~**Placement Physics**: `_component_sizes` is computed once at init and never updated as rotations change, so boundary/repulsion checks can use stale AABB sizes after rotation constraints are applied. File: `atoplace/placement/force_directed.py`.~~ **FIXED**: Added `_update_component_sizes()` method that recalculates AABB dimensions based on current state rotations. Called from `_apply_rotation_constraints()` when any rotation changes.
- ~~**Placement Locks**: `lock_placed` prevents position updates but rotation constraints still rotate locked components, so "locked" parts can still move via rotation. File: `atoplace/placement/force_directed.py`.~~ **FIXED**: Added `_is_component_locked()` check in `_apply_rotation_constraints()` to skip locked components.

## 2026-01-12 - FIX VERIFICATION

### High - OPEN

(none)

### High - RESOLVED (2026-01-12 Session)

- ~~**Atopile Module Mapping**: `instance_to_ref_map` expects `designator` entries at the root of `ato-lock.yaml`, but lock data stores components under `components` without designators.~~ **FIXED**: `instance_to_ref_map` now handles multiple lock file formats: (1) root-level entries with `designator` field, (2) entries under `components` key, (3) entries with `address` field. Also adds self-mapping and case-insensitive lookup for KiCad refs.

- ~~**Legalizer Implementation**: The legalizer implementation is incomplete and not integrated into the main workflow.~~ **FIXED**: Legalizer is now integrated in `cmd_place` (cli.py lines 191-205). Added `--skip-legalization` CLI flag. Grid size configurable via `--grid` option.

- ~~**CLI Atopile Grouping**: `cmd_place` calls `GroupingConstraint(component_refs=..., strength=...)`, which does not match signature.~~ **FIXED**: Changed to `GroupingConstraint(components=comp_refs, max_spread=15.0)` in cli.py lines 153-158.

### Medium - OPEN

- **Legalizer R-Tree**: `_remove_overlaps` uses a naive $O(N^2)$ pair check instead of a spatial index (R-Tree), which will be slow for large boards.
- **Polygon Outlines**: Boundary enforcement and edge clearance checks still use rectangular `origin/width/height` even when polygon outlines are available, so components can violate complex outlines without detection. Files: `atoplace/placement/force_directed.py`, `atoplace/validation/confidence.py`, `atoplace/validation/drc.py`.
- **Cutout Clearance**: `BoardOutline.contains_point` enforces margin against the outer polygon but does not apply margin checks against holes, so clearance to cutouts is not enforced. File: `atoplace/board/abstraction.py`.
- **Validation Defaults**: `ConfidenceScorer._check_boundaries` and `DRCChecker._check_edge_clearance` still use rectangular bounding boxes rather than `BoardOutline.contains_point`, so polygon outlines and holes are ignored during validation. Files: `atoplace/validation/confidence.py`, `atoplace/validation/drc.py`.
- **DFM Placement Gap**: `cmd_place` accepts a DFM profile but placement spacing uses hardcoded `RefinementConfig` defaults; DFM spacing only affects validation, not placement forces. Files: `atoplace/cli.py`, `atoplace/placement/force_directed.py`.
- **Polygon Utilization**: `ConfidenceScorer._check_density` uses rectangular `outline.width * outline.height` even for polygon boards, which can substantially misreport utilization on irregular outlines. File: `atoplace/validation/confidence.py`.
- **Attraction Weighting**: Net attraction uses unique component refs only, ignoring multiple pads/pins on the same net. Multi-pin ICs are underweighted vs single-pin passives, which can skew net-based attraction. File: `atoplace/placement/force_directed.py`.
- **Legalization Ignores Outline**: The legalizer's grid snapping and overlap resolution do not check board boundaries or polygon outlines, so legalization can push components outside the board after refinement. File: `atoplace/placement/legalizer.py`.
- **Convergence Scaling**: Energy variance threshold is absolute and not normalized by component count or board scale; using total force magnitudes means convergence behavior varies widely with board size and net count. File: `atoplace/placement/force_directed.py`.
- **Preferred Clearance Forces**: The "between min and preferred clearance" repulsion uses the minimum axis shortfall, which can under-repel when one axis is tight and the other is slack, leaving components too close in one dimension. File: `atoplace/placement/force_directed.py`.
- **Unbounded Attraction**: Attraction force scales linearly with distance and has no cap or decay; with repulsion cut off at 50mm, long nets can dominate and pull clusters across the board. File: `atoplace/placement/force_directed.py`.
- **Mass/Size Blindness**: Force integration treats all components equally; large connectors/ICs move as easily as small passives, which can yield unrealistic placement dynamics. File: `atoplace/placement/force_directed.py`.
- **Locked Overlaps Persist**: When `lock_placed` is enabled, locked components are skipped in force updates entirely, so overlaps involving locked parts are never resolved and can remain indefinitely. File: `atoplace/placement/force_directed.py`.
- **Legalizer Spacing Drift**: `_distribute_evenly` enforces `row_spacing + min_clearance` but does not update the cached component sizes afterward; later overlap resolution may use stale sizes for moved components. File: `atoplace/placement/legalizer.py`.
- **Legalizer Grid Conflict**: Overlap resolution snaps to `secondary_grid` even for large components, which can drift them off the primary grid used for snapping and alignment. File: `atoplace/placement/legalizer.py`.

### Medium - RESOLVED (2026-01-12 Session)

- ~~**Validation**: `PreRouteValidator` now uses DFM-driven spacing, but `cmd_validate` instantiates it without the selected DFM profile.~~ **FIXED**: `cmd_validate` now passes `dfm_profile` to `PreRouteValidator` constructor.

- ~~**Legalizer Passive Detection**: `_detect_passive_size` relies on strict string matching ("0402") for imperial codes only.~~ **FIXED**: Added `METRIC_TO_IMPERIAL` mapping dictionary supporting metric codes (0603, 1005, 1608, 2012, 3216, 3225, 5025, 6332). Detection now checks both imperial and metric patterns.

- ~~**DFM Minimum Board Size Unchecked**: `DFMProfile.min_board_size` is defined but never enforced in confidence checks.~~ **FIXED**: Added minimum board size check in `_check_dfm()` that flags undersized boards with a WARNING severity and 0.9 score penalty.

- ~~**Boundary Forces Lack Projection**: Boundary correction overwrites `fx` or `fy` without combining forces when both axes violated.~~ **FIXED**: Changed `=` to `+=` for force accumulation in `_add_boundary_forces()` so corner violations properly combine X and Y corrections.

- ~~**Convergence Criteria**: `refine()` stops when EITHER max movement OR energy variance is low, risking frozen high-energy states.~~ **FIXED**: Now requires BOTH low movement AND (low variance OR low energy) to converge. Also checks `total_energy` threshold as fallback before energy history is populated.

- ~~**Unused Connectivity Matrix**: `_connectivity_matrix` is computed but never used, adding startup cost.~~ **FIXED**: Commented out the computation with a note explaining it's reserved for future pin-count weighted attraction enhancement.

- ~~**Atopile Lock Schema Mismatch**: `_apply_component_metadata` reads `lock_data["components"]`, but `instance_to_ref_map` iterates `lock_data` at the top level.~~ **FIXED**: `instance_to_ref_map` now handles both formats: root-level entries with `designator` and entries under `components` key with `address` field. Also maps KiCad refs to themselves for direct lookup.

- ~~**Constraint Parsing Warnings Dropped**: `ConstraintParser.parse_interactive` returns warnings but CLI paths never surface them.~~ **FIXED**: Interactive mode now prints summary even when no constraints found, ensuring warnings are visible.

- ~~**DFM API Export Gap**: `list_profiles` exists in `atoplace/dfm/profiles.py` but is not exported in `__init__.py`.~~ **FIXED**: Added `list_profiles` to `atoplace/dfm/__init__.py` exports.

- ~~**Atopile Build Detection**: `detect_board_source` always calls `get_board_path()` with the default build.~~ **FIXED**: Now accepts optional `build_name` parameter and falls back to first available build if "default" doesn't exist.

## 2026-01-12 - FIXES APPLIED

### High - RESOLVED
- ~~**Atopile Integration**: Fallback YAML parser is unreliable; Module mapping is broken (instance name vs reference designator).~~ **FIXED**: Added PyYAML>=6.0.0 to dependencies. Implemented `instance_to_ref_map` property to map atopile instance paths to KiCad designators. Updated `_apply_module_hierarchy` to use this mapping with fallback strategies. Wired up `--use-ato-modules` in CLI to create GroupingConstraints from atopile modules.

- ~~**Placement Physics**: High-degree net collapse ($O(N^2)$ attraction) and unused connectivity matrix.~~ **FIXED**: Implemented Hybrid Net Model in `_add_attraction_forces` - small nets (<=3 pins) use pairwise attraction, large nets use Star Model with centroid attraction scaled by 1/k. Replaced diagonal radius with AABB checks in boundary forces. Added rolling average energy convergence detection with configurable window and variance threshold.

### Medium - RESOLVED
- ~~**Board Outline Extraction**: `_extract_outline` currently uses `GetBoardEdgesBoundingBox`, reducing complex shapes to a rectangle.~~ **FIXED**: Implemented `_extract_polygon_outline` to chain Edge.Cuts segments (lines, arcs, circles, rectangles, polygons) into closed polygons. `BoardOutline` now supports `polygon` vertices and `holes`. Implemented `contains_point` using Ray Casting algorithm with margin support.

- ~~**Board Adapter**: Differential pair detection only flags nets ending in `+` or `_P`.~~ **FIXED**: `_extract_net` now also marks `-` and `_N` suffixed nets as differential pairs with correct pair references.

- ~~**DRC**: Pad size check compares component pad dimensions against `min_via_annular * 2`.~~ **FIXED**: `_check_minimum_sizes` now differentiates between through-hole pads (checked against drill and annular ring rules) and SMD pads (checked against min_spacing).

- ~~**CLI**: `--use-ato-modules` flag is defined but ignored in `cmd_place`.~~ **FIXED**: When flag is set and board is from atopile, creates GroupingConstraints for each module with 2+ components.

- ~~**Placement Physics**: Boundary forces use component diagonal as a circular margin.~~ **FIXED**: `_compute_component_sizes` now returns AABB half-dimensions accounting for rotation. `_add_boundary_forces` and `_add_repulsion_forces` use proper AABB collision detection.

- ~~**Placement Physics**: Convergence check only looks at `max_movement`.~~ **FIXED**: Added energy history tracking with configurable `energy_window` and `energy_variance_threshold` for detecting oscillation and stall conditions.

## 2026-01-12 - FIXES APPLIED (Batch 2)

### High - RESOLVED
- ~~**NLP Modification Logic**: `ModificationHandler` fails to execute "move closer to X" or "move away from X" commands.~~ **FIXED**: Split `move` pattern into `move` (directional) and `move_relative` (closer to/away from). Now captures target component separately. Updated `_extract_modification` and `apply_modification` to properly handle the target reference.

### Medium - RESOLVED
- ~~**Pre-Route Validation**: `_check_overlapping_pads` uses hardcoded clearance and coarse grid.~~ **FIXED**: Now uses DFM profile's `min_spacing` for clearance. Grid size is dynamically calculated based on minimum pad dimensions to handle fine-pitch components. Added `dfm_profile` parameter to `PreRouteValidator.__init__`.

- ~~**Confidence Scoring**: `_check_decoupling` enforces hard 5mm distance limit.~~ **FIXED**: Now uses adaptive distance limits based on IC type (high-speed: <2mm, digital: <5mm, standard: <10mm). Detects IC speed class from value/footprint patterns (USB, ETH, STM32, etc.).

- ~~**Pad Geometry**: Pad rotation not stored or applied.~~ **FIXED**: Added `rotation` field to `Pad` class. Added `absolute_rotation()` and `get_bounding_box()` methods. Updated `_pad_to_pad` to extract pad rotation from KiCad. Updated `_check_overlapping_pads` to use rotated bounding boxes.

## 2025-02-14 - RESOLVED (Historical)

### High - RESOLVED
- ~~Saving to a new path creates an empty KiCad board.~~ **FIXED**: `save_kicad_board` now copies source file when output path differs.
- ~~Constraint integration mismatch / duplication.~~ **FIXED**: `_add_constraint_forces` supports both interfaces via duck typing.
- ~~Module detection enum lookup can crash.~~ **FIXED**: Added `PATTERN_TO_MODULE_TYPE` mapping dict.

### Medium - RESOLVED
- ~~Pad positions double-rotated.~~ **FIXED**: `_pad_to_pad` reverse-rotates before storing local coordinates.
- ~~Component dimensions from rotated bbox.~~ **FIXED**: `_estimate_unrotated_dimensions` uses pad extents.
- ~~Pad layer not mapped.~~ **FIXED**: `_map_pad_layer` maps from KiCad.
- ~~Regex overlap detection incomplete.~~ **FIXED**: Updated to handle all overlap cases.
- ~~`lock_placed`/`preferred_clearance` unused.~~ **FIXED**: Implemented in force calculations.
- ~~Inverse-square repulsion too strong.~~ **FIXED**: Added cutoff and changed to inverse-distance.
- ~~Separation force inverted.~~ **FIXED**: Removed incorrect multiplier.
- ~~Rotation constraints not enforced.~~ **FIXED**: Added `_apply_rotation_constraints`.
- ~~Import stubs missing.~~ **FIXED**: Lazy imports with error messages.
- ~~Analog/Digital grouping placeholder refs.~~ **FIXED**: Now resolves actual component references.
- ~~Move/flip modification not implemented.~~ **FIXED**: Implemented `move` handler.
