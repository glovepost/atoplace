# Project Issues Log

This file tracks code review findings and risks discovered during development.

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
