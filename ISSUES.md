# Project Issues Log

This file tracks code review findings and risks discovered during development.

## 2026-01-11 - REVIEW FINDINGS

### High - OPEN
- **Atopile Integration**: Fallback YAML parser (`_parse_simple_yaml`) fails on nested structures (critical for `ato.yaml`). It flattens section headers and skips nested content, making project loading unreliable without PyYAML. File: `atoplace/board/atopile_adapter.py`.
- **Atopile Module Mapping**: `AtopileModuleParser` extracts instance names (e.g., `res1`) but `Board` uses reference designators (e.g., `R1`). There is no mapping mechanism, so `ato_module` metadata is not applied to components. File: `atoplace/board/atopile_adapter.py`.
- **Placement Physics**: `_add_attraction_forces` iterates all pairs in a net. For high-degree nets (GND/VCC), this creates massive attraction forces ($O(N^2)$), causing component collapse. `_build_connectivity_matrix` is computed but unused. File: `atoplace/placement/force_directed.py`.

### Medium - OPEN
- **Placement Physics**: Boundary forces use component diagonal as a circular margin. This is inaccurate for long rectangular components and can push them off-board or constrain them too tightly. File: `atoplace/placement/force_directed.py`.
- **Placement Physics**: Convergence check only looks at `max_movement`. No oscillation or stall detection. File: `atoplace/placement/force_directed.py`.
- **Board Outline Extraction**: `_extract_outline` currently uses `GetBoardEdgesBoundingBox`, reducing complex shapes to a rectangle.
    - **Fix Strategy**:
        1. Update `BoardOutline` in `atoplace/board/abstraction.py` to support `polygon` (list of vertices) and `holes`.
        2. Update `_extract_outline` in `atoplace/board/kicad_adapter.py` to use `kicad_board.GetBoardPolygonOutlines()` (returns `SHAPE_POLY_SET`) to get the true outline. If unavailable, iterate `Edge.Cuts` drawings to chain segments.
        3. Implement `contains_point` using the Ray Casting algorithm (even-odd rule) or integrate `shapely`.
        4. Update `ForceDirectedRefiner` to use `contains_point` for boundary forces instead of simple bounds checking.
- **Board Adapter**: Differential pair detection only flags nets ending in `+` or `_P`. The corresponding `-` or `_N` nets are not marked as differential. File: `atoplace/board/kicad_adapter.py`.
- **DRC**: Pad size check compares component pad dimensions against `min_via_annular * 2`. This conflates via rules with SMD pad requirements, likely causing false positives. File: `atoplace/validation/drc.py`.
- **CLI**: `--use-ato-modules` flag is defined but ignored in `cmd_place`. File: `atoplace/cli.py`.

## 2025-02-14 - RESOLVED

### High - RESOLVED
- ~~Constraint integration mismatch: `ForceDirectedRefiner` expects constraints with `calculate_forces(...)`, while `ConstraintParser` returns constraints that only implement `calculate_force(...)`.~~ **FIXED**: `_add_constraint_forces` now supports both interfaces via duck typing.
- ~~Module detection enum lookup can crash: pattern keys like `rf`, `esd`, `opamp`, and `crystal` do not exist as `ModuleType` members.~~ **FIXED**: Added `PATTERN_TO_MODULE_TYPE` mapping dict.

### Medium - RESOLVED
- ~~Separation force direction is inverted for group_b.~~ **FIXED**: Removed incorrect `direction` multiplier.
- ~~Rotation constraints are parsed but never enforced.~~ **FIXED**: Added `_apply_rotation_constraints` method to refiner and updated `FixedConstraint` to support rotation-only mode.
- ~~Import stubs point to missing modules.~~ **FIXED**: Changed to lazy imports with helpful error messages.
- ~~"Analog"/"Digital" grouping/separation returns placeholder refs.~~ **FIXED**: `_get_analog_components` and `_get_digital_components` now resolve to actual component references.
- ~~Modification parsing recognizes `move` and `flip`, but `apply_modification` only implements `rotate` and `swap`.~~ **FIXED**: Implemented `move` handler with directional support.

## 2025-02-14 (KiCad Adapter Review) - PARTIALLY RESOLVED

### High - RESOLVED
- ~~Saving to a new path creates an empty KiCad board.~~ **FIXED**: `save_kicad_board` now copies source file when output path differs.

### Medium - RESOLVED
- ~~Pad positions are stored in board coordinates relative to the footprint origin, then rotated again in `Pad.absolute_position`.~~ **FIXED**: `_pad_to_pad` reverse-rotates pad positions before storing local coordinates.
- ~~Component `width`/`height` are taken from the axis-aligned KiCad bounding box in board coordinates.~~ **FIXED**: `_estimate_unrotated_dimensions` estimates unrotated dimensions using pad extents.
- ~~Pad layer is not mapped from KiCad pad layers, so pads default to top copper.~~ **FIXED**: `_map_pad_layer` maps pad layers from KiCad.

### Medium - OPEN
- Pre-route overlapping pad check is still grid-binned at 0.1 mm and can flag false positives for adjacent pads in different components. File: `atoplace/validation/pre_route.py`.
- Board outline is reduced to a rectangle via the board edges bounding box; complex outlines are lost, which invalidates edge/zone constraints and boundary forces. File: `atoplace/board/kicad_adapter.py`.

## 2025-02-14 (NLP Parsing Review) - PARTIALLY RESOLVED

### Medium - RESOLVED
- ~~Overlap detection for regex matches can miss full-span overlaps (only checks partial overlap), allowing duplicate or conflicting constraints from the same text.~~ **FIXED**: Updated overlap check to handle all cases: start inside, end inside, new contains existing, and existing contains new.

### Medium - OPEN
- Modification patterns for "closer to"/"away from" do not capture a target reference, so move adjustments toward/away from a specific component never apply. File: `atoplace/nlp/constraint_parser.py`.

## 2025-02-14 (Placement Physics Review) - MOSTLY RESOLVED

### High - RESOLVED
- ~~Constraint system duplication: `force_directed.py` defines its own `PlacementConstraint` hierarchy with `calculate_forces(...)`, but the primary constraints live in `placement/constraints.py` with `calculate_force(...)`.~~ **FIXED**: `_add_constraint_forces` now supports both interfaces.

### Medium - RESOLVED
- ~~`lock_placed` and `preferred_clearance` are defined in `RefinementConfig` but never used.~~ **FIXED**: Added `_is_component_locked` method and skip locked components in `_apply_forces`. Updated `_add_repulsion_forces` to use `preferred_clearance` for spacing.
- ~~Repulsion applies inverse-square force at all distances, which keeps long-range forces large in dense boards.~~ **FIXED**: Added 50mm cutoff distance and changed to inverse-distance (not inverse-square) for distant components.

### Medium - OPEN
- `_build_connectivity_matrix` is computed but never used, and attraction forces always use all pairwise nets, which overweights high-degree nets and can cause component collapse on large nets. File: `atoplace/placement/force_directed.py`.
- Boundary forces use a circular "size" (diagonal) margin, which under-approximates rotated rectangles for tall/narrow parts and can still leave corners outside the board. File: `atoplace/placement/force_directed.py`.
- Convergence check uses only max movement; no oscillation/stall detection, so placement can stop in high-energy jitter or drift if damping is small. File: `atoplace/placement/force_directed.py`.

## 2025-02-14 (Atopile Adapter Review) - OPEN

### Medium - OPEN
- Fallback YAML parser in `AtopileProjectLoader` does not parse nested structures, so builds/entry points are lost when PyYAML is unavailable; atopile loading can fail silently or choose wrong paths. File: `atoplace/board/atopile_adapter.py`.
- Module hierarchy parser records instance names rather than KiCad reference designators; module-to-component mapping is likely empty, so `ato_module` metadata is not applied. File: `atoplace/board/atopile_adapter.py`.

### High - OPEN
- PyYAML is not a declared dependency, so `AtopileProjectLoader` almost always falls back to its limited parser, making atopile project loading unreliable by default. File: `pyproject.toml`, `atoplace/board/atopile_adapter.py`.

## 2025-02-14 (Board/DRC Review) - OPEN

### Medium - OPEN
- Polygonal board outlines are not supported: `BoardOutline.contains_point` raises `NotImplementedError` when `polygon` is set, so non-rectangular outlines cannot be validated. File: `atoplace/board/abstraction.py`.
- DRC pad size check derives minimum pad size from `min_via_annular * 2`, which conflates via annular ring and pad size requirements and can produce false warnings. File: `atoplace/validation/drc.py`.

## 2025-02-14 (Net Classification Review) - OPEN

### Low - OPEN
- Differential pair detection only marks nets ending in `+` or `_P`; the corresponding `-`/`_N` nets are not marked as differential, which can break downstream logic that expects both nets flagged. File: `atoplace/board/kicad_adapter.py`.

## 2025-02-14 (CLI Review) - OPEN

### Low - OPEN
- `--use-ato-modules` flag is defined but never used, so users cannot enable atopile module grouping as described in CLI help. File: `atoplace/cli.py`.