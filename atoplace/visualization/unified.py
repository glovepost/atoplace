"""Unified visualizer combining placement and routing into a single timeline.

This module provides a UnifiedVisualizer class that merges placement and routing
visualization into a single HTML file, enabling seamless workflow visualization
from initial placement through routing completion.

The viewer supports:
- Placement phase: Component movement, forces, overlaps, module groups
- Routing phase: Traces appearing, vias, A* debug visualization
- Seamless transitions between phases
- All existing layer controls and visual settings
"""

import json
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from ..visualization_color_manager import get_color_manager

logger = logging.getLogger(__name__)


@dataclass
class UnifiedFrame:
    """A single frame in the unified visualization timeline.

    Supports both placement and routing data in a single frame structure.
    Either or both can be populated depending on the workflow phase.
    """
    index: int
    label: str = ""
    phase: str = ""  # "placement", "routing", "transition"
    iteration: int = 0

    # Placement data (component positions)
    changed_components: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    changed_modules: Dict[str, str] = field(default_factory=dict)
    forces: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)
    overlaps: List[Tuple[str, str]] = field(default_factory=list)

    # Routing data
    traces: List[Dict[str, Any]] = field(default_factory=list)
    vias: List[Dict[str, Any]] = field(default_factory=list)
    astar_explored: List[Tuple[float, float, int]] = field(default_factory=list)
    astar_frontier: List[Tuple[float, float, int]] = field(default_factory=list)
    astar_path: List[Tuple[float, float, int]] = field(default_factory=list)

    # Metrics (shared between phases)
    energy: float = 0.0
    max_move: float = 0.0
    overlap_count: int = 0
    wire_length: float = 0.0

    # Routing-specific metrics
    nets_routed: int = 0
    current_net: str = ""


class UnifiedVisualizer:
    """Unified visualizer combining placement and routing visualization.

    Usage:
        viz = UnifiedVisualizer(board)

        # Add placement frames from PlacementVisualizer
        viz.add_placement_frames(placement_visualizer.frames)

        # Add routing frames from RouteVisualizer
        viz.add_routing_frames(route_visualizer.frames)

        # Export unified HTML
        viz.export_html("unified_viz.html")
    """

    def __init__(self, board, scale: float = 10.0, margin: float = 20.0):
        """Initialize unified visualizer.

        Args:
            board: Board instance from atoplace.board.abstraction
            scale: Rendering scale (pixels per mm)
            margin: Margin around board in mm (larger to show components outside board edge)
        """
        self.board = board
        self.scale = scale
        self.margin = margin

        # Unified frame timeline
        self.frames: List[UnifiedFrame] = []

        # Static component properties (computed once from board)
        self.static_props: Dict[str, Dict[str, Any]] = {}
        self.component_layers: Dict[str, str] = {}
        self.netlist: Dict[str, List[Tuple[str, int]]] = {}
        self.module_colors: Dict[str, str] = {}

        # Board bounds
        self._compute_board_bounds()
        self._extract_static_props()
        self._build_netlist()

    def _compute_board_bounds(self):
        """Compute board bounding box.

        Uses board outline if available, then falls back to component bounds,
        and finally uses a safe default (100x100mm centered at origin) if the
        board has no outline and no components.
        """
        bounds = None

        if self.board.outline:
            if hasattr(self.board.outline, 'get_bounding_box'):
                bbox = self.board.outline.get_bounding_box()
                if bbox:
                    bounds = bbox
            elif hasattr(self.board.outline, 'polygon') and self.board.outline.polygon:
                xs = [p[0] for p in self.board.outline.polygon]
                ys = [p[1] for p in self.board.outline.polygon]
                bounds = (min(xs), min(ys), max(xs), max(ys))

        if not bounds:
            # Fall back to component bounds
            if self.board.components:
                min_x = min_y = float('inf')
                max_x = max_y = float('-inf')
                for comp in self.board.components.values():
                    bbox = comp.get_bounding_box()
                    min_x = min(min_x, bbox[0])
                    min_y = min(min_y, bbox[1])
                    max_x = max(max_x, bbox[2])
                    max_y = max(max_y, bbox[3])
                bounds = (min_x - 5, min_y - 5, max_x + 5, max_y + 5)
            else:
                # Empty board with no outline and no components:
                # Use a safe default 100x100mm region centered at origin
                logger.warning(
                    "Board has no outline and no components. "
                    "Using default bounds (100x100mm centered at origin)."
                )
                bounds = (-50, -50, 50, 50)

        self.bounds = bounds
        self.board_width = bounds[2] - bounds[0]
        self.board_height = bounds[3] - bounds[1]
        self.svg_width = (self.board_width + 2 * self.margin) * self.scale
        self.svg_height = (self.board_height + 2 * self.margin) * self.scale

    def _extract_static_props(self):
        """Extract static component properties from board.

        Pad data format: [x, y, width, height, rotation, net, layer, is_through_hole]
        - layer: 'top', 'bottom', or 'inner' (for inner layer pads)
        - is_through_hole: true if pad has drill hole (spans all layers)
        """
        from ..board.abstraction import Layer

        def layer_to_string(layer: Layer) -> str:
            """Convert Layer enum to string for JavaScript."""
            if layer == Layer.TOP_COPPER:
                return 'top'
            elif layer == Layer.BOTTOM_COPPER:
                return 'bottom'
            elif layer in (Layer.INNER1, Layer.INNER2, Layer.INNER3, Layer.INNER4):
                return 'inner'
            else:
                return 'top'  # Default to top for unknown layers

        for ref, comp in self.board.components.items():
            # Pad geometry with layer information (Issue #37)
            pad_geom = []
            for pad in comp.pads:
                pad_rot = getattr(pad, 'rotation', 0.0) or 0.0
                pad_layer = layer_to_string(pad.layer)
                is_through_hole = pad.drill is not None and pad.drill > 0
                pad_geom.append([
                    pad.x,
                    pad.y,
                    pad.width,
                    pad.height,
                    pad_rot,
                    pad.net or "",
                    pad_layer,
                    is_through_hole
                ])

            self.static_props[ref] = {
                'width': comp.width,
                'height': comp.height,
                'pads': pad_geom
            }

            # Component layer
            if comp.layer == Layer.TOP_COPPER:
                self.component_layers[ref] = 'top'
            elif comp.layer == Layer.BOTTOM_COPPER:
                self.component_layers[ref] = 'bottom'
            else:
                self.component_layers[ref] = 'top'

    def _build_netlist(self):
        """Build netlist from board for ratsnest visualization."""
        for ref, comp in self.board.components.items():
            if getattr(comp, 'dnp', False):
                continue
            for pad_idx, pad in enumerate(comp.pads):
                net_name = (pad.net or "").strip()
                if not net_name:
                    continue
                self.netlist.setdefault(net_name, []).append([ref, pad_idx])

        # Sort for consistent ordering
        for net_name in sorted(self.netlist.keys()):
            self.netlist[net_name] = sorted(self.netlist[net_name])

    def add_placement_frames(self, placement_frames, static_props=None, component_layers=None):
        """Add placement frames to the timeline.

        Args:
            placement_frames: List of PlacementFrame objects from PlacementVisualizer
            static_props: Optional dict of precomputed static component properties.
                If provided, these will be merged into self.static_props, avoiding
                redundant extraction from the board. Accepts either:
                - Dict[str, ComponentStaticProps] (from PlacementVisualizer.static_props)
                - Dict[str, Dict[str, Any]] with keys 'width', 'height', 'pads'
            component_layers: Optional dict mapping component refs to layer strings
                ('top' or 'bottom'). If provided with static_props, avoids re-extracting
                layer information from the board.
        """
        # Merge provided static_props if given (Issue #33)
        if static_props is not None:
            for ref, props in static_props.items():
                # Handle ComponentStaticProps dataclass or plain dict
                if hasattr(props, 'width') and not isinstance(props, dict):
                    # ComponentStaticProps dataclass - convert to dict format
                    pad_geom = []
                    if hasattr(props, 'pads') and props.pads:
                        for pad in props.pads:
                            # PadStaticProps: x, y, width, height, rotation, net
                            pad_rot = getattr(pad, 'rotation', 0.0) or 0.0
                            pad_net = getattr(pad, 'net', '') or ''
                            # Use extended format if layer info available, else basic format
                            pad_layer = getattr(pad, 'layer', 'top') or 'top'
                            is_through_hole = getattr(pad, 'is_through_hole', False)
                            pad_geom.append([
                                pad.x, pad.y, pad.width, pad.height,
                                pad_rot, pad_net, pad_layer, is_through_hole
                            ])
                    self.static_props[ref] = {
                        'width': props.width,
                        'height': props.height,
                        'pads': pad_geom
                    }
                elif isinstance(props, dict) and 'width' in props:
                    # Already in dict format - use as-is
                    self.static_props[ref] = props
                # else: skip invalid entries

        # Merge provided component_layers if given (Issue #33)
        if component_layers is not None:
            self.component_layers.update(component_layers)

        start_idx = len(self.frames)

        for i, pf in enumerate(placement_frames):
            # Convert PlacementFrame to UnifiedFrame
            uf = UnifiedFrame(
                index=start_idx + i,
                label=pf.label,
                phase="placement",
                iteration=pf.iteration,
                changed_components=dict(pf.components),
                changed_modules=dict(pf.modules) if pf.modules else {},
                forces=dict(pf.forces) if pf.forces else {},
                overlaps=list(pf.overlaps) if pf.overlaps else [],
                energy=pf.energy,
                max_move=pf.max_move,
                overlap_count=pf.overlap_count,
                wire_length=pf.total_wire_length,
            )
            self.frames.append(uf)

            # Collect module colors
            for ref, module_name in pf.modules.items():
                if module_name and module_name not in self.module_colors:
                    self.module_colors[module_name] = get_color_manager().get_module_color(module_name)

        logger.info(f"Added {len(placement_frames)} placement frames (total: {len(self.frames)})")

    def add_routing_frames(self, routing_frames):
        """Add routing frames to the timeline.

        Args:
            routing_frames: List of VisualizationFrame objects from RouteVisualizer
        """
        start_idx = len(self.frames)

        # Get final component positions from last placement frame (if any)
        final_components = {}
        final_modules = {}
        for f in reversed(self.frames):
            if f.phase == "placement" and f.changed_components:
                final_components = dict(f.changed_components)
                final_modules = dict(f.changed_modules)
                break

        # Track cumulative routing metrics across frames (Issue #34)
        routed_nets = set()

        for i, rf in enumerate(routing_frames):
            # Convert VisualizationFrame to UnifiedFrame
            traces = []
            frame_wire_length = 0.0
            for trace in rf.completed_traces:
                # Prefer net_name over net_id for matching netlist (Issue #31)
                net = trace.net_name or (str(trace.net_id) if trace.net_id else '')
                traces.append({
                    'start': [trace.start[0], trace.start[1]],
                    'end': [trace.end[0], trace.end[1]],
                    'layer': trace.layer,
                    'width': trace.width,
                    'net': net
                })
                # Track routed nets and calculate wire length (Issue #34)
                if net:
                    routed_nets.add(net)
                # Calculate segment length
                dx = trace.end[0] - trace.start[0]
                dy = trace.end[1] - trace.start[1]
                frame_wire_length += math.sqrt(dx * dx + dy * dy)

            vias = []
            for via in rf.completed_vias:
                # Prefer net_name over net_id for matching netlist (Issue #31)
                net = via.net_name or (str(via.net_id) if via.net_id else '')
                vias.append({
                    'x': via.x,
                    'y': via.y,
                    'drill': via.drill_diameter,
                    'pad': via.pad_diameter,
                    'net': net
                })
                # Track routed nets from vias too (Issue #34)
                if net:
                    routed_nets.add(net)

            # First routing frame includes component positions for continuity
            changed_components = final_components if i == 0 else {}
            changed_modules = final_modules if i == 0 else {}

            uf = UnifiedFrame(
                index=start_idx + i,
                label=rf.label or f"Routing {i}",
                phase="routing",
                iteration=rf.iteration,
                changed_components=changed_components,
                changed_modules=changed_modules,
                traces=traces,
                vias=vias,
                astar_explored=list(rf.explored_nodes) if rf.explored_nodes else [],
                astar_frontier=list(rf.frontier_nodes) if rf.frontier_nodes else [],
                astar_path=rf.current_path if rf.current_path else [],
                current_net=rf.current_net_name or "",
                # Populate routing metrics (Issue #34)
                wire_length=frame_wire_length,
                nets_routed=len(routed_nets),
            )
            self.frames.append(uf)

        logger.info(f"Added {len(routing_frames)} routing frames (total: {len(self.frames)})")

    def add_transition_frame(self, label: str = "Placement Complete → Starting Routing"):
        """Add a transition frame between placement and routing phases."""
        if not self.frames:
            return

        # Get final state from last frame
        last_frame = self.frames[-1]

        uf = UnifiedFrame(
            index=len(self.frames),
            label=label,
            phase="transition",
            iteration=0,
            changed_components={},  # No changes, just a marker
            energy=last_frame.energy,
            wire_length=last_frame.wire_length,
            overlap_count=last_frame.overlap_count,
        )
        self.frames.append(uf)

    def _convert_frames_to_delta_format(self) -> List[Dict[str, Any]]:
        """Convert unified frames to delta format for the JavaScript viewer.

        DELTA ENCODING CONTRACT
        =======================

        This method produces delta frames that are consumed by `svg-delta-viewer.js`.
        The JavaScript viewer uses incremental `applyDelta()` when playing forward
        and full `reconstructState()` when seeking backward or jumping. Understanding
        this contract is critical for correct visualization behavior.

        How the JavaScript Viewer Handles Deltas
        ----------------------------------------

        The viewer maintains a `currentState` object that accumulates state across
        frames. The `applyDelta(delta)` function in JavaScript:

        1. **For `changed_components`**: Merges into `currentState.components`.
           - Keys present in delta update `currentState.components[ref]`
           - Keys NOT present in delta are left unchanged (kept from prior frame)
           - Values can use `null` for individual fields: `[x, null, rot]` means
             "update x and rotation, keep previous y"

        2. **For `changed_modules`**: Merges into `currentState.modules`.
           - Keys present update the module assignment
           - Keys NOT present are left unchanged
           - No `null` clearing is currently supported for modules

        3. **For overlay arrays** (traces, vias, astar_*, overlaps, forces):
           - If the key is PRESENT in delta, it REPLACES the current state entirely
           - If the key is OMITTED from delta, the PREVIOUS value is kept
           - To CLEAR an overlay, include the key with an empty array/dict `[]` or `{}`

        4. **For scalar metrics** (energy, max_move, overlap_count, wire_length,
           nets_routed, current_net):
           - If the key is PRESENT, the value replaces `currentState`
           - If the key is OMITTED, the previous value is kept
           - These should be present every frame for consistent info bar display

        Frame 0 Requirements (Full Snapshot)
        ------------------------------------

        Frame 0 MUST be a complete snapshot containing:
        - ALL components with full `[x, y, rotation]` values (no nulls)
        - ALL module assignments for components that have modules
        - ALL metrics initialized (energy, max_move, overlap_count, wire_length)
        - Empty arrays for overlays if none exist yet

        This is required because `reconstructState(toFrame)` replays deltas from
        frame 0 when seeking backward. Without a complete frame 0, components
        would appear at origin (0, 0, 0) instead of their actual starting positions.

        Delta Encoding Rules
        --------------------

        1. **Component Positions** (`changed_components`):
           - Format: `{ref: [x, y, rotation]}` or `{ref: [x, null, rotation]}`
           - Include ref ONLY when its position/rotation changes
           - Omit ref entirely if nothing changed for that component
           - Frame 0 must include ALL components with full [x, y, rotation] values

        2. **Module Assignments** (`changed_modules`):
           - Format: `{ref: "module_name"}`
           - Include ref ONLY when its module assignment changes
           - Omit ref if module hasn't changed
           - No `null` to clear modules is currently supported

        3. **Overlay Arrays** (traces, vias, astar_explored, astar_frontier,
           astar_path, overlaps, forces):
           - ALWAYS include these keys EVERY frame (even if empty)
           - Use empty list `[]` or dict `{}` to clear the overlay
           - Omitting the key causes stale data to persist (Issue #30)

        4. **Scalar Metrics** (energy, max_move, overlap_count, wire_length,
           current_net, nets_routed):
           - ALWAYS include these keys EVERY frame
           - The info bar does not back-fill missing values
           - Default to 0 or "" if no meaningful value exists

        5. **Sequential Playback**:
           - Forward playback uses incremental `applyDelta()` (fast path)
           - Backward seeking or jumping replays from frame 0
           - Deltas must be relative to prior state, not absolute snapshots
           - Frame 0 must be a complete snapshot for correct reconstruction

        Example Delta Frame
        -------------------

        Frame 0 (full snapshot):
        ```json
        {
            "index": 0,
            "label": "Initial",
            "phase": "placement",
            "iteration": 0,
            "changed_components": {"U1": [10.0, 20.0, 0.0], "R1": [15.0, 25.0, 90.0]},
            "changed_modules": {"U1": "mcu", "R1": "mcu"},
            "forces": {},
            "overlaps": [],
            "energy": 150.5,
            "max_move": 0.0,
            "overlap_count": 2,
            "wire_length": 45.3,
            "traces": [],
            "vias": [],
            "astar_explored": [],
            "astar_frontier": [],
            "astar_path": [],
            "current_net": "",
            "nets_routed": 0
        }
        ```

        Frame N (delta with changes):
        ```json
        {
            "index": 5,
            "label": "Iteration 5",
            "phase": "placement",
            "iteration": 5,
            "changed_components": {"U1": [10.5, 20.2, 0.0]},  // Only U1 moved
            "changed_modules": {},  // No module changes
            "forces": {"U1": [[0.1, 0.2, "repulsion"]]},  // Replace previous
            "overlaps": [],  // Clear overlaps
            "energy": 145.2,
            "max_move": 0.5,
            "overlap_count": 1,
            "wire_length": 44.8,
            "traces": [],  // Keep empty
            "vias": [],
            "astar_explored": [],
            "astar_frontier": [],
            "astar_path": [],
            "current_net": "",
            "nets_routed": 0
        }
        ```

        See Also
        --------
        - `atoplace/visualization/assets/svg-delta-viewer.js`: JavaScript viewer
          implementation with `applyDelta()` and `reconstructState()` functions
        - Issue #30: Routing debug layers never clear (fixed by always including keys)
        - Issue #39: This documentation addresses the delta contract guidance need

        Returns:
            List of delta frame dictionaries ready for JSON serialization to JavaScript
        """
        delta_frames = []

        for frame in self.frames:
            # Core frame metadata - always present
            delta = {
                'index': frame.index,
                'label': frame.label,
                'phase': frame.phase,
                'iteration': frame.iteration,
            }

            # Component positions and modules (merged incrementally by JS)
            delta['changed_components'] = frame.changed_components
            delta['changed_modules'] = frame.changed_modules

            # Overlay arrays - ALWAYS include to ensure proper clearing
            # (see Delta Encoding Rules #3 above)
            delta['forces'] = frame.forces
            delta['overlaps'] = frame.overlaps

            # Scalar metrics - ALWAYS include for consistent info bar display
            # (see Delta Encoding Rules #4 above)
            delta['energy'] = frame.energy
            delta['max_move'] = frame.max_move
            delta['overlap_count'] = frame.overlap_count
            delta['wire_length'] = frame.wire_length
            delta['nets_routed'] = frame.nets_routed

            # Routing overlay arrays - ALWAYS include (even when empty) so JavaScript
            # clears previous state. Without this, old A* nodes/paths persist
            # when a frame legitimately has empty routing data (Issue #30).
            delta['traces'] = frame.traces
            delta['vias'] = frame.vias
            delta['astar_explored'] = frame.astar_explored
            delta['astar_frontier'] = frame.astar_frontier
            delta['astar_path'] = frame.astar_path
            delta['current_net'] = frame.current_net

            delta_frames.append(delta)

        return delta_frames

    def _generate_board_outline_svg(self, tx, ty, board_outline_color: str) -> str:
        """Generate SVG path for board outline, including polygon and holes.

        Args:
            tx: Transform function for X coordinate
            ty: Transform function for Y coordinate
            board_outline_color: Color for the outline stroke

        Returns:
            SVG element(s) for board outline
        """
        outline = self.board.outline

        # Check if we have a polygon outline
        if outline and hasattr(outline, 'polygon') and outline.polygon:
            # Build SVG path for polygon outline
            polygon = outline.polygon
            if len(polygon) >= 3:
                # Main outline path (clockwise for SVG fill-rule)
                path_data = f"M {tx(polygon[0][0])} {ty(polygon[0][1])}"
                for point in polygon[1:]:
                    path_data += f" L {tx(point[0])} {ty(point[1])}"
                path_data += " Z"  # Close path

                # Add holes/cutouts (counter-clockwise for fill-rule)
                if hasattr(outline, 'holes') and outline.holes:
                    for hole in outline.holes:
                        if len(hole) >= 3:
                            path_data += f" M {tx(hole[0][0])} {ty(hole[0][1])}"
                            for point in hole[1:]:
                                path_data += f" L {tx(point[0])} {ty(point[1])}"
                            path_data += " Z"

                return f'''<path d="{path_data}"
                      fill="none"
                      stroke="{board_outline_color}"
                      stroke-width="2"
                      fill-rule="evenodd"
                      class="board-outline"/>'''

        # Fall back to rectangle (bounding box)
        return f'''<rect x="{self.margin * self.scale}"
                  y="{self.margin * self.scale}"
                  width="{self.board_width * self.scale}"
                  height="{self.board_height * self.scale}"
                  fill="none"
                  stroke="{board_outline_color}"
                  stroke-width="2"
                  class="board-outline"/>'''

    def _generate_initial_svg(self) -> str:
        """Generate initial SVG with board outline and components at frame 0 positions."""
        from ..board.abstraction import Layer

        def tx(x):
            return (x - self.bounds[0] + self.margin) * self.scale

        def ty(y):
            return (y - self.bounds[1] + self.margin) * self.scale

        def ts(size):
            return size * self.scale

        # Get initial positions from frame 0 if available
        initial_positions = {}
        if self.frames:
            initial_positions = self.frames[0].changed_components

        # Generate component SVG elements
        component_svg = []
        pad_svg = []
        label_svg = []

        for ref, comp in self.board.components.items():
            # Use frame 0 position if available, otherwise use current board position
            if ref in initial_positions:
                x, y, rotation = initial_positions[ref]
            else:
                x, y, rotation = comp.x, comp.y, comp.rotation

            cx, cy = tx(x), ty(y)
            hw = ts(comp.width / 2)
            hh = ts(comp.height / 2)

            # Determine layer class
            layer_class = "comp-top"
            if comp.layer == Layer.BOTTOM_COPPER:
                layer_class = "comp-bottom"

            # Component body
            component_svg.append(
                f'<g class="component {layer_class}" data-ref="{ref}" '
                f'transform="rotate({-rotation} {cx} {cy})">'
                f'<rect x="{cx - hw}" y="{cy - hh}" '
                f'width="{ts(comp.width)}" height="{ts(comp.height)}" '
                f'fill="#2d5a3d" stroke="#1a1a1a" stroke-width="1" opacity="0.9"/>'
                f'</g>'
            )

            # Component pads - use data-pad-index for JavaScript to find them
            # Use pad's actual layer instead of component layer (Issue #37)
            for pad_idx, pad in enumerate(comp.pads):
                rad = math.radians(rotation)
                cos_r, sin_r = math.cos(rad), math.sin(rad)
                px = x + pad.x * cos_r - pad.y * sin_r
                py = y + pad.x * sin_r + pad.y * cos_r

                pad_cx, pad_cy = tx(px), ty(py)
                pad_hw = ts(pad.width / 2)
                pad_hh = ts(pad.height / 2)
                pad_rot = rotation + (getattr(pad, 'rotation', 0) or 0)

                # Determine pad layer class based on pad's actual layer (Issue #37)
                is_through_hole = pad.drill is not None and pad.drill > 0
                if is_through_hole:
                    # Through-hole pads span all layers - use special class
                    pad_layer_class = "pad-through-hole"
                    pad_color = "#b8860b"  # Goldenrod for through-hole
                elif pad.layer == Layer.TOP_COPPER:
                    pad_layer_class = "comp-top"
                    pad_color = "#4a9"
                elif pad.layer == Layer.BOTTOM_COPPER:
                    pad_layer_class = "comp-bottom"
                    pad_color = "#49a"
                else:
                    # Inner layer pads
                    pad_layer_class = "pad-inner"
                    pad_color = "#6a9"

                # Render pad based on shape (Issue #35)
                pad_shape = getattr(pad, 'shape', 'rect') or 'rect'

                common_attrs = (
                    f'class="pad-element {pad_layer_class}" data-ref="{ref}" data-pad-index="{pad_idx}" '
                    f'fill="{pad_color}" stroke="#1a1a1a" stroke-width="0.5" opacity="0.9"'
                )

                if pad_shape == "circle":
                    # Circle pad - use radius as half the larger dimension
                    pad_r = max(pad_hw, pad_hh)
                    pad_svg.append(
                        f'<circle {common_attrs} '
                        f'cx="{pad_cx}" cy="{pad_cy}" r="{pad_r}"/>'
                    )
                elif pad_shape == "oval":
                    # Oval pad - use ellipse element
                    pad_svg.append(
                        f'<ellipse {common_attrs} '
                        f'cx="{pad_cx}" cy="{pad_cy}" rx="{pad_hw}" ry="{pad_hh}" '
                        f'transform="rotate({-pad_rot} {pad_cx} {pad_cy})"/>'
                    )
                elif pad_shape == "roundrect":
                    # Rounded rectangle pad - use rect with rx/ry
                    # Corner radius is typically 25% of the smaller dimension
                    corner_radius = min(pad_hw, pad_hh) * 0.25
                    pad_svg.append(
                        f'<rect {common_attrs} '
                        f'x="{pad_cx - pad_hw}" y="{pad_cy - pad_hh}" '
                        f'width="{ts(pad.width)}" height="{ts(pad.height)}" '
                        f'rx="{corner_radius}" ry="{corner_radius}" '
                        f'transform="rotate({-pad_rot} {pad_cx} {pad_cy})"/>'
                    )
                else:
                    # Default: rectangular pad
                    pad_svg.append(
                        f'<rect {common_attrs} '
                        f'x="{pad_cx - pad_hw}" y="{pad_cy - pad_hh}" '
                        f'width="{ts(pad.width)}" height="{ts(pad.height)}" '
                        f'transform="rotate({-pad_rot} {pad_cx} {pad_cy})"/>'
                    )

                # Add drill hole for through-hole pads (Issue #35)
                if is_through_hole:
                    drill_r = ts(pad.drill / 2)
                    pad_svg.append(
                        f'<circle class="drill-hole {pad_layer_class}" data-ref="{ref}" data-pad-index="{pad_idx}" '
                        f'cx="{pad_cx}" cy="{pad_cy}" r="{drill_r}" '
                        f'fill="#1a1a2e" stroke="#333" stroke-width="0.3" opacity="1.0"/>'
                    )

            # Component label
            label_svg.append(
                f'<text class="ref-label {layer_class}" data-ref="{ref}" '
                f'x="{cx}" y="{cy}" '
                f'text-anchor="middle" dominant-baseline="middle" '
                f'font-size="8" fill="white" opacity="0.8">{ref}</text>'
            )

        components_group = '\n'.join(component_svg)
        pads_group = '\n'.join(pad_svg)
        labels_group = '\n'.join(label_svg)

        board_outline_color = get_color_manager().get_routing_color("board_outline")
        # Generate board outline (polygon or rectangle) - Issue #32
        board_outline_svg = self._generate_board_outline_svg(tx, ty, board_outline_color)

        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg"
             id="unified-svg"
             width="{self.svg_width}" height="{self.svg_height}"
             viewBox="0 0 {self.svg_width} {self.svg_height}">
            <rect width="100%" height="100%" fill="#1a1a2e"/>
            <!-- Board outline -->
            {board_outline_svg}
            <!-- Dynamic content groups (copper and debug layers) -->
            <g class="astar-debug-group"></g>
            <g class="ratsnest-group"></g>
            <g class="traces-group"></g>
            <g class="vias-group"></g>
            <!-- Component layers (rendered above copper for visibility) -->
            <g class="components-group">
                {components_group}
            </g>
            <g class="pads-group">
                {pads_group}
            </g>
            <!-- Labels (always on top) -->
            <g class="labels-group">
                {labels_group}
            </g>
        </svg>'''

        return svg_content

    def export_html(
        self,
        filename: str = "unified_visualization.html",
        output_dir: str = "placement_debug"
    ) -> Optional[Path]:
        """Export unified visualization as HTML file.

        Args:
            filename: Output filename
            output_dir: Output directory

        Returns:
            Path to generated HTML file, or None if no frames
        """
        if not self.frames:
            logger.warning("No frames to export")
            return None

        from . import get_svg_delta_viewer_js, get_styles_css
        from ..placement.viewer_template import generate_viewer_html_template

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Convert frames to delta format
        delta_frames = self._convert_frames_to_delta_format()

        # Calculate board bounds for JavaScript
        board_bounds = {
            'minX': self.bounds[0],
            'minY': self.bounds[1],
            'maxX': self.bounds[2],
            'maxY': self.bounds[3],
            'scale': self.scale,
            'padding': self.margin * self.scale
        }

        # Generate JavaScript with data
        js_code = get_svg_delta_viewer_js()
        data_js = f'''
// Board bounds for coordinate transforms
const boardBounds = {json.dumps(board_bounds)};

// Static properties for components
const staticProps = {json.dumps(self.static_props)};

// Delta frames (unified placement + routing)
const deltaFrames = {json.dumps(delta_frames)};

// Total frames
const totalFrames = {len(delta_frames)};

// Module colors
const moduleColors = {json.dumps(self.module_colors)};

// Component layers
const componentLayers = {json.dumps(self.component_layers)};

// Netlist for ratsnest
const netlist = {json.dumps(self.netlist)};

{js_code}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {{
    showFrame(0);
    drawEnergyGraph();
}});
'''

        # Generate initial SVG
        svg_content = self._generate_initial_svg()

        # Collect unique module types for legend
        module_types = set()
        for frame in self.frames:
            module_types.update(frame.changed_modules.values())

        # Generate HTML using shared template
        html = generate_viewer_html_template(
            title="Unified Visualization (Placement + Routing)",
            static_content='<div id="svg-container">' + svg_content + '</div>',
            dynamic_content='',
            javascript_code=data_js,
            module_types={m: self.module_colors.get(m, '#3498db') for m in module_types if m},
            total_frames=len(delta_frames),
            is_streaming=False
        )

        html_path = out_dir / filename
        html_path.write_text(html)
        logger.info(f"Exported unified visualization to {html_path}")
        return html_path


def create_unified_visualizer(board) -> UnifiedVisualizer:
    """Create a unified visualizer from a Board object.

    Args:
        board: Board instance from atoplace.board.abstraction

    Returns:
        UnifiedVisualizer configured for the board
    """
    return UnifiedVisualizer(board)
