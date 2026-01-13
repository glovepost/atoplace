"""
KiCad Board Adapter

Provides loading and saving of KiCad PCB files (.kicad_pcb) to/from
the unified Board abstraction.

Requires pcbnew (KiCad's Python API) to be available.
"""

from pathlib import Path
from typing import Optional, Dict, List
import re
import logging

from .abstraction import Board, Component, Net, Pad, Layer, BoardOutline, RefDesText

logger = logging.getLogger(__name__)


# Try to import pcbnew - may not be available in all environments
PCBNEW_AVAILABLE = False
_wx_app = None

try:
    # Suppress wx debug messages (e.g., "Adding duplicate image handler")
    # This must be done before importing wx or pcbnew
    import os
    os.environ.setdefault('WX_DEBUG', '0')

    # Also try to suppress via wx.Log if available
    try:
        import wx
        # Disable wx logging to suppress debug spam
        wx.Log.EnableLogging(False)

        # KiCad's Python may require wxApp to be initialized before certain
        # operations. If wx is available but no app is running, create a minimal one.
        # This prevents "create wxApp before calling this" errors.
        if not wx.App.Get():
            # Create a minimal wxApp for headless operation
            # Use redirect=False to avoid stdout/stderr redirection issues
            _wx_app = wx.App(redirect=False)

        # Re-enable logging but at a higher threshold to filter debug messages
        wx.Log.EnableLogging(True)
        wx.Log.SetLogLevel(wx.LOG_Warning)
    except (ImportError, RuntimeError, AttributeError, Exception):
        # wx not available or already initialized - that's fine
        pass

    import pcbnew
    PCBNEW_AVAILABLE = True

except ImportError:
    PCBNEW_AVAILABLE = False


def check_pcbnew():
    """Raise error if pcbnew is not available."""
    if not PCBNEW_AVAILABLE:
        raise ImportError(
            "pcbnew not available. Run with KiCad's Python interpreter:\n"
            "  /Applications/KiCad/KiCad.app/Contents/Frameworks/"
            "Python.framework/Versions/Current/bin/python3"
        )


def load_kicad_board(path: Path) -> Board:
    """
    Load a KiCad PCB file into a Board abstraction.

    Args:
        path: Path to .kicad_pcb file

    Returns:
        Board instance
    """
    check_pcbnew()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Board file not found: {path}")

    # Load the KiCad board
    kicad_board = pcbnew.LoadBoard(str(path))

    # Create our board abstraction
    board = Board(
        name=path.stem,
        source_file=path,
    )

    # Extract board outline
    board.outline = _extract_outline(kicad_board)

    # Extract layer count
    board.layer_count = kicad_board.GetCopperLayerCount()

    # Extract design rules
    ds = kicad_board.GetDesignSettings()
    board.default_trace_width = pcbnew.ToMM(ds.GetCurrentTrackWidth())
    board.default_clearance = pcbnew.ToMM(ds.GetSmallestClearanceValue())
    board.default_via_drill = pcbnew.ToMM(ds.GetCurrentViaDrill())

    # Extract components
    for fp in kicad_board.GetFootprints():
        component = _footprint_to_component(fp)
        board.add_component(component)

    # Build pad-to-net lookup table ONCE (O(footprints × pads))
    # This avoids scanning all pads for every net which was O(nets × footprints × pads)
    pad_net_map = {}  # net_name -> [(ref, pad_number), ...]
    for fp in kicad_board.GetFootprints():
        ref = str(fp.GetReference())
        for pad in fp.Pads():
            net_name = str(pad.GetNetname())
            if net_name:
                if net_name not in pad_net_map:
                    pad_net_map[net_name] = []
                pad_net_map[net_name].append((ref, pad.GetNumber()))

    # Extract nets using pre-built lookup
    for net_info in kicad_board.GetNetInfo().NetsByName().items():
        net_name, net_item = net_info
        if net_name:  # Skip empty net
            net = _extract_net(net_name, net_item, kicad_board, pad_net_map)
            board.add_net(net)

    return board


def save_kicad_board(board: Board, path: Path):
    """
    Save a Board abstraction to a KiCad PCB file.

    This updates component positions in an existing file rather than
    creating from scratch, preserving traces and other elements.

    Args:
        board: Board instance to save
        path: Path to .kicad_pcb file
    """
    check_pcbnew()
    import shutil

    path = Path(path)

    # If output path differs from source, copy source first to preserve all elements
    if board.source_file and board.source_file != path:
        if board.source_file.exists():
            # Copy source to destination to preserve traces, zones, etc.
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(board.source_file, path)

    # Load existing board if it exists
    if path.exists():
        kicad_board = pcbnew.LoadBoard(str(path))
    elif board.source_file and board.source_file.exists():
        # Load from source file if output doesn't exist but source does
        kicad_board = pcbnew.LoadBoard(str(board.source_file))
    else:
        # Create new board (basic setup)
        kicad_board = pcbnew.BOARD()
        # TODO: Set up layer stack, design rules, etc.

    # Update component positions
    import math
    for fp in kicad_board.GetFootprints():
        ref = fp.GetReference()
        if ref in board.components:
            comp = board.components[ref]

            # Convert centroid position back to KiCad origin position
            # The origin offset is in local (unrotated) coordinates
            offset_x = comp.origin_offset_x
            offset_y = comp.origin_offset_y

            if comp.rotation != 0 and (offset_x != 0 or offset_y != 0):
                rad = math.radians(comp.rotation)
                cos_r = math.cos(rad)
                sin_r = math.sin(rad)
                rotated_offset_x = offset_x * cos_r - offset_y * sin_r
                rotated_offset_y = offset_x * sin_r + offset_y * cos_r
            else:
                rotated_offset_x = offset_x
                rotated_offset_y = offset_y

            # Subtract offset to get KiCad origin from centroid
            kicad_x = comp.x - rotated_offset_x
            kicad_y = comp.y - rotated_offset_y

            # Update position
            fp.SetPosition(pcbnew.VECTOR2I(
                pcbnew.FromMM(kicad_x),
                pcbnew.FromMM(kicad_y)
            ))
            # Update rotation
            fp.SetOrientationDegrees(comp.rotation)
            # Update layer if changed
            if comp.layer == Layer.BOTTOM_COPPER:
                if fp.GetLayer() != pcbnew.B_Cu:
                    fp.Flip(fp.GetPosition(), False)
            elif comp.layer == Layer.TOP_COPPER:
                if fp.GetLayer() != pcbnew.F_Cu:
                    fp.Flip(fp.GetPosition(), False)

            # Update reference designator text position if available
            if comp.ref_des_text:
                _update_ref_des_text(fp, comp, kicad_x, kicad_y)

    # Update board outline on Edge.Cuts layer if it was generated/compacted
    if board.outline and board.outline.has_outline:
        _update_edge_cuts_outline(kicad_board, board.outline)

    # Save the board
    pcbnew.SaveBoard(str(path), kicad_board)


def _update_edge_cuts_outline(kicad_board, outline: BoardOutline):
    """Update or create Edge.Cuts outline on the KiCad board.

    Only creates/replaces the outline if:
    1. The outline was auto-generated (no original Edge.Cuts existed), OR
    2. The outline was explicitly marked for update

    This preserves user-drawn outlines while allowing auto-generated
    outlines to be written for boards that had none.

    Args:
        kicad_board: KiCad board object
        outline: BoardOutline with polygon to write
    """
    # Only update if this outline was auto-generated or compacted
    # This prevents overwriting carefully designed board shapes
    if not getattr(outline, 'auto_generated', False):
        return

    # Get outline polygon - either explicit polygon or rectangular
    if outline.polygon and len(outline.polygon) >= 3:
        polygon = outline.polygon
    else:
        # Create rectangle from dimensions
        polygon = [
            (outline.origin_x, outline.origin_y),
            (outline.origin_x + outline.width, outline.origin_y),
            (outline.origin_x + outline.width, outline.origin_y + outline.height),
            (outline.origin_x, outline.origin_y + outline.height),
        ]

    # Remove existing Edge.Cuts drawings (only lines/arcs, not complex shapes)
    # We only remove if we're auto-generating, which means there were none to begin with
    items_to_remove = []
    for drawing in kicad_board.GetDrawings():
        if drawing.GetLayer() == pcbnew.Edge_Cuts:
            # Only remove simple line segments (PCB_SHAPE with SEGMENT type)
            if hasattr(drawing, 'GetShape'):
                shape_type = drawing.GetShape()
                # pcbnew.S_SEGMENT = line, pcbnew.S_RECT = rectangle
                if shape_type in (pcbnew.S_SEGMENT, pcbnew.S_RECT):
                    items_to_remove.append(drawing)

    for item in items_to_remove:
        kicad_board.Remove(item)

    # Create new Edge.Cuts line segments for the polygon outline
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        # Create line segment
        line = pcbnew.PCB_SHAPE(kicad_board)
        line.SetShape(pcbnew.S_SEGMENT)
        line.SetLayer(pcbnew.Edge_Cuts)
        line.SetStart(pcbnew.VECTOR2I(
            pcbnew.FromMM(p1[0]),
            pcbnew.FromMM(p1[1])
        ))
        line.SetEnd(pcbnew.VECTOR2I(
            pcbnew.FromMM(p2[0]),
            pcbnew.FromMM(p2[1])
        ))
        # Standard Edge.Cuts line width (0.1mm)
        line.SetWidth(pcbnew.FromMM(0.1))
        kicad_board.Add(line)

    logger.debug(
        "Created Edge.Cuts outline: %d segments, bounds (%.1f, %.1f) to (%.1f, %.1f)",
        len(polygon),
        outline.origin_x, outline.origin_y,
        outline.origin_x + outline.width, outline.origin_y + outline.height
    )


def _update_ref_des_text(fp, comp: Component, kicad_x: float, kicad_y: float):
    """Update reference designator text position in KiCad footprint.

    Args:
        fp: KiCad footprint object
        comp: Component with ref_des_text data
        kicad_x, kicad_y: Footprint position in KiCad coordinates (origin, mm)
    """
    import math

    if not comp.ref_des_text:
        return

    ref_text = comp.ref_des_text

    # Get the reference field
    try:
        ref_field = fp.Reference()
    except AttributeError:
        try:
            ref_field = fp.GetField(0)
        except:
            return

    if not ref_field:
        return

    try:
        # Convert local offset to board coordinates
        # The offset is relative to centroid, but we need it relative to KiCad origin
        # Add back the centroid offset
        local_x = ref_text.offset_x + comp.origin_offset_x
        local_y = ref_text.offset_y + comp.origin_offset_y

        # Rotate to board coordinates
        rad = math.radians(comp.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        board_rel_x = local_x * cos_r - local_y * sin_r
        board_rel_y = local_x * sin_r + local_y * cos_r

        # Calculate absolute text position
        text_x = kicad_x + board_rel_x
        text_y = kicad_y + board_rel_y

        # Update text position
        ref_field.SetPosition(pcbnew.VECTOR2I(
            pcbnew.FromMM(text_x),
            pcbnew.FromMM(text_y)
        ))

        # Update text rotation
        text_abs_rotation = (ref_text.rotation + comp.rotation) % 360
        try:
            ref_field.SetTextAngleDegrees(text_abs_rotation)
        except AttributeError:
            try:
                ref_field.SetTextAngle(pcbnew.EDA_ANGLE(text_abs_rotation, pcbnew.DEGREES_T))
            except:
                pass  # Skip rotation update if method not available

        # Update text size if significantly different
        try:
            current_size = pcbnew.ToMM(ref_field.GetTextSize().y)
            if abs(current_size - ref_text.size) > 0.05:  # Only update if > 0.05mm difference
                ref_field.SetTextSize(pcbnew.VECTOR2I(
                    pcbnew.FromMM(ref_text.size),
                    pcbnew.FromMM(ref_text.size)
                ))
        except:
            pass

        # Update visibility
        try:
            if ref_field.IsVisible() != ref_text.visible:
                ref_field.SetVisible(ref_text.visible)
        except:
            pass

        # Update layer
        try:
            target_layer = pcbnew.B_SilkS if ref_text.layer == Layer.BOTTOM_SILK else pcbnew.F_SilkS
            if ref_field.GetLayer() != target_layer:
                ref_field.SetLayer(target_layer)
        except:
            pass

    except Exception as e:
        logger.debug(f"Failed to update ref des text for {comp.reference}: {e}")


def _extract_outline(kicad_board) -> BoardOutline:
    """Extract board outline from KiCad board.

    Attempts to extract the actual polygon outline from Edge.Cuts layer,
    falling back to bounding box for simple rectangular boards.

    Sets has_outline=False when no explicit Edge.Cuts outline was found,
    indicating that boundary validation should be skipped.
    """
    # Try to extract polygon outline from Edge.Cuts drawings
    polygon, holes = _extract_polygon_outline(kicad_board)

    if polygon and len(polygon) >= 3:
        # Calculate bounding box for the polygon
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        return BoardOutline(
            width=max(xs) - min(xs),
            height=max(ys) - min(ys),
            origin_x=min(xs),
            origin_y=min(ys),
            polygon=polygon,
            holes=holes,
            has_outline=True,
        )

    # Fall back to bounding box from edge cuts
    bbox = kicad_board.GetBoardEdgesBoundingBox()

    if bbox.GetWidth() > 0 and bbox.GetHeight() > 0:
        return BoardOutline(
            width=pcbnew.ToMM(bbox.GetWidth()),
            height=pcbnew.ToMM(bbox.GetHeight()),
            origin_x=pcbnew.ToMM(bbox.GetX()),
            origin_y=pcbnew.ToMM(bbox.GetY()),
            has_outline=True,
        )

    # Fall back to component bounding box - no explicit outline defined
    # Boundary checks should be skipped for this board
    bbox = kicad_board.ComputeBoundingBox()
    return BoardOutline(
        width=pcbnew.ToMM(bbox.GetWidth()),
        height=pcbnew.ToMM(bbox.GetHeight()),
        origin_x=pcbnew.ToMM(bbox.GetX()),
        origin_y=pcbnew.ToMM(bbox.GetY()),
        has_outline=False,
    )


def _extract_polygon_outline(kicad_board) -> tuple:
    """Extract polygon outline and holes from Edge.Cuts layer.

    Returns:
        Tuple of (main_polygon, holes) where main_polygon is a list of (x, y) tuples
        and holes is a list of polygon vertex lists.
    """
    import math

    # Collect all Edge.Cuts segments
    segments = []

    for drawing in kicad_board.GetDrawings():
        if drawing.GetLayer() != pcbnew.Edge_Cuts:
            continue

        # Handle different drawing types
        shape_type = drawing.GetClass()

        if shape_type == "PCB_SHAPE":
            shape = drawing.GetShape()

            if shape == pcbnew.SHAPE_T_SEGMENT:
                # Line segment
                start = drawing.GetStart()
                end = drawing.GetEnd()
                segments.append((
                    (pcbnew.ToMM(start.x), pcbnew.ToMM(start.y)),
                    (pcbnew.ToMM(end.x), pcbnew.ToMM(end.y))
                ))

            elif shape == pcbnew.SHAPE_T_ARC:
                # Arc - convert to line segments for simplicity
                center = drawing.GetCenter()
                start = drawing.GetStart()
                end = drawing.GetEnd()
                radius = pcbnew.ToMM(drawing.GetRadius())
                start_angle = drawing.GetArcAngleStart().AsDegrees()
                arc_angle = drawing.GetArcAngle().AsDegrees()

                # Discretize arc into line segments
                num_segments = max(8, int(abs(arc_angle) / 10))
                cx, cy = pcbnew.ToMM(center.x), pcbnew.ToMM(center.y)

                prev_point = None
                for i in range(num_segments + 1):
                    angle = math.radians(start_angle + arc_angle * i / num_segments)
                    px = cx + radius * math.cos(angle)
                    py = cy + radius * math.sin(angle)
                    point = (px, py)

                    if prev_point:
                        segments.append((prev_point, point))
                    prev_point = point

            elif shape == pcbnew.SHAPE_T_CIRCLE:
                # Circle - convert to polygon
                center = drawing.GetCenter()
                radius = pcbnew.ToMM(drawing.GetRadius())
                cx, cy = pcbnew.ToMM(center.x), pcbnew.ToMM(center.y)

                num_points = 32
                prev_point = None
                first_point = None
                for i in range(num_points + 1):
                    angle = 2 * math.pi * i / num_points
                    px = cx + radius * math.cos(angle)
                    py = cy + radius * math.sin(angle)
                    point = (px, py)

                    if first_point is None:
                        first_point = point
                    if prev_point:
                        segments.append((prev_point, point))
                    prev_point = point

            elif shape == pcbnew.SHAPE_T_RECT:
                # Rectangle
                start = drawing.GetStart()
                end = drawing.GetEnd()
                x1, y1 = pcbnew.ToMM(start.x), pcbnew.ToMM(start.y)
                x2, y2 = pcbnew.ToMM(end.x), pcbnew.ToMM(end.y)

                # Add four edges
                segments.append(((x1, y1), (x2, y1)))
                segments.append(((x2, y1), (x2, y2)))
                segments.append(((x2, y2), (x1, y2)))
                segments.append(((x1, y2), (x1, y1)))

            elif shape == pcbnew.SHAPE_T_POLY:
                # Polygon - extract vertices
                try:
                    poly_set = drawing.GetPolyShape()
                    for outline_idx in range(poly_set.OutlineCount()):
                        outline = poly_set.Outline(outline_idx)
                        points = []
                        for pt_idx in range(outline.PointCount()):
                            pt = outline.GetPoint(pt_idx)
                            points.append((pcbnew.ToMM(pt.x), pcbnew.ToMM(pt.y)))

                        # Add edges from polygon
                        for i in range(len(points)):
                            j = (i + 1) % len(points)
                            segments.append((points[i], points[j]))
                except:
                    pass

    if not segments:
        return None, []

    # Chain segments into polygons
    polygons = _chain_segments_to_polygons(segments)

    if not polygons:
        return None, []

    # Find the largest polygon (main outline) and treat others as holes
    main_polygon = max(polygons, key=lambda p: _polygon_area(p))
    holes = [p for p in polygons if p is not main_polygon]

    return main_polygon, holes


def _chain_segments_to_polygons(segments, tolerance=0.01):
    """Chain line segments into closed polygons.

    Args:
        segments: List of ((x1, y1), (x2, y2)) segment tuples
        tolerance: Distance tolerance for connecting segments (mm)

    Returns:
        List of polygon vertex lists
    """
    if not segments:
        return []

    # Copy segments so we can modify the list
    remaining = list(segments)
    polygons = []

    while remaining:
        # Start a new polygon with the first remaining segment
        current_polygon = list(remaining.pop(0))

        # Try to extend the polygon
        made_progress = True
        while made_progress:
            made_progress = False

            for i, seg in enumerate(remaining):
                start, end = seg

                # Check if segment connects to end of polygon
                if _points_close(current_polygon[-1], start, tolerance):
                    current_polygon.append(end)
                    remaining.pop(i)
                    made_progress = True
                    break
                elif _points_close(current_polygon[-1], end, tolerance):
                    current_polygon.append(start)
                    remaining.pop(i)
                    made_progress = True
                    break
                # Check if segment connects to start of polygon
                elif _points_close(current_polygon[0], end, tolerance):
                    current_polygon.insert(0, start)
                    remaining.pop(i)
                    made_progress = True
                    break
                elif _points_close(current_polygon[0], start, tolerance):
                    current_polygon.insert(0, end)
                    remaining.pop(i)
                    made_progress = True
                    break

        # Check if polygon is closed
        if len(current_polygon) >= 3:
            if _points_close(current_polygon[0], current_polygon[-1], tolerance):
                current_polygon.pop()  # Remove duplicate closing point
            polygons.append(current_polygon)

    return polygons


def _points_close(p1, p2, tolerance):
    """Check if two points are within tolerance distance."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) <= tolerance * tolerance


def _polygon_area(polygon):
    """Calculate area of polygon using shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2


def _footprint_to_component(fp) -> Component:
    """Convert KiCad footprint to Component.

    Note: KiCad's GetBoundingBox() returns the axis-aligned bounding box in
          board coordinates (already accounting for rotation). We need to
          estimate the original unrotated dimensions so that
          Component.get_bounding_box() can properly calculate the rotated box.

    The component position (x, y) is stored as the centroid of the pad extents,
    not the KiCad footprint origin. This ensures the placement algorithm works
    correctly with the bounding box. The origin offset is stored so we can
    correctly save back to KiCad.
    """
    import math

    ref = fp.GetReference()
    pos = fp.GetPosition()
    rotation = fp.GetOrientationDegrees()

    # Get bounding box for dimensions (axis-aligned in board coords)
    bbox = fp.GetBoundingBox()
    bbox_width = pcbnew.ToMM(bbox.GetWidth())
    bbox_height = pcbnew.ToMM(bbox.GetHeight())

    # Estimate original (unrotated) dimensions and centroid offset
    # The offset is the vector from KiCad origin to centroid in local coords
    width, height, offset_x, offset_y = _estimate_unrotated_dimensions(
        fp, bbox_width, bbox_height, rotation
    )

    # Determine layer
    layer = Layer.TOP_COPPER if fp.GetLayer() == pcbnew.F_Cu else Layer.BOTTOM_COPPER

    # Calculate centroid position in board coordinates
    # The offset is in local (unrotated) coordinates, so we need to rotate it
    kicad_x = pcbnew.ToMM(pos.x)
    kicad_y = pcbnew.ToMM(pos.y)

    if rotation != 0:
        rad = math.radians(rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        rotated_offset_x = offset_x * cos_r - offset_y * sin_r
        rotated_offset_y = offset_x * sin_r + offset_y * cos_r
    else:
        rotated_offset_x = offset_x
        rotated_offset_y = offset_y

    # Store centroid position as x, y (not KiCad origin)
    centroid_x = kicad_x + rotated_offset_x
    centroid_y = kicad_y + rotated_offset_y

    # Check DNP (Do Not Populate) flag - KiCad 9+
    dnp = False
    if hasattr(fp, 'IsDNP'):
        dnp = fp.IsDNP()
    elif hasattr(fp, 'IsExcludedFromBOM'):
        # Fallback for older KiCad versions
        dnp = fp.IsExcludedFromBOM()

    component = Component(
        reference=ref,
        footprint=fp.GetFPIDAsString(),
        value=fp.GetValue(),
        x=centroid_x,
        y=centroid_y,
        rotation=rotation,
        layer=layer,
        width=width,
        height=height,
        origin_offset_x=offset_x,  # Store for save-back
        origin_offset_y=offset_y,
        locked=fp.IsLocked(),
        dnp=dnp,
    )

    # Extract footprint properties (used by atopile for instance paths)
    _extract_footprint_properties(fp, component)

    # Extract reference designator text positioning
    component.ref_des_text = _extract_ref_des_text(fp, pos, rotation, offset_x, offset_y)

    # Extract pads (pass rotation and centroid offset for coordinate transformation)
    # The centroid offset is needed so pad positions are relative to the centroid
    # (which is what comp.x/y represents) rather than the KiCad origin
    for kicad_pad in fp.Pads():
        pad = _pad_to_pad(kicad_pad, pos, rotation, offset_x, offset_y)
        component.pads.append(pad)

    return component


def _extract_footprint_properties(fp, component: Component):
    """Extract custom properties from KiCad footprint.

    Atopile sets 'atopile_address' property containing the instance path
    (e.g., 'accel.c_bulk' for the bulk capacitor in the accel module).
    This is essential for mapping .ato module hierarchy to KiCad components.
    """
    try:
        # KiCad 9+: Use GetFields() to iterate over all fields
        # Standard fields are: Reference, Value, Datasheet, Description
        # Custom fields include: atopile_address, LCSC, Manufacturer, etc.
        if hasattr(fp, 'GetFields'):
            for field in fp.GetFields():
                name = str(field.GetName())
                # Skip standard fields that are already handled elsewhere
                if name in ('Reference', 'Value', 'Footprint', 'Datasheet', 'Description', ''):
                    continue
                # Use GetText() which is reliable across KiCad versions
                value = str(field.GetText())
                if value:
                    component.properties[name] = value

        # Direct lookup for atopile_address using GetFieldByName (KiCad 9+)
        if 'atopile_address' not in component.properties:
            if hasattr(fp, 'GetFieldByName'):
                field = fp.GetFieldByName('atopile_address')
                if field:
                    value = str(field.GetText())
                    if value:
                        component.properties['atopile_address'] = value
            # Fallback for older KiCad versions
            elif hasattr(fp, 'HasFieldByName') and fp.HasFieldByName('atopile_address'):
                if hasattr(fp, 'GetField'):
                    field = fp.GetField('atopile_address')
                    if field:
                        value = str(field.GetText())
                        if value:
                            component.properties['atopile_address'] = value

    except Exception as e:
        # Property extraction is non-critical - log and continue
        logger.debug(f"Failed to extract properties from {component.reference}: {e}")


def _extract_ref_des_text(fp, fp_pos, fp_rotation: float,
                          centroid_offset_x: float = 0.0,
                          centroid_offset_y: float = 0.0) -> RefDesText:
    """Extract reference designator text positioning from KiCad footprint.

    Args:
        fp: KiCad footprint object
        fp_pos: Footprint position in board coordinates (KiCad origin)
        fp_rotation: Footprint rotation in degrees
        centroid_offset_x: Offset from KiCad origin to centroid (local coords)
        centroid_offset_y: Offset from KiCad origin to centroid (local coords)

    Returns:
        RefDesText object with text positioning information

    Note: KiCad stores the reference field position relative to the footprint
          origin. We need to convert this to be relative to the component
          centroid (since that's what Component.x/y represents).
    """
    import math

    # Get the reference field (contains text position, size, etc.)
    try:
        ref_field = fp.Reference()
    except AttributeError:
        # Fallback for older KiCad versions
        try:
            ref_field = fp.GetField(0)  # Field 0 is typically Reference
        except:
            return RefDesText()  # Return defaults if we can't get the field

    if not ref_field:
        return RefDesText()

    try:
        # Get text position in board coordinates
        text_pos = ref_field.GetPosition()
        text_board_x = pcbnew.ToMM(text_pos.x)
        text_board_y = pcbnew.ToMM(text_pos.y)

        # Calculate position relative to footprint origin (board coordinates)
        fp_x = pcbnew.ToMM(fp_pos.x)
        fp_y = pcbnew.ToMM(fp_pos.y)
        board_rel_x = text_board_x - fp_x
        board_rel_y = text_board_y - fp_y

        # Reverse-rotate to get local footprint coordinates
        rad = math.radians(-fp_rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        local_x = board_rel_x * cos_r - board_rel_y * sin_r
        local_y = board_rel_x * sin_r + board_rel_y * cos_r

        # Adjust to be relative to centroid (not KiCad origin)
        offset_x = local_x - centroid_offset_x
        offset_y = local_y - centroid_offset_y

        # Get text rotation relative to footprint
        try:
            text_abs_rotation = ref_field.GetTextAngleDegrees()
        except AttributeError:
            try:
                text_abs_rotation = ref_field.GetTextAngle().AsDegrees()
            except:
                text_abs_rotation = 0.0

        text_local_rotation = (text_abs_rotation - fp_rotation) % 360

        # Get text size
        try:
            text_size = ref_field.GetTextSize()
            size = pcbnew.ToMM(text_size.y)  # Use height as size
        except:
            size = 1.0  # Default 1mm

        # Get text thickness
        try:
            thickness = pcbnew.ToMM(ref_field.GetTextThickness())
        except:
            thickness = 0.15  # Default

        # Get visibility
        try:
            visible = ref_field.IsVisible()
        except:
            visible = True

        # Determine layer
        try:
            field_layer = ref_field.GetLayer()
            if field_layer == pcbnew.B_SilkS:
                layer = Layer.BOTTOM_SILK
            else:
                layer = Layer.TOP_SILK
        except:
            layer = Layer.TOP_SILK

        return RefDesText(
            offset_x=offset_x,
            offset_y=offset_y,
            rotation=text_local_rotation,
            size=size,
            thickness=thickness,
            visible=visible,
            layer=layer,
        )

    except Exception as e:
        logger.debug(f"Failed to extract ref des text from {fp.GetReference()}: {e}")
        return RefDesText()  # Return defaults


def _estimate_unrotated_dimensions(fp, bbox_width: float, bbox_height: float,
                                   rotation: float) -> tuple:
    """Estimate unrotated component dimensions and centroid offset.

    Uses pad extents when available, otherwise reverse-calculates from
    the axis-aligned bounding box.

    Returns:
        (width, height, centroid_offset_x, centroid_offset_y)
        The centroid offset is the vector from the KiCad footprint origin
        to the actual pad-based centroid, in local (unrotated) coordinates.
    """
    import math

    # Try to get dimensions from pad extents (more reliable)
    pads = list(fp.Pads())
    if pads:
        # Calculate pad extents in local coordinates
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        fp_pos = fp.GetPosition()
        rad = math.radians(-rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)

        for pad in pads:
            pad_pos = pad.GetPosition()
            # Get position relative to footprint center
            rel_x = pcbnew.ToMM(pad_pos.x - fp_pos.x)
            rel_y = pcbnew.ToMM(pad_pos.y - fp_pos.y)
            # Reverse-rotate to get local coordinates
            local_x = rel_x * cos_r - rel_y * sin_r
            local_y = rel_x * sin_r + rel_y * cos_r

            pad_size = pad.GetSize()
            pad_w = pcbnew.ToMM(pad_size.x) / 2
            pad_h = pcbnew.ToMM(pad_size.y) / 2

            # Get pad rotation within footprint (relative to footprint)
            # Pad orientation is absolute, so we subtract footprint rotation
            try:
                pad_abs_rot = pad.GetOrientationDegrees()
            except AttributeError:
                try:
                    pad_abs_rot = pad.GetOrientation().AsDegrees()
                except:
                    pad_abs_rot = 0.0

            pad_local_rot = (pad_abs_rot - rotation) % 360
            pad_rad = math.radians(pad_local_rot)

            # Calculate axis-aligned bounding box half-extents of rotated pad
            cos_p = abs(math.cos(pad_rad))
            sin_p = abs(math.sin(pad_rad))
            pad_half_w = pad_w * cos_p + pad_h * sin_p
            pad_half_h = pad_w * sin_p + pad_h * cos_p

            min_x = min(min_x, local_x - pad_half_w)
            max_x = max(max_x, local_x + pad_half_w)
            min_y = min(min_y, local_y - pad_half_h)
            max_y = max(max_y, local_y + pad_half_h)

        if min_x != float('inf'):
            # Add margin for component body beyond pads
            # Use a proportional margin based on component size rather than fixed 0.5mm
            # This avoids over-inflating small components while still accounting for
            # component body extending beyond pad area on larger parts
            pad_width = max_x - min_x
            pad_height = max_y - min_y
            # 10% margin, minimum 0.1mm, maximum 0.5mm per side
            # Apply margin to BOTH sides (multiply by 2 for total width/height adjustment)
            margin = max(0.1, min(0.5, max(pad_width, pad_height) * 0.1))
            width = pad_width + 2 * margin  # margin on left + margin on right
            height = pad_height + 2 * margin  # margin on top + margin on bottom

            # Calculate centroid offset from footprint origin (in local coords)
            # This is the vector from KiCad's origin to the actual center of pads
            centroid_offset_x = (min_x + max_x) / 2
            centroid_offset_y = (min_y + max_y) / 2
            return (width, height, centroid_offset_x, centroid_offset_y)

    # Fallback: use bbox dimensions directly (may be slightly inflated for rotated parts)
    # For 0/90/180/270 rotations, swap if needed
    rot_normalized = rotation % 180
    if 45 < rot_normalized < 135:
        # Rotated ~90 degrees, swap width and height
        return (bbox_height, bbox_width, 0.0, 0.0)

    return (bbox_width, bbox_height, 0.0, 0.0)


def _map_pad_layer(kicad_pad) -> Layer:
    """Map KiCad pad layers to our Layer enum.

    Determines the primary layer for the pad based on its layer set.
    """
    # Check if pad is on bottom copper
    layer_set = kicad_pad.GetLayerSet()

    # Check for through-hole (has drill and on both copper layers)
    if kicad_pad.GetDrillSize().x > 0:
        # Through-hole pads - could be on both layers
        # Return TOP_COPPER as primary but the drill info indicates through-hole
        return Layer.TOP_COPPER

    # Check specific copper layers
    if layer_set.Contains(pcbnew.B_Cu) and not layer_set.Contains(pcbnew.F_Cu):
        return Layer.BOTTOM_COPPER

    # Default to top copper for SMD pads and most cases
    return Layer.TOP_COPPER


def _pad_to_pad(kicad_pad, fp_pos, fp_rotation: float,
                centroid_offset_x: float = 0.0, centroid_offset_y: float = 0.0) -> Pad:
    """Convert KiCad pad to Pad.

    Args:
        kicad_pad: KiCad pad object
        fp_pos: Footprint position in board coordinates (KiCad origin)
        fp_rotation: Footprint rotation in degrees
        centroid_offset_x: Offset from KiCad origin to centroid (local coords)
        centroid_offset_y: Offset from KiCad origin to centroid (local coords)

    Note: KiCad returns pad positions in board coordinates (already rotated).
          We need to reverse-rotate to get local footprint coordinates, since
          Pad.absolute_position() will re-apply the rotation.

          Pad rotation is also stored relative to the footprint, so we subtract
          the footprint rotation from the absolute pad rotation.

          The centroid offset is subtracted so that pad positions are relative
          to the component centroid (which is what Component.x/y represents),
          not the KiCad footprint origin. This ensures Pad.absolute_position()
          and Pad.get_bounding_box() work correctly with the centroid-based
          component position.
    """
    import math

    pad_pos = kicad_pad.GetPosition()
    size = kicad_pad.GetSize()

    # Calculate position relative to footprint center (board coordinates)
    board_rel_x = pcbnew.ToMM(pad_pos.x - fp_pos.x)
    board_rel_y = pcbnew.ToMM(pad_pos.y - fp_pos.y)

    # Reverse-rotate to get local footprint coordinates
    # This undoes the footprint rotation so Pad.absolute_position() can re-apply it
    rad = math.radians(-fp_rotation)  # Negative to reverse the rotation
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    local_x = board_rel_x * cos_r - board_rel_y * sin_r
    local_y = board_rel_x * sin_r + board_rel_y * cos_r

    # Adjust pad position to be relative to centroid, not KiCad origin
    # The centroid offset is in local (unrotated) coordinates
    rel_x = local_x - centroid_offset_x
    rel_y = local_y - centroid_offset_y

    # Get pad rotation relative to footprint
    # KiCad returns pad rotation in board coordinates, so subtract footprint rotation
    try:
        pad_abs_rotation = kicad_pad.GetOrientationDegrees()
    except AttributeError:
        # Older KiCad versions may use different method
        try:
            pad_abs_rotation = kicad_pad.GetOrientation() / 10.0  # Was in decidegrees
        except:
            pad_abs_rotation = 0.0

    pad_local_rotation = (pad_abs_rotation - fp_rotation) % 360

    # Determine shape
    shape_map = {
        pcbnew.PAD_SHAPE_RECT: "rect",
        pcbnew.PAD_SHAPE_CIRCLE: "circle",
        pcbnew.PAD_SHAPE_OVAL: "oval",
        pcbnew.PAD_SHAPE_ROUNDRECT: "roundrect",
        pcbnew.PAD_SHAPE_TRAPEZOID: "trapezoid",
        pcbnew.PAD_SHAPE_CUSTOM: "custom",
    }
    shape = shape_map.get(kicad_pad.GetShape(), "rect")

    # Get net name - convert to Python str to handle KiCad's wxString type
    # wxString doesn't work well with Board.nets (keyed by str), affecting module detection
    net_raw = kicad_pad.GetNetname()
    net = str(net_raw) if net_raw else None

    # Get drill size for through-hole pads
    drill = None
    if kicad_pad.GetDrillSize().x > 0:
        drill = pcbnew.ToMM(kicad_pad.GetDrillSize().x)

    # Map pad layer from KiCad
    layer = _map_pad_layer(kicad_pad)

    return Pad(
        number=kicad_pad.GetNumber(),
        x=rel_x,
        y=rel_y,
        width=pcbnew.ToMM(size.x),
        height=pcbnew.ToMM(size.y),
        shape=shape,
        layer=layer,
        net=net,
        drill=drill,
        rotation=pad_local_rotation,
    )


def _extract_net(net_name, net_item, kicad_board, pad_net_map: dict = None) -> Net:
    """Extract net information including net class rules.

    Args:
        net_name: Net name (may be str or KiCad wxString)
        net_item: KiCad net item object
        kicad_board: KiCad board object
        pad_net_map: Pre-built lookup table mapping net names to [(ref, pad_number), ...]
                     If provided, used for O(1) lookup instead of scanning all pads.
    """
    # Convert net_name to Python str to handle KiCad's wxString type
    # wxString doesn't have .upper() method, so we need to ensure it's a str
    net_name_str = str(net_name)

    net = Net(
        name=net_name_str,
        code=net_item.GetNetCode(),
    )

    # Extract net class information from KiCad design settings
    ds = kicad_board.GetDesignSettings()
    try:
        # Get the net class for this net
        net_class = net_item.GetNetClass()
        if net_class:
            net.net_class = str(net_class.GetName())
            # Extract trace width and clearance from net class
            net.trace_width = pcbnew.ToMM(net_class.GetTrackWidth())
            net.clearance = pcbnew.ToMM(net_class.GetClearance())
    except (AttributeError, RuntimeError):
        # Fallback for older KiCad versions or if net class not available
        # Use design settings defaults
        net.net_class = "Default"
        net.trace_width = pcbnew.ToMM(ds.GetCurrentTrackWidth())
        net.clearance = pcbnew.ToMM(ds.GetSmallestClearanceValue())

    # Determine if power/ground net
    name_upper = net_name_str.upper()
    if any(pwr in name_upper for pwr in ['VCC', 'VDD', 'V3V3', 'V5V', '3V3', '5V', 'VBAT', 'VIN']):
        net.is_power = True
    if any(gnd in name_upper for gnd in ['GND', 'VSS', 'GROUND', 'AGND', 'DGND']):
        net.is_ground = True

    # Check for differential pairs - mark both + and - nets
    # Positive nets
    if name_upper.endswith('+'):
        net.is_differential = True
        net.diff_pair_net = net_name_str[:-1] + '-'
    elif name_upper.endswith('_P'):
        net.is_differential = True
        net.diff_pair_net = net_name_str[:-2] + '_N'
    # Negative nets - also mark these as differential
    elif name_upper.endswith('-'):
        net.is_differential = True
        net.diff_pair_net = net_name_str[:-1] + '+'
    elif name_upper.endswith('_N'):
        net.is_differential = True
        net.diff_pair_net = net_name_str[:-2] + '_P'

    # Add pad connections - use pre-built lookup if available (O(1) per net)
    # Otherwise fall back to scanning all pads (O(footprints × pads) per net)
    if pad_net_map is not None:
        for ref, pad_number in pad_net_map.get(net_name_str, []):
            net.add_connection(ref, pad_number)
    else:
        # Legacy path - scan all footprints and pads
        for fp in kicad_board.GetFootprints():
            ref = str(fp.GetReference())
            for pad in fp.Pads():
                pad_net = str(pad.GetNetname())
                if pad_net == net_name_str:
                    net.add_connection(ref, pad.GetNumber())

    return net


def get_kicad_python_path() -> str:
    """Get the path to KiCad's Python interpreter."""
    import sys
    import os

    # Common KiCad Python paths
    paths = [
        # macOS
        "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3",
        # Linux
        "/usr/bin/python3",  # If kicad is installed system-wide
        # Windows
        r"C:\Program Files\KiCad\bin\python.exe",
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    # Fall back to current Python
    return sys.executable
