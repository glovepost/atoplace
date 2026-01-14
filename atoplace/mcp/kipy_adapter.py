"""
KiPy Adapter - Conversion between kipy and atoplace data structures.

This module provides utilities for:
- Unit conversion (kipy uses nanometers, atoplace uses mm)
- Board/Component/Net conversion from kipy to atoplace models
- Footprint lookup helpers

Requires: kicad-python (kipy) package for KiCad 9+ IPC API
"""

import logging
from typing import Optional, Dict, List, TYPE_CHECKING

from ..board.abstraction import Board, Component, Net, Pad, Layer, BoardOutline

if TYPE_CHECKING:
    # Avoid import errors when kipy is not installed
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Conversion
# =============================================================================

# Note: kipy provides from_mm/to_mm in kipy.util.units, but we provide
# fallbacks in case they're not available or for internal use


def mm_to_nm(mm: float) -> int:
    """Convert millimeters to nanometers (kipy internal units).

    Args:
        mm: Value in millimeters

    Returns:
        Value in nanometers (integer, rounded to avoid drift)
    """
    return round(mm * 1_000_000)


def nm_to_mm(nm: int) -> float:
    """Convert nanometers (kipy internal units) to millimeters.

    Args:
        nm: Value in nanometers

    Returns:
        Value in millimeters (float)
    """
    return nm / 1_000_000.0


def get_kipy_units():
    """Get kipy unit conversion functions if available.

    Returns:
        Tuple of (from_mm, to_mm) functions, or None if kipy not available
    """
    try:
        from kipy.util.units import from_mm, to_mm
        return from_mm, to_mm
    except ImportError:
        return None


# =============================================================================
# Board Conversion: kipy -> atoplace
# =============================================================================

def kipy_board_to_atoplace(kipy_board) -> Board:
    """
    Convert a kipy board to an atoplace Board model.

    Args:
        kipy_board: Board object from kipy's kicad.get_board()

    Returns:
        atoplace Board instance
    """
    board = Board(
        name=_get_board_name(kipy_board),
    )

    # Extract board outline/bounds
    board.outline = _extract_outline(kipy_board)

    # Extract layer count if available
    if hasattr(kipy_board, 'get_layer_count'):
        try:
            board.layer_count = kipy_board.get_layer_count()
        except Exception:
            board.layer_count = 2

    # Extract components (footprints)
    footprints = kipy_board.get_footprints()
    for fp in footprints:
        try:
            component = _kipy_footprint_to_component(fp)
            board.add_component(component)
        except Exception as e:
            logger.warning("Failed to convert footprint: %s", e)

    # Build net information from pad connections
    _extract_nets(kipy_board, board)

    logger.info("Converted kipy board: %d components, %d nets",
                len(board.components), len(board.nets))

    return board


def _get_board_name(kipy_board) -> str:
    """Extract board name from kipy board."""
    try:
        # Try file_path first
        if hasattr(kipy_board, 'file_path') and kipy_board.file_path:
            from pathlib import Path
            return Path(kipy_board.file_path).stem
        # Try filename
        if hasattr(kipy_board, 'filename') and kipy_board.filename:
            from pathlib import Path
            return Path(kipy_board.filename).stem
    except Exception as e:
        logger.debug("Could not extract board name: %s", e)
    return "board"


def _extract_outline(kipy_board) -> BoardOutline:
    """Extract board outline from kipy board."""
    try:
        # kipy may expose bounding box method
        if hasattr(kipy_board, 'get_bounding_box'):
            bbox = kipy_board.get_bounding_box()
            # bbox may be a Box2 object with x, y, width, height
            if hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                return BoardOutline(
                    width=nm_to_mm(bbox.width),
                    height=nm_to_mm(bbox.height),
                    origin_x=nm_to_mm(bbox.x) if hasattr(bbox, 'x') else 0,
                    origin_y=nm_to_mm(bbox.y) if hasattr(bbox, 'y') else 0,
                    has_outline=True,
                )
            # Or it might be (min_x, min_y, max_x, max_y)
            elif hasattr(bbox, '__iter__'):
                coords = list(bbox)
                if len(coords) >= 4:
                    min_x, min_y, max_x, max_y = coords[:4]
                    return BoardOutline(
                        width=nm_to_mm(max_x - min_x),
                        height=nm_to_mm(max_y - min_y),
                        origin_x=nm_to_mm(min_x),
                        origin_y=nm_to_mm(min_y),
                        has_outline=True,
                    )
    except Exception as e:
        logger.warning("Could not extract outline: %s", e)

    # Return default outline
    return BoardOutline(has_outline=False)


def _kipy_footprint_to_component(fp) -> Component:
    """Convert a kipy footprint to an atoplace Component."""
    # Extract reference designator
    ref = _get_footprint_reference(fp)

    # Extract position (in nanometers)
    pos = fp.position
    x_mm = nm_to_mm(pos.x) if hasattr(pos, 'x') else 0
    y_mm = nm_to_mm(pos.y) if hasattr(pos, 'y') else 0

    # Extract rotation
    rotation = 0.0
    if hasattr(fp, 'orientation'):
        orient = fp.orientation
        if hasattr(orient, 'degrees'):
            rotation = orient.degrees
        elif hasattr(orient, 'as_degrees'):
            rotation = orient.as_degrees()
        elif isinstance(orient, (int, float)):
            rotation = float(orient)

    # Determine layer
    layer = _get_component_layer(fp)

    # Create component
    component = Component(
        reference=ref,
        footprint=_get_footprint_name(fp),
        value=_get_footprint_value(fp),
        x=x_mm,
        y=y_mm,
        rotation=rotation,
        layer=layer,
        locked=getattr(fp, 'locked', False),
    )

    # Extract pads
    if hasattr(fp, 'pads'):
        for kipy_pad in fp.pads:
            try:
                pad = _kipy_pad_to_pad(kipy_pad, fp)
                component.pads.append(pad)
            except Exception as e:
                logger.debug("Failed to convert pad: %s", e)

    # Calculate component dimensions from pads
    _calculate_component_dimensions(component)

    return component


def _get_footprint_reference(fp) -> str:
    """Get reference designator from kipy footprint."""
    # Try reference_field.text.value (kipy pattern)
    if hasattr(fp, 'reference_field'):
        ref_field = fp.reference_field
        if hasattr(ref_field, 'text') and hasattr(ref_field.text, 'value'):
            return str(ref_field.text.value)
    # Try reference property
    if hasattr(fp, 'reference'):
        ref = fp.reference
        if callable(ref):
            return str(ref())
        return str(ref)
    # Try GetReference (pcbnew style)
    if hasattr(fp, 'GetReference'):
        return str(fp.GetReference())
    return "??"


def _get_footprint_name(fp) -> str:
    """Get footprint library:name from kipy footprint."""
    if hasattr(fp, 'footprint_name'):
        return str(fp.footprint_name)
    if hasattr(fp, 'lib_id'):
        return str(fp.lib_id)
    if hasattr(fp, 'GetFPID'):
        return str(fp.GetFPID().GetUniStringLibId())
    return ""


def _get_footprint_value(fp) -> str:
    """Get value from kipy footprint."""
    # Try value_field.text.value
    if hasattr(fp, 'value_field'):
        val_field = fp.value_field
        if hasattr(val_field, 'text') and hasattr(val_field.text, 'value'):
            return str(val_field.text.value)
    # Try value property
    if hasattr(fp, 'value'):
        val = fp.value
        if callable(val):
            return str(val())
        return str(val)
    return ""


def _get_component_layer(fp) -> Layer:
    """Determine component layer from kipy footprint."""
    layer = Layer.TOP_COPPER

    if hasattr(fp, 'layer'):
        layer_val = fp.layer
        layer_str = str(layer_val)

        if 'B.Cu' in layer_str or 'Bottom' in layer_str or 'BL_B_Cu' in layer_str:
            layer = Layer.BOTTOM_COPPER

    return layer


def _kipy_pad_to_pad(kipy_pad, fp) -> Pad:
    """Convert a kipy pad to an atoplace Pad."""
    # Get pad position (absolute in kipy)
    pad_pos = kipy_pad.position
    fp_pos = fp.position

    # Calculate relative position to footprint center
    rel_x = nm_to_mm(pad_pos.x - fp_pos.x) if hasattr(pad_pos, 'x') else 0
    rel_y = nm_to_mm(pad_pos.y - fp_pos.y) if hasattr(pad_pos, 'y') else 0

    # Get pad size
    width = 0.5
    height = 0.5
    if hasattr(kipy_pad, 'size'):
        size = kipy_pad.size
        if hasattr(size, 'x') and hasattr(size, 'y'):
            width = nm_to_mm(size.x)
            height = nm_to_mm(size.y)

    # Get net name
    net = None
    if hasattr(kipy_pad, 'net') and kipy_pad.net:
        if hasattr(kipy_pad.net, 'name'):
            net = str(kipy_pad.net.name)
        else:
            net = str(kipy_pad.net)

    # Get pad number
    number = "1"
    if hasattr(kipy_pad, 'number'):
        number = str(kipy_pad.number)
    elif hasattr(kipy_pad, 'name'):
        number = str(kipy_pad.name)

    # Get pad shape
    shape = "rect"
    if hasattr(kipy_pad, 'shape'):
        shape_val = str(kipy_pad.shape).lower()
        if 'circle' in shape_val:
            shape = "circle"
        elif 'oval' in shape_val:
            shape = "oval"
        elif 'roundrect' in shape_val:
            shape = "roundrect"

    # Get drill size (for through-hole)
    drill = None
    if hasattr(kipy_pad, 'drill') and kipy_pad.drill:
        drill_val = kipy_pad.drill
        if hasattr(drill_val, 'x'):
            drill = nm_to_mm(drill_val.x)
        elif isinstance(drill_val, (int, float)):
            drill = nm_to_mm(drill_val)

    return Pad(
        number=number,
        x=rel_x,
        y=rel_y,
        width=width,
        height=height,
        net=net,
        shape=shape,
        drill=drill,
    )


def _calculate_component_dimensions(component: Component):
    """Calculate component width/height from pad extents."""
    if not component.pads:
        component.width = 1.0
        component.height = 1.0
        return

    min_x = min(p.x - p.width / 2 for p in component.pads)
    max_x = max(p.x + p.width / 2 for p in component.pads)
    min_y = min(p.y - p.height / 2 for p in component.pads)
    max_y = max(p.y + p.height / 2 for p in component.pads)

    # Add small margin for component body
    margin = 0.2
    component.width = (max_x - min_x) + margin * 2
    component.height = (max_y - min_y) + margin * 2


def _extract_nets(kipy_board, board: Board):
    """Extract net information from kipy board and component pads."""
    # Build net lookup from pad connections
    for ref, comp in board.components.items():
        for pad in comp.pads:
            if pad.net:
                if pad.net not in board.nets:
                    net = Net(name=pad.net)
                    _classify_net(net)
                    board.add_net(net)
                board.nets[pad.net].add_connection(ref, pad.number)


def _classify_net(net: Net):
    """Classify net as power/ground based on name patterns."""
    name_upper = net.name.upper()

    # Power net patterns
    power_patterns = ['VCC', 'VDD', 'V3V3', 'V5V', '3V3', '5V', 'VBAT',
                      'VIN', 'VBUS', '+3V3', '+5V', '+12V', 'VREF']
    if any(pwr in name_upper for pwr in power_patterns):
        net.is_power = True

    # Ground net patterns
    ground_patterns = ['GND', 'VSS', 'GROUND', 'AGND', 'DGND', 'PGND', 'SGND']
    if any(gnd in name_upper for gnd in ground_patterns):
        net.is_ground = True


# =============================================================================
# Footprint Lookup
# =============================================================================

def find_kipy_footprint(kipy_board, ref: str):
    """
    Find a footprint in kipy board by reference designator.

    Args:
        kipy_board: kipy board object
        ref: Reference designator to find (e.g., "C6", "U1")

    Returns:
        kipy footprint object, or None if not found
    """
    for fp in kipy_board.get_footprints():
        fp_ref = _get_footprint_reference(fp)
        if fp_ref == ref:
            return fp
    return None


def find_kipy_footprints(kipy_board, refs: List[str]) -> Dict[str, any]:
    """
    Find multiple footprints in kipy board by reference designators.

    Args:
        kipy_board: kipy board object
        refs: List of reference designators to find

    Returns:
        Dict mapping ref -> kipy footprint (missing refs not included)
    """
    ref_set = set(refs)
    result = {}

    for fp in kipy_board.get_footprints():
        fp_ref = _get_footprint_reference(fp)
        if fp_ref in ref_set:
            result[fp_ref] = fp
            if len(result) == len(refs):
                break  # Found all

    return result


# =============================================================================
# Sync atoplace -> kipy (for updates)
# =============================================================================

def update_kipy_footprint_position(fp, x_mm: float, y_mm: float,
                                    rotation_deg: Optional[float] = None):
    """
    Update a kipy footprint's position and optionally rotation.

    This prepares the footprint for board.update_items() - caller must
    handle the commit transaction.

    Args:
        fp: kipy footprint object
        x_mm: New X position in millimeters
        y_mm: New Y position in millimeters
        rotation_deg: New rotation in degrees (optional)
    """
    try:
        from kipy.geometry import Vector2, Angle
        from kipy.util.units import from_mm
    except ImportError:
        # Fallback without kipy.util.units
        from kipy.geometry import Vector2, Angle
        from_mm = mm_to_nm

    # Update position
    fp.position = Vector2.from_xy(from_mm(x_mm), from_mm(y_mm))

    # Update rotation if provided
    if rotation_deg is not None:
        fp.orientation = Angle.from_degrees(rotation_deg)
