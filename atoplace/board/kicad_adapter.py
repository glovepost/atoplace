"""
KiCad Board Adapter

Provides loading and saving of KiCad PCB files (.kicad_pcb) to/from
the unified Board abstraction.

Requires pcbnew (KiCad's Python API) to be available.
"""

from pathlib import Path
from typing import Optional, Dict, List
import re

from .abstraction import Board, Component, Net, Pad, Layer, BoardOutline


# Try to import pcbnew - may not be available in all environments
try:
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

    # Extract nets
    for net_info in kicad_board.GetNetInfo().NetsByName().items():
        net_name, net_item = net_info
        if net_name:  # Skip empty net
            net = _extract_net(net_name, net_item, kicad_board)
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
    for fp in kicad_board.GetFootprints():
        ref = fp.GetReference()
        if ref in board.components:
            comp = board.components[ref]
            # Update position
            fp.SetPosition(pcbnew.VECTOR2I(
                pcbnew.FromMM(comp.x),
                pcbnew.FromMM(comp.y)
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

    # Save the board
    pcbnew.SaveBoard(str(path), kicad_board)


def _extract_outline(kicad_board) -> BoardOutline:
    """Extract board outline from KiCad board."""
    # Try to get bounding box from edge cuts
    bbox = kicad_board.GetBoardEdgesBoundingBox()

    if bbox.GetWidth() > 0 and bbox.GetHeight() > 0:
        return BoardOutline(
            width=pcbnew.ToMM(bbox.GetWidth()),
            height=pcbnew.ToMM(bbox.GetHeight()),
            origin_x=pcbnew.ToMM(bbox.GetX()),
            origin_y=pcbnew.ToMM(bbox.GetY()),
        )

    # Fall back to component bounding box
    bbox = kicad_board.ComputeBoundingBox()
    return BoardOutline(
        width=pcbnew.ToMM(bbox.GetWidth()),
        height=pcbnew.ToMM(bbox.GetHeight()),
        origin_x=pcbnew.ToMM(bbox.GetX()),
        origin_y=pcbnew.ToMM(bbox.GetY()),
    )


def _footprint_to_component(fp) -> Component:
    """Convert KiCad footprint to Component.

    Note: KiCad's GetBoundingBox() returns the axis-aligned bounding box in
          board coordinates (already accounting for rotation). We need to
          estimate the original unrotated dimensions so that
          Component.get_bounding_box() can properly calculate the rotated box.
    """
    import math

    ref = fp.GetReference()
    pos = fp.GetPosition()
    rotation = fp.GetOrientationDegrees()

    # Get bounding box for dimensions (axis-aligned in board coords)
    bbox = fp.GetBoundingBox()
    bbox_width = pcbnew.ToMM(bbox.GetWidth())
    bbox_height = pcbnew.ToMM(bbox.GetHeight())

    # Estimate original (unrotated) dimensions by reverse-calculating
    # For rotated rectangles, the axis-aligned bbox expands
    # We use the pad extents as a more reliable measure when available
    width, height = _estimate_unrotated_dimensions(fp, bbox_width, bbox_height, rotation)

    # Determine layer
    layer = Layer.TOP_COPPER if fp.GetLayer() == pcbnew.F_Cu else Layer.BOTTOM_COPPER

    component = Component(
        reference=ref,
        footprint=fp.GetFPIDAsString(),
        value=fp.GetValue(),
        x=pcbnew.ToMM(pos.x),
        y=pcbnew.ToMM(pos.y),
        rotation=rotation,
        layer=layer,
        width=width,
        height=height,
        locked=fp.IsLocked(),
    )

    # Extract pads (pass rotation for coordinate transformation)
    for kicad_pad in fp.Pads():
        pad = _pad_to_pad(kicad_pad, pos, rotation)
        component.pads.append(pad)

    return component


def _estimate_unrotated_dimensions(fp, bbox_width: float, bbox_height: float,
                                   rotation: float) -> tuple:
    """Estimate unrotated component dimensions.

    Uses pad extents when available, otherwise reverse-calculates from
    the axis-aligned bounding box.
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

            min_x = min(min_x, local_x - pad_w)
            max_x = max(max_x, local_x + pad_w)
            min_y = min(min_y, local_y - pad_h)
            max_y = max(max_y, local_y + pad_h)

        if min_x != float('inf'):
            # Add margin for component body beyond pads
            margin = 0.5  # mm typical component body margin
            width = (max_x - min_x) + margin
            height = (max_y - min_y) + margin
            return (width, height)

    # Fallback: use bbox dimensions directly (may be slightly inflated for rotated parts)
    # For 0/90/180/270 rotations, swap if needed
    rot_normalized = rotation % 180
    if 45 < rot_normalized < 135:
        # Rotated ~90 degrees, swap width and height
        return (bbox_height, bbox_width)

    return (bbox_width, bbox_height)


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


def _pad_to_pad(kicad_pad, fp_pos, fp_rotation: float) -> Pad:
    """Convert KiCad pad to Pad.

    Args:
        kicad_pad: KiCad pad object
        fp_pos: Footprint position in board coordinates
        fp_rotation: Footprint rotation in degrees

    Note: KiCad returns pad positions in board coordinates (already rotated).
          We need to reverse-rotate to get local footprint coordinates, since
          Pad.absolute_position() will re-apply the rotation.
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
    rel_x = board_rel_x * cos_r - board_rel_y * sin_r
    rel_y = board_rel_x * sin_r + board_rel_y * cos_r

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

    # Get net name
    net = kicad_pad.GetNetname() or None

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
    )


def _extract_net(net_name: str, net_item, kicad_board) -> Net:
    """Extract net information."""
    net = Net(
        name=net_name,
        code=net_item.GetNetCode(),
    )

    # Determine if power/ground net
    name_upper = net_name.upper()
    if any(pwr in name_upper for pwr in ['VCC', 'VDD', 'V3V3', 'V5V', '3V3', '5V', 'VBAT', 'VIN']):
        net.is_power = True
    if any(gnd in name_upper for gnd in ['GND', 'VSS', 'GROUND', 'AGND', 'DGND']):
        net.is_ground = True

    # Check for differential pairs
    if name_upper.endswith('+') or name_upper.endswith('_P'):
        net.is_differential = True
        # Try to find matching pair
        if name_upper.endswith('+'):
            net.diff_pair_net = net_name[:-1] + '-'
        elif name_upper.endswith('_P'):
            net.diff_pair_net = net_name[:-2] + '_N'

    # Find all pads connected to this net
    for fp in kicad_board.GetFootprints():
        ref = fp.GetReference()
        for pad in fp.Pads():
            if pad.GetNetname() == net_name:
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
