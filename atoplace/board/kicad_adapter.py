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

    path = Path(path)

    # Load existing board if it exists
    if path.exists():
        kicad_board = pcbnew.LoadBoard(str(path))
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
    """Convert KiCad footprint to Component."""
    ref = fp.GetReference()
    pos = fp.GetPosition()

    # Get bounding box for dimensions
    bbox = fp.GetBoundingBox()

    # Determine layer
    layer = Layer.TOP_COPPER if fp.GetLayer() == pcbnew.F_Cu else Layer.BOTTOM_COPPER

    component = Component(
        reference=ref,
        footprint=fp.GetFPIDAsString(),
        value=fp.GetValue(),
        x=pcbnew.ToMM(pos.x),
        y=pcbnew.ToMM(pos.y),
        rotation=fp.GetOrientationDegrees(),
        layer=layer,
        width=pcbnew.ToMM(bbox.GetWidth()),
        height=pcbnew.ToMM(bbox.GetHeight()),
        locked=fp.IsLocked(),
    )

    # Extract pads
    for kicad_pad in fp.Pads():
        pad = _pad_to_pad(kicad_pad, pos)
        component.pads.append(pad)

    return component


def _pad_to_pad(kicad_pad, fp_pos) -> Pad:
    """Convert KiCad pad to Pad."""
    pad_pos = kicad_pad.GetPosition()
    size = kicad_pad.GetSize()

    # Calculate relative position
    rel_x = pcbnew.ToMM(pad_pos.x - fp_pos.x)
    rel_y = pcbnew.ToMM(pad_pos.y - fp_pos.y)

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

    return Pad(
        number=kicad_pad.GetNumber(),
        x=rel_x,
        y=rel_y,
        width=pcbnew.ToMM(size.x),
        height=pcbnew.ToMM(size.y),
        shape=shape,
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
