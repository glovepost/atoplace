"""Fanout pattern geometry generators.

Implements the geometric patterns for BGA escape routing:
- Dogbone: Standard for pitch >= 0.65mm. Trace + offset via.
- Via-in-Pad (VIP): Standard for pitch <= 0.5mm. Via directly in pad center.
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class Quadrant(Enum):
    """BGA quadrant for determining via offset direction."""
    NE = "northeast"  # Upper-right
    NW = "northwest"  # Upper-left
    SE = "southeast"  # Lower-right
    SW = "southwest"  # Lower-left


@dataclass
class FanoutVia:
    """Represents a via generated during fanout.

    Attributes:
        x: Via center X coordinate (mm)
        y: Via center Y coordinate (mm)
        drill_diameter: Via drill size (mm)
        pad_diameter: Via pad diameter (mm)
        start_layer: Starting copper layer index (0=top)
        end_layer: Ending copper layer index (1=bottom for 2-layer)
        net_name: Net name this via belongs to
        pad_number: Source pad number this via connects to
    """
    x: float
    y: float
    drill_diameter: float = 0.3
    pad_diameter: float = 0.6
    start_layer: int = 0
    end_layer: int = 1
    net_name: Optional[str] = None
    pad_number: Optional[str] = None


@dataclass
class FanoutTrace:
    """Represents a short trace from pad to via.

    Attributes:
        start: (x, y) coordinates of trace start (at pad)
        end: (x, y) coordinates of trace end (at via)
        width: Trace width (mm)
        layer: Routing layer index
        net_name: Net name this trace belongs to
    """
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float = 0.2
    layer: int = 0
    net_name: Optional[str] = None


def calculate_optimal_dogbone_offset(
    pad_pitch: float,
    pad_size: float,
    via_pad_diameter: float,
    trace_width: float,
    clearance: float,
) -> float:
    """Calculate optimal via offset for dogbone pattern.

    The via must be placed far enough from the pad to maintain clearance,
    but as close as possible to minimize trace length.

    Args:
        pad_pitch: Center-to-center distance between adjacent pads (mm)
        pad_size: Pad diameter/width (mm)
        via_pad_diameter: Via annular ring diameter (mm)
        trace_width: Width of connecting trace (mm)
        clearance: Required clearance between copper features (mm)

    Returns:
        Optimal offset distance from pad center to via center (mm)
    """
    # The via must clear adjacent pads
    # Minimum offset = half pad size + clearance + half via pad
    min_offset_for_pad_clearance = (
        pad_size / 2 + clearance + via_pad_diameter / 2
    )

    # The via should fit in the diagonal gap between pads
    # In a grid, the diagonal gap center is at pitch * sqrt(2) / 2 from pad
    diagonal_gap_center = pad_pitch * math.sqrt(2) / 2

    # Use the larger of the two constraints
    offset = max(min_offset_for_pad_clearance, diagonal_gap_center * 0.7)

    # Cap at reasonable maximum (don't go too far into the gap)
    max_offset = pad_pitch * 0.8
    return min(offset, max_offset)


def get_quadrant(
    pad_x: float,
    pad_y: float,
    component_center_x: float,
    component_center_y: float,
) -> Quadrant:
    """Determine which quadrant a pad is in relative to component center.

    BGA pads are divided into quadrants to determine via offset direction:
    - NE quadrant: via goes toward upper-right
    - NW quadrant: via goes toward upper-left
    - SE quadrant: via goes toward lower-right
    - SW quadrant: via goes toward lower-left
    """
    dx = pad_x - component_center_x
    dy = pad_y - component_center_y

    if dx >= 0 and dy <= 0:  # Y increases downward in PCB coords
        return Quadrant.NE
    elif dx < 0 and dy <= 0:
        return Quadrant.NW
    elif dx >= 0 and dy > 0:
        return Quadrant.SE
    else:
        return Quadrant.SW


def get_quadrant_offset(quadrant: Quadrant, offset_distance: float) -> Tuple[float, float]:
    """Get (dx, dy) offset for via placement based on quadrant.

    Returns the offset vector pointing diagonally outward from component center.
    """
    # Offset toward the corner (45 degrees)
    diag = offset_distance / math.sqrt(2)

    offsets = {
        Quadrant.NE: (diag, -diag),   # Right and up (negative Y is up)
        Quadrant.NW: (-diag, -diag),  # Left and up
        Quadrant.SE: (diag, diag),    # Right and down
        Quadrant.SW: (-diag, diag),   # Left and down
    }
    return offsets[quadrant]


@dataclass
class DogbonePattern:
    """Dogbone fanout pattern generator.

    Creates via + short trace for each BGA pad. The via is placed
    diagonally toward the corner, in the gap between pads.

    Standard for pitch >= 0.65mm where there's enough space for
    the via between pads.
    """
    # DFM parameters
    via_drill: float = 0.3
    via_pad: float = 0.6
    trace_width: float = 0.2
    clearance: float = 0.15

    def generate(
        self,
        pad_x: float,
        pad_y: float,
        pad_size: float,
        pad_pitch: float,
        component_center_x: float,
        component_center_y: float,
        layer: int = 0,
        net_name: Optional[str] = None,
        pad_number: Optional[str] = None,
    ) -> Tuple[FanoutVia, FanoutTrace]:
        """Generate dogbone pattern for a single pad.

        Args:
            pad_x, pad_y: Absolute pad center position
            pad_size: Pad diameter/width
            pad_pitch: Grid pitch of BGA
            component_center_x, component_center_y: Component center position
            layer: Starting layer for the trace
            net_name: Net name for the pad
            pad_number: Pad number identifier

        Returns:
            Tuple of (FanoutVia, FanoutTrace)
        """
        # Calculate optimal offset distance
        offset = calculate_optimal_dogbone_offset(
            pad_pitch=pad_pitch,
            pad_size=pad_size,
            via_pad_diameter=self.via_pad,
            trace_width=self.trace_width,
            clearance=self.clearance,
        )

        # Determine quadrant and get offset direction
        quadrant = get_quadrant(pad_x, pad_y, component_center_x, component_center_y)
        dx, dy = get_quadrant_offset(quadrant, offset)

        # Calculate via position
        via_x = pad_x + dx
        via_y = pad_y + dy

        # Create via
        via = FanoutVia(
            x=via_x,
            y=via_y,
            drill_diameter=self.via_drill,
            pad_diameter=self.via_pad,
            start_layer=layer,
            end_layer=layer + 1,  # Via to next layer
            net_name=net_name,
            pad_number=pad_number,
        )

        # Create connecting trace
        trace = FanoutTrace(
            start=(pad_x, pad_y),
            end=(via_x, via_y),
            width=self.trace_width,
            layer=layer,
            net_name=net_name,
        )

        return via, trace


@dataclass
class VIPPattern:
    """Via-in-Pad (VIP) fanout pattern generator.

    Places via directly in the pad center. Requires manufacturing
    support (via filling and planarization).

    Standard for fine-pitch BGAs (pitch <= 0.5mm) where there's
    no space between pads for dogbone vias.
    """
    # DFM parameters (typically smaller for fine pitch)
    via_drill: float = 0.2  # Microvia typical
    via_pad: float = 0.35   # Smaller annular ring

    def generate(
        self,
        pad_x: float,
        pad_y: float,
        layer: int = 0,
        target_layer: int = 1,
        net_name: Optional[str] = None,
        pad_number: Optional[str] = None,
    ) -> FanoutVia:
        """Generate VIP pattern for a single pad.

        Args:
            pad_x, pad_y: Absolute pad center position
            layer: Starting layer (where the pad is)
            target_layer: Layer to escape to
            net_name: Net name for the pad
            pad_number: Pad number identifier

        Returns:
            FanoutVia at pad center
        """
        return FanoutVia(
            x=pad_x,
            y=pad_y,
            drill_diameter=self.via_drill,
            pad_diameter=self.via_pad,
            start_layer=layer,
            end_layer=target_layer,
            net_name=net_name,
            pad_number=pad_number,
        )


@dataclass
class ChannelPattern:
    """Channel routing pattern for outer ring pins.

    Routes traces between rows of pins without vias, escaping
    to the edge of the BGA area. Only works for outer 1-2 rings
    where there's direct path to the outside.

    This is the most efficient pattern when possible, as it
    avoids using vias and inner layers.
    """
    trace_width: float = 0.2
    clearance: float = 0.15

    def can_channel_route(
        self,
        pad_x: float,
        pad_y: float,
        pad_pitch: float,
        ring_index: int,
        total_rings: int,
    ) -> bool:
        """Check if a pad can be channel routed (no via needed).

        Only outer rings can channel route - they have direct access
        to the routing field without needing to cross other pin rows.

        Args:
            pad_x, pad_y: Pad position
            pad_pitch: Grid pitch
            ring_index: Which ring the pad is in (0 = outermost)
            total_rings: Total number of rings in the BGA

        Returns:
            True if channel routing is possible
        """
        # Typically only ring 0 (outermost) can reliably channel route
        # Ring 1 might work with careful clearance checking
        return ring_index == 0

    def generate_escape_path(
        self,
        pad_x: float,
        pad_y: float,
        component_center_x: float,
        component_center_y: float,
        escape_distance: float,
        layer: int = 0,
        net_name: Optional[str] = None,
    ) -> List[FanoutTrace]:
        """Generate trace segments to escape from pad to clear routing space.

        Creates traces that route horizontally or vertically between
        pin rows to reach the BGA edge.

        Args:
            pad_x, pad_y: Pad position
            component_center_x, component_center_y: Component center
            escape_distance: Distance to route to (edge of BGA courtyard)
            layer: Routing layer
            net_name: Net name

        Returns:
            List of trace segments forming the escape path
        """
        traces = []

        # Determine primary escape direction (outward from center)
        dx = pad_x - component_center_x
        dy = pad_y - component_center_y

        # Escape in the direction away from center
        if abs(dx) > abs(dy):
            # Escape horizontally
            escape_x = pad_x + math.copysign(escape_distance, dx)
            escape_y = pad_y
        else:
            # Escape vertically
            escape_x = pad_x
            escape_y = pad_y + math.copysign(escape_distance, dy)

        traces.append(FanoutTrace(
            start=(pad_x, pad_y),
            end=(escape_x, escape_y),
            width=self.trace_width,
            layer=layer,
            net_name=net_name,
        ))

        return traces
