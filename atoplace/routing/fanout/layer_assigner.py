"""Layer assignment for BGA fanout using the "Onion" model.

The onion model assigns escape layers based on pin ring depth:
- Ring 0 (outermost): Route on top layer (no via needed for edge pins)
- Ring 1: Route on bottom layer (one via)
- Ring 2: Route on Inner Layer 1 (via to inner layer)
- Ring N: Route on deeper layers as needed

This maximizes routing efficiency by:
1. Using outer layers for easy-to-reach pins
2. Reserving inner layers for buried pins
3. Spreading routing load across available layers
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class PinRing:
    """Represents a ring of pins in a BGA array.

    A ring is defined by its distance from the center of the BGA.
    Ring 0 is the outermost ring.

    Attributes:
        index: Ring index (0 = outermost)
        pad_numbers: Set of pad numbers in this ring
        pin_count: Number of pins in this ring
    """
    index: int
    pad_numbers: Set[str] = field(default_factory=set)

    @property
    def pin_count(self) -> int:
        return len(self.pad_numbers)


@dataclass
class LayerMapping:
    """Maps pins to their escape layers.

    Attributes:
        pad_to_layer: Dict mapping pad number to escape layer index
        layer_to_pads: Dict mapping layer index to set of pad numbers
        ring_to_layer: Dict mapping ring index to layer index
    """
    pad_to_layer: Dict[str, int] = field(default_factory=dict)
    layer_to_pads: Dict[int, Set[str]] = field(default_factory=dict)
    ring_to_layer: Dict[int, int] = field(default_factory=dict)

    def get_escape_layer(self, pad_number: str) -> Optional[int]:
        """Get the assigned escape layer for a pad."""
        return self.pad_to_layer.get(pad_number)

    def get_pads_on_layer(self, layer: int) -> Set[str]:
        """Get all pads assigned to escape on a given layer."""
        return self.layer_to_pads.get(layer, set())


class LayerAssigner:
    """Assigns escape layers to BGA pins using the onion model.

    The onion model assigns layers based on ring depth:
    - Outer rings use outer layers (top/bottom)
    - Inner rings use inner layers

    This allows efficient escape routing with minimal layer transitions.
    """

    def __init__(self, layer_count: int = 2):
        """
        Args:
            layer_count: Total number of routing layers (2, 4, 6, 8...)
        """
        self.layer_count = layer_count
        self._signal_layers = self._get_signal_layer_order()

    def _get_signal_layer_order(self) -> List[int]:
        """Get signal layers in preferred routing order.

        Order: Top (0), Bottom (1), Inner1, Inner2, etc.
        This order prefers outer layers for easier manufacturing.

        Returns:
            List of layer indices in preferred order
        """
        if self.layer_count == 2:
            return [0, 1]
        elif self.layer_count == 4:
            return [0, 1, 2, 3]  # Top, Bottom, In1, In2
        elif self.layer_count == 6:
            return [0, 1, 2, 3, 4, 5]
        else:
            # General case: outer layers first
            return list(range(self.layer_count))

    def analyze_rings(
        self,
        pads: List[Tuple[str, float, float]],
        pitch: float,
        tolerance: float = 0.1,
    ) -> List[PinRing]:
        """Analyze BGA pad positions to identify concentric rings.

        BGA pins are organized in rings based on their distance from
        the center of the package. The outermost ring is ring 0.

        Args:
            pads: List of (pad_number, x, y) tuples with absolute positions
            pitch: BGA grid pitch (mm)
            tolerance: Position tolerance as fraction of pitch

        Returns:
            List of PinRing objects, sorted from outermost (0) to innermost
        """
        if not pads:
            return []

        # Calculate centroid
        cx = sum(p[1] for p in pads) / len(pads)
        cy = sum(p[2] for p in pads) / len(pads)

        # Calculate distance from center for each pad
        # Use Chebyshev (chessboard) distance for grid alignment
        pad_distances: Dict[str, float] = {}
        for pad_num, x, y in pads:
            # Chebyshev distance = max(|dx|, |dy|) in grid units
            dx = abs(x - cx) / pitch
            dy = abs(y - cy) / pitch
            dist = max(dx, dy)
            pad_distances[pad_num] = dist

        # Group pads into rings based on quantized distance
        rings_dict: Dict[int, Set[str]] = {}
        for pad_num, dist in pad_distances.items():
            # Round to nearest half-grid step
            ring_idx = round(dist * 2) / 2
            ring_idx = int(round(ring_idx))
            if ring_idx not in rings_dict:
                rings_dict[ring_idx] = set()
            rings_dict[ring_idx].add(pad_num)

        # Convert to PinRing objects
        # Invert so ring 0 is outermost
        max_ring = max(rings_dict.keys()) if rings_dict else 0
        rings = []
        for grid_dist in sorted(rings_dict.keys(), reverse=True):
            inverted_idx = max_ring - grid_dist
            rings.append(PinRing(
                index=inverted_idx,
                pad_numbers=rings_dict[grid_dist],
            ))

        # Sort by ring index (outermost first)
        rings.sort(key=lambda r: r.index)

        logger.debug(f"Identified {len(rings)} rings from {len(pads)} pads")
        for ring in rings:
            logger.debug(f"  Ring {ring.index}: {ring.pin_count} pins")

        return rings

    def assign_layers(
        self,
        rings: List[PinRing],
        strategy: str = "balanced",
    ) -> LayerMapping:
        """Assign escape layers to rings using the onion model.

        Args:
            rings: List of PinRing objects from analyze_rings()
            strategy: Assignment strategy:
                - "balanced": Spread load across layers evenly
                - "outer_first": Maximize use of outer layers
                - "minimum_vias": Minimize total via count

        Returns:
            LayerMapping with pad-to-layer assignments
        """
        mapping = LayerMapping()

        if not rings:
            return mapping

        num_rings = len(rings)
        num_layers = len(self._signal_layers)

        if strategy == "balanced":
            # Distribute rings across layers evenly
            rings_per_layer = max(1, math.ceil(num_rings / num_layers))
            for i, ring in enumerate(rings):
                layer_idx = min(i // rings_per_layer, num_layers - 1)
                layer = self._signal_layers[layer_idx]
                mapping.ring_to_layer[ring.index] = layer
                for pad in ring.pad_numbers:
                    mapping.pad_to_layer[pad] = layer
                    if layer not in mapping.layer_to_pads:
                        mapping.layer_to_pads[layer] = set()
                    mapping.layer_to_pads[layer].add(pad)

        elif strategy == "outer_first":
            # Use outer layers (top/bottom) for as many rings as possible
            # Only use inner layers for the innermost rings
            for i, ring in enumerate(rings):
                if i < num_layers:
                    layer = self._signal_layers[i]
                else:
                    # Beyond available layers, use innermost layer
                    layer = self._signal_layers[-1]
                mapping.ring_to_layer[ring.index] = layer
                for pad in ring.pad_numbers:
                    mapping.pad_to_layer[pad] = layer
                    if layer not in mapping.layer_to_pads:
                        mapping.layer_to_pads[layer] = set()
                    mapping.layer_to_pads[layer].add(pad)

        elif strategy == "minimum_vias":
            # Ring 0 (outermost) on top layer (no via needed for edge escape)
            # All other rings on bottom/inner layers
            for i, ring in enumerate(rings):
                if i == 0:
                    layer = 0  # Top layer
                else:
                    # Alternate between available layers
                    layer_idx = min(i, num_layers - 1)
                    layer = self._signal_layers[layer_idx]
                mapping.ring_to_layer[ring.index] = layer
                for pad in ring.pad_numbers:
                    mapping.pad_to_layer[pad] = layer
                    if layer not in mapping.layer_to_pads:
                        mapping.layer_to_pads[layer] = set()
                    mapping.layer_to_pads[layer].add(pad)

        # Log the assignment
        logger.debug("Layer assignment:")
        for layer, pads in sorted(mapping.layer_to_pads.items()):
            logger.debug(f"  Layer {layer}: {len(pads)} pads")

        return mapping

    def get_layer_for_ring(self, ring_index: int) -> int:
        """Get the default layer for a given ring index.

        Simple mapping based on available layers:
        - Ring 0 -> Layer 0 (Top)
        - Ring 1 -> Layer 1 (Bottom)
        - Ring 2 -> Layer 2 (Inner 1)
        - etc.

        Args:
            ring_index: Ring index (0 = outermost)

        Returns:
            Layer index for escape routing
        """
        if ring_index < len(self._signal_layers):
            return self._signal_layers[ring_index]
        # Fall back to innermost layer for deep rings
        return self._signal_layers[-1]


class StackupInfo:
    """Information about PCB layer stackup for fanout planning.

    Provides layer capabilities and constraints for fanout routing.
    """

    def __init__(
        self,
        layer_count: int = 2,
        via_types: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        """
        Args:
            layer_count: Total copper layer count
            via_types: Dict of via type name -> (start_layer, end_layer)
                      If None, assumes standard through-hole vias
        """
        self.layer_count = layer_count
        self.via_types = via_types or {
            "through": (0, layer_count - 1),
        }

        # Add blind/buried vias for 4+ layer boards
        if layer_count >= 4 and via_types is None:
            self.via_types["blind_top"] = (0, 1)
            self.via_types["blind_bottom"] = (layer_count - 2, layer_count - 1)
            if layer_count >= 6:
                self.via_types["buried"] = (1, layer_count - 2)

    def get_via_for_layers(
        self, start_layer: int, end_layer: int
    ) -> Optional[str]:
        """Find a via type that connects the given layers.

        Args:
            start_layer: Starting copper layer index
            end_layer: Ending copper layer index

        Returns:
            Via type name, or None if no suitable via exists
        """
        # First try exact match
        for via_type, (via_start, via_end) in self.via_types.items():
            if via_start == start_layer and via_end == end_layer:
                return via_type
            # Check reverse direction
            if via_start == end_layer and via_end == start_layer:
                return via_type

        # Try through-hole as fallback
        if "through" in self.via_types:
            return "through"

        return None

    def get_signal_layers(self) -> List[int]:
        """Get signal layer indices (excluding power planes if any)."""
        # For now, assume all layers are signal layers
        # In a real implementation, this would check for power plane assignments
        return list(range(self.layer_count))
