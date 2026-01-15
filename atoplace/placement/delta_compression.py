"""Delta compression for visualization frames.

Reduces memory usage and HTML file size by storing only changes between frames
instead of complete state snapshots.

Key optimization: Most components don't move between iterations, so we only
store position/rotation deltas for components that actually changed.

Memory savings: ~90% for typical placement runs where 10-20% of components
move each iteration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComponentDelta:
    """Position/rotation change for a single component between frames.

    Only stores fields that changed. If a field is None, it hasn't changed.
    """
    x: Optional[float] = None
    y: Optional[float] = None
    rotation: Optional[float] = None

    def has_changes(self) -> bool:
        """Check if this delta contains any actual changes."""
        return self.x is not None or self.y is not None or self.rotation is not None


@dataclass
class FrameDelta:
    """Delta-compressed frame storing only changes from previous frame.

    Structure:
    - index: Frame number
    - label: Frame description
    - changed_components: Only components that moved
    - forces: Force vectors (always stored, they change every frame)
    - overlaps: Overlap pairs (usually sparse)
    - stats: Energy, wire length, etc.

    To reconstruct full frame state, apply this delta to the previous frame.
    """
    index: int
    label: str
    iteration: int = 0
    phase: str = ""

    # Delta data: Only changed components
    # ref -> ComponentDelta with only changed fields set
    changed_components: Dict[str, ComponentDelta] = field(default_factory=dict)

    # These fields change frequently, so always store them
    forces: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)
    overlaps: List[Tuple[str, str]] = field(default_factory=list)

    # Module assignments (only store if changed)
    changed_modules: Dict[str, str] = field(default_factory=dict)

    # Stats always stored (small, change every frame)
    energy: float = 0.0
    max_move: float = 0.0
    overlap_count: int = 0
    total_wire_length: float = 0.0

    # Movement tracking
    movement: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    # Net connections (static, only in frame 0)
    connections: List[Tuple[str, str, str]] = field(default_factory=list)


class DeltaCompressor:
    """Compresses visualization frames using delta encoding.

    Maintains state of the previous frame to compute deltas efficiently.
    """

    def __init__(self, position_threshold: float = 0.001, rotation_threshold: float = 0.01):
        """Initialize compressor.

        Args:
            position_threshold: Minimum position change (mm) to record
            rotation_threshold: Minimum rotation change (degrees) to record
        """
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold

        # Track previous frame state for delta computation
        self.prev_components: Dict[str, Tuple[float, float, float]] = {}
        self.prev_modules: Dict[str, str] = {}

        # Statistics
        self.total_frames = 0
        self.total_deltas = 0
        self.total_full_state_size = 0
        self.total_delta_size = 0

    def compress_frame(
        self,
        index: int,
        label: str,
        iteration: int,
        phase: str,
        components: Dict[str, Tuple[float, float, float]],
        modules: Dict[str, str],
        forces: Dict[str, List[Tuple[float, float, str]]],
        overlaps: List[Tuple[str, str]],
        movement: Dict[str, Tuple[float, float, float]],
        connections: List[Tuple[str, str, str]],
        energy: float,
        max_move: float,
        overlap_count: int,
        total_wire_length: float,
    ) -> FrameDelta:
        """Compress a frame by computing deltas from previous frame.

        Returns:
            FrameDelta with only changed components
        """
        changed_components = {}
        changed_modules = {}

        # Compute component deltas
        for ref, comp_data in components.items():
            # Handle both 3-tuple (x, y, rotation) and 5-tuple (x, y, rotation, width, height)
            x, y, rotation = comp_data[0], comp_data[1], comp_data[2]

            if ref not in self.prev_components:
                # New component - store full state
                changed_components[ref] = ComponentDelta(x=x, y=y, rotation=rotation)
            else:
                # Existing component - compute delta
                prev_data = self.prev_components[ref]
                prev_x, prev_y, prev_rot = prev_data[0], prev_data[1], prev_data[2]
                delta = ComponentDelta()

                if abs(x - prev_x) > self.position_threshold:
                    delta.x = x
                if abs(y - prev_y) > self.position_threshold:
                    delta.y = y
                if abs(rotation - prev_rot) > self.rotation_threshold:
                    delta.rotation = rotation

                if delta.has_changes():
                    changed_components[ref] = delta

        # Compute module deltas (only store changed module assignments)
        for ref, module_type in modules.items():
            if ref not in self.prev_modules or self.prev_modules[ref] != module_type:
                changed_modules[ref] = module_type

        # Update state for next frame
        self.prev_components = components.copy()
        self.prev_modules = modules.copy()

        # Create delta frame
        delta = FrameDelta(
            index=index,
            label=label,
            iteration=iteration,
            phase=phase,
            changed_components=changed_components,
            forces=forces,
            overlaps=overlaps,
            changed_modules=changed_modules,
            energy=energy,
            max_move=max_move,
            overlap_count=overlap_count,
            total_wire_length=total_wire_length,
            movement=movement,
            connections=connections if index == 0 else [],  # Only store in frame 0
        )

        # Update statistics
        self.total_frames += 1
        self.total_deltas += len(changed_components)
        self.total_full_state_size += len(components) * 3  # 3 floats per component
        self.total_delta_size += len(changed_components) * 3  # Average 3 fields per delta

        if index % 100 == 0 and index > 0:
            compression_ratio = (1 - self.total_delta_size / max(1, self.total_full_state_size)) * 100
            logger.debug(
                f"Delta compression stats: {self.total_frames} frames, "
                f"{self.total_deltas} total deltas, "
                f"{compression_ratio:.1f}% size reduction"
            )

        return delta

    def reset(self):
        """Reset compressor state (e.g., between different placement runs)."""
        self.prev_components = {}
        self.prev_modules = {}
        self.total_frames = 0
        self.total_deltas = 0
        self.total_full_state_size = 0
        self.total_delta_size = 0


class DeltaDecompressor:
    """Reconstructs full frame state from delta-compressed frames.

    Used during playback to convert deltas back to complete frame state.
    """

    def __init__(self):
        """Initialize decompressor."""
        self.current_components: Dict[str, Tuple[float, float, float]] = {}
        self.current_modules: Dict[str, str] = {}
        self.connections: List[Tuple[str, str, str]] = []

    def decompress_frame(self, delta: FrameDelta) -> Dict:
        """Reconstruct full frame state from delta.

        Returns:
            Dictionary with complete frame data ready for rendering
        """
        # Apply component deltas to current state
        for ref, comp_delta in delta.changed_components.items():
            if ref not in self.current_components:
                # New component - initialize with delta values
                x = comp_delta.x or 0.0
                y = comp_delta.y or 0.0
                rotation = comp_delta.rotation or 0.0
                self.current_components[ref] = (x, y, rotation)
            else:
                # Update existing component with delta
                x, y, rotation = self.current_components[ref]
                if comp_delta.x is not None:
                    x = comp_delta.x
                if comp_delta.y is not None:
                    y = comp_delta.y
                if comp_delta.rotation is not None:
                    rotation = comp_delta.rotation
                self.current_components[ref] = (x, y, rotation)

        # Apply module deltas
        for ref, module_type in delta.changed_modules.items():
            self.current_modules[ref] = module_type

        # Store connections from frame 0
        if delta.index == 0 and delta.connections:
            self.connections = delta.connections

        # Return complete frame state
        return {
            "index": delta.index,
            "label": delta.label,
            "iteration": delta.iteration,
            "phase": delta.phase,
            "components": self.current_components.copy(),
            "modules": self.current_modules.copy(),
            "forces": delta.forces,
            "overlaps": delta.overlaps,
            "movement": delta.movement,
            "connections": self.connections,
            "energy": delta.energy,
            "max_move": delta.max_move,
            "overlap_count": delta.overlap_count,
            "total_wire_length": delta.total_wire_length,
        }

    def reset(self):
        """Reset decompressor state."""
        self.current_components = {}
        self.current_modules = {}
        self.connections = []


def estimate_compression_ratio(
    num_components: int,
    num_frames: int,
    avg_moved_per_frame: float = 0.15,
) -> float:
    """Estimate compression ratio for a placement run.

    Args:
        num_components: Total number of components
        num_frames: Total number of frames
        avg_moved_per_frame: Average fraction of components that move per frame

    Returns:
        Estimated compression ratio (0.0 = no compression, 0.9 = 90% reduction)
    """
    # Full state: num_frames * num_components * 3 floats
    full_size = num_frames * num_components * 3

    # Delta size: frame_0 (full) + remaining frames (only moved components)
    delta_size = num_components * 3  # Frame 0
    delta_size += (num_frames - 1) * num_components * avg_moved_per_frame * 3  # Delta frames

    compression_ratio = 1.0 - (delta_size / full_size)
    return compression_ratio
