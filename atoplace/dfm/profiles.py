"""
Design for Manufacturing (DFM) Profiles

Defines manufacturing constraints for specific fabrication houses.
Each profile contains the design rules that must be satisfied for
successful manufacturing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DFMProfile:
    """Design for Manufacturing constraints for a specific fab."""

    name: str
    description: str = ""

    # Trace/space rules (mm)
    min_trace_width: float = 0.127  # 5 mil default
    min_spacing: float = 0.127
    min_trace_to_edge: float = 0.3

    # Via rules (mm)
    min_via_drill: float = 0.3
    min_via_annular: float = 0.15
    min_via_to_via: float = 0.254
    min_via_to_trace: float = 0.127

    # Hole rules (mm)
    min_hole_diameter: float = 0.3
    max_hole_diameter: float = 6.3
    min_hole_to_hole: float = 0.5
    min_hole_to_edge: float = 0.3

    # Solder mask rules (mm)
    min_mask_dam: float = 0.1
    mask_expansion: float = 0.05

    # Silkscreen rules (mm)
    min_silk_width: float = 0.15
    min_silk_height: float = 0.8
    min_silk_to_pad: float = 0.15

    # Layer count options
    supported_layers: List[int] = field(default_factory=lambda: [1, 2])

    # Board size limits (mm)
    min_board_size: float = 10.0
    max_board_size: float = 500.0

    # Cost tier
    cost_tier: str = "standard"  # "standard", "advanced", "hdi"

    def validate_trace(self, width: float) -> bool:
        """Check if trace width meets minimum requirement."""
        return width >= self.min_trace_width

    def validate_spacing(self, spacing: float) -> bool:
        """Check if spacing meets minimum requirement."""
        return spacing >= self.min_spacing

    def validate_via(self, drill: float, annular: float) -> bool:
        """Check if via dimensions meet requirements."""
        return drill >= self.min_via_drill and annular >= self.min_via_annular

    def validate_hole(self, diameter: float) -> bool:
        """Check if hole diameter is within allowed range."""
        return self.min_hole_diameter <= diameter <= self.max_hole_diameter

    def to_kicad_drc(self) -> Dict:
        """Export as KiCad DRC rule settings (mm)."""
        return {
            "min_track_width": self.min_trace_width,
            "min_clearance": self.min_spacing,
            "min_via_diameter": self.min_via_drill + 2 * self.min_via_annular,
            "min_via_drill": self.min_via_drill,
            "min_through_hole_diameter": self.min_hole_diameter,
            "min_hole_to_hole": self.min_hole_to_hole,
            "min_silk_text_height": self.min_silk_height,
            "min_silk_text_thickness": self.min_silk_width,
        }


# Pre-defined profiles for common fabs

JLCPCB_STANDARD = DFMProfile(
    name="JLCPCB Standard (1-2 layer)",
    description="Standard capabilities for 1-2 layer boards at JLCPCB",
    min_trace_width=0.127,  # 5 mil
    min_spacing=0.127,  # 5 mil
    min_trace_to_edge=0.3,
    min_via_drill=0.3,
    min_via_annular=0.15,
    min_via_to_via=0.254,
    min_via_to_trace=0.127,
    min_hole_diameter=0.3,
    max_hole_diameter=6.3,
    min_hole_to_hole=0.5,
    min_hole_to_edge=0.3,
    min_mask_dam=0.1,
    mask_expansion=0.05,
    min_silk_width=0.15,
    min_silk_height=0.8,
    min_silk_to_pad=0.15,
    supported_layers=[1, 2],
    cost_tier="standard",
)

JLCPCB_STANDARD_4LAYER = DFMProfile(
    name="JLCPCB Standard (4 layer)",
    description="Standard capabilities for 4 layer boards at JLCPCB",
    min_trace_width=0.09,  # 3.5 mil
    min_spacing=0.09,
    min_trace_to_edge=0.25,
    min_via_drill=0.2,
    min_via_annular=0.1,
    min_via_to_via=0.2,
    min_via_to_trace=0.1,
    min_hole_diameter=0.2,
    max_hole_diameter=6.3,
    min_hole_to_hole=0.45,
    min_hole_to_edge=0.25,
    min_mask_dam=0.08,
    mask_expansion=0.05,
    min_silk_width=0.12,
    min_silk_height=0.7,
    min_silk_to_pad=0.1,
    supported_layers=[4],
    cost_tier="standard",
)

JLCPCB_ADVANCED = DFMProfile(
    name="JLCPCB Advanced",
    description="Advanced capabilities for tighter tolerances at JLCPCB",
    min_trace_width=0.075,  # 3 mil
    min_spacing=0.075,
    min_trace_to_edge=0.2,
    min_via_drill=0.15,
    min_via_annular=0.075,
    min_via_to_via=0.15,
    min_via_to_trace=0.075,
    min_hole_diameter=0.15,
    max_hole_diameter=6.3,
    min_hole_to_hole=0.35,
    min_hole_to_edge=0.2,
    min_mask_dam=0.06,
    mask_expansion=0.04,
    min_silk_width=0.1,
    min_silk_height=0.6,
    min_silk_to_pad=0.08,
    supported_layers=[1, 2, 4, 6, 8],
    cost_tier="advanced",
)

OSHPARK_2LAYER = DFMProfile(
    name="OSH Park 2-Layer",
    description="OSH Park standard 2-layer service",
    min_trace_width=0.152,  # 6 mil
    min_spacing=0.152,
    min_trace_to_edge=0.381,  # 15 mil
    min_via_drill=0.254,  # 10 mil
    min_via_annular=0.127,  # 5 mil
    min_via_to_via=0.254,
    min_via_to_trace=0.152,
    min_hole_diameter=0.254,
    max_hole_diameter=6.35,
    min_hole_to_hole=0.381,
    min_hole_to_edge=0.381,
    min_mask_dam=0.102,  # 4 mil
    mask_expansion=0.051,
    min_silk_width=0.152,
    min_silk_height=0.762,
    min_silk_to_pad=0.152,
    supported_layers=[2],
    cost_tier="standard",
)

PCBWay_STANDARD = DFMProfile(
    name="PCBWay Standard",
    description="PCBWay standard manufacturing capabilities",
    min_trace_width=0.1,  # ~4 mil
    min_spacing=0.1,
    min_trace_to_edge=0.25,
    min_via_drill=0.2,
    min_via_annular=0.1,
    min_via_to_via=0.2,
    min_via_to_trace=0.1,
    min_hole_diameter=0.2,
    max_hole_diameter=6.5,
    min_hole_to_hole=0.4,
    min_hole_to_edge=0.25,
    min_mask_dam=0.08,
    mask_expansion=0.05,
    min_silk_width=0.12,
    min_silk_height=0.7,
    min_silk_to_pad=0.1,
    supported_layers=[1, 2, 4, 6, 8, 10],
    cost_tier="standard",
)

# Profile registry
PROFILES: Dict[str, DFMProfile] = {
    "jlcpcb_standard": JLCPCB_STANDARD,
    "jlcpcb_standard_4layer": JLCPCB_STANDARD_4LAYER,
    "jlcpcb_advanced": JLCPCB_ADVANCED,
    "oshpark_2layer": OSHPARK_2LAYER,
    "pcbway_standard": PCBWay_STANDARD,
}


def get_profile(name: str) -> DFMProfile:
    """
    Get DFM profile by name.

    Args:
        name: Profile identifier (e.g., "jlcpcb_standard")

    Returns:
        DFMProfile instance

    Raises:
        ValueError: If profile name is not found
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(f"Unknown DFM profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> List[str]:
    """List all available DFM profile names."""
    return sorted(PROFILES.keys())


def get_profile_for_layers(layer_count: int, fab: str = "jlcpcb") -> DFMProfile:
    """
    Get the appropriate DFM profile for a given layer count and fab.

    Args:
        layer_count: Number of copper layers
        fab: Fabrication house identifier

    Returns:
        Most appropriate DFMProfile
    """
    if fab == "jlcpcb":
        if layer_count <= 2:
            return JLCPCB_STANDARD
        elif layer_count == 4:
            return JLCPCB_STANDARD_4LAYER
        else:
            return JLCPCB_ADVANCED
    elif fab == "oshpark":
        return OSHPARK_2LAYER
    elif fab == "pcbway":
        return PCBWay_STANDARD
    else:
        # Default to JLCPCB standard
        return JLCPCB_STANDARD
