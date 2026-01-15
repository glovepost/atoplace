"""
Tests for differential pair detection.

Tests cover:
- Detection of various naming patterns
- Case insensitivity
- Net pairing correctness
- Edge cases (empty lists, no pairs, partial matches)
"""

import pytest
from typing import List

from atoplace.routing.diff_pairs import (
    DiffPairDetector,
    DiffPairSpec,
    DiffPairPattern,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def usb_nets() -> List[str]:
    """USB differential pair nets."""
    return ["USB_D+", "USB_D-", "VCC", "GND", "RESET"]


@pytest.fixture
def multi_pattern_nets() -> List[str]:
    """Nets with multiple differential pair patterns."""
    return [
        # USB style
        "USB_D+", "USB_D-",
        # P/N suffix
        "LVDS_TX_P", "LVDS_TX_N",
        "LVDS_RX_P", "LVDS_RX_N",
        # Direct suffix
        "DATAP", "DATAN",
        # POS/NEG style
        "CLK_POS", "CLK_NEG",
        # Non-diff nets
        "VCC", "GND", "RESET", "EN",
    ]


@pytest.fixture
def hdmi_nets() -> List[str]:
    """HDMI differential pairs (multiple channels)."""
    return [
        "HDMI_D0_P", "HDMI_D0_N",
        "HDMI_D1_P", "HDMI_D1_N",
        "HDMI_D2_P", "HDMI_D2_N",
        "HDMI_CLK_P", "HDMI_CLK_N",
        "HDMI_CEC", "HDMI_HPD", "+5V",
    ]


# =============================================================================
# Detection Tests
# =============================================================================

class TestDiffPairDetector:
    """Test the DiffPairDetector class."""

    def test_detect_usb_style(self, usb_nets):
        """Test detection of USB D+/D- style pairs."""
        detector = DiffPairDetector(usb_nets)
        pairs = detector.detect()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.positive_net == "USB_D+"
        assert pair.negative_net == "USB_D-"
        assert pair.pattern == DiffPairPattern.USB_STYLE

    def test_detect_p_n_suffix(self):
        """Test detection of _P/_N suffix style."""
        nets = ["LVDS_TX_P", "LVDS_TX_N", "VCC", "GND"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.positive_net == "LVDS_TX_P"
        assert pair.negative_net == "LVDS_TX_N"
        assert pair.pattern == DiffPairPattern.PLUS_MINUS

    def test_detect_direct_suffix(self):
        """Test detection of direct P/N suffix (NETP/NETN)."""
        nets = ["DATAP", "DATAN", "CLK", "RST"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.positive_net == "DATAP"
        assert pair.negative_net == "DATAN"
        assert pair.pattern == DiffPairPattern.SUFFIX_P_N

    def test_detect_pos_neg_style(self):
        """Test detection of _POS/_NEG style."""
        nets = ["CLK_POS", "CLK_NEG", "DATA", "EN"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.positive_net == "CLK_POS"
        assert pair.negative_net == "CLK_NEG"
        assert pair.pattern == DiffPairPattern.POSITIVE_NEGATIVE

    def test_detect_multiple_pairs(self, multi_pattern_nets):
        """Test detection of multiple differential pairs."""
        detector = DiffPairDetector(multi_pattern_nets)
        pairs = detector.detect()

        # Should detect USB, LVDS_TX, LVDS_RX, DATA, CLK
        assert len(pairs) == 5

        # Verify all pairs are valid
        pair_names = {p.name for p in pairs}
        # Names may vary based on pattern matching
        assert len(pair_names) == 5

    def test_detect_hdmi_pairs(self, hdmi_nets):
        """Test detection of HDMI differential pairs."""
        detector = DiffPairDetector(hdmi_nets)
        pairs = detector.detect()

        # Should detect D0, D1, D2, CLK
        assert len(pairs) == 4

        # Verify each channel is detected
        positive_nets = {p.positive_net for p in pairs}
        assert "HDMI_D0_P" in positive_nets
        assert "HDMI_D1_P" in positive_nets
        assert "HDMI_D2_P" in positive_nets
        assert "HDMI_CLK_P" in positive_nets

    def test_case_insensitivity(self):
        """Test that detection is case-insensitive."""
        nets = ["usb_d+", "USB_D-", "vcc", "gnd"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert len(pairs) == 1

    def test_empty_net_list(self):
        """Test handling of empty net list."""
        detector = DiffPairDetector([])
        pairs = detector.detect()

        assert pairs == []

    def test_no_pairs_found(self):
        """Test when no differential pairs exist."""
        nets = ["VCC", "GND", "RESET", "CLK", "DATA"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert pairs == []

    def test_orphan_nets(self):
        """Test that nets without complements are not paired."""
        # Only positive net, no negative
        nets = ["USB_D+", "VCC", "GND"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert pairs == []

    def test_nets_not_reused(self):
        """Test that each net is only used in one pair."""
        # Potential ambiguity: DATA_P could pair with DATA_N or DATAP with DATAN
        nets = ["DATA_P", "DATA_N", "DATAP", "DATAN"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        # Each net should only appear once
        all_nets = []
        for p in pairs:
            all_nets.append(p.positive_net)
            all_nets.append(p.negative_net)

        assert len(all_nets) == len(set(all_nets))  # No duplicates

    def test_plus_minus_direct_style(self):
        """Test detection of direct +/- suffix (NET+/NET-)."""
        nets = ["CLK+", "CLK-", "VCC", "GND"]
        detector = DiffPairDetector(nets)
        pairs = detector.detect()

        assert len(pairs) == 1
        assert pairs[0].positive_net == "CLK+"
        assert pairs[0].negative_net == "CLK-"


class TestDiffPairSpec:
    """Test DiffPairSpec dataclass."""

    def test_default_electrical_params(self):
        """Test default electrical parameters."""
        spec = DiffPairSpec(
            name="USB",
            positive_net="USB_D+",
            negative_net="USB_D-",
            pattern=DiffPairPattern.USB_STYLE,
        )

        assert spec.impedance == 100.0
        assert spec.trace_width == 0.2
        assert spec.spacing == 0.15
        assert spec.max_skew == 0.1
        assert spec.max_uncoupled == 2.0

    def test_custom_electrical_params(self):
        """Test custom electrical parameters."""
        spec = DiffPairSpec(
            name="HDMI",
            positive_net="HDMI_D0_P",
            negative_net="HDMI_D0_N",
            pattern=DiffPairPattern.PLUS_MINUS,
            impedance=90.0,
            trace_width=0.15,
            spacing=0.1,
        )

        assert spec.impedance == 90.0
        assert spec.trace_width == 0.15
        assert spec.spacing == 0.1
