"""
Tests for the RoutingManager multi-phase routing pipeline.

Tests cover:
- Configuration and initialization
- RoutingManagerConfig dataclass
- RoutingManagerResult dataclass
- RoutingPhase and NetPriority enums
"""

import pytest
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock

from atoplace.board.abstraction import (
    Board,
    BoardOutline,
    Component,
    Net,
    Pad,
)
from atoplace.routing.manager import (
    RoutingManager,
    RoutingManagerConfig,
    RoutingManagerResult,
    RoutingPhase,
    NetPriority,
    DiffPair,
)
from atoplace.routing.fanout import FanoutStrategy
from atoplace.dfm.profiles import DFMProfile


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_board() -> Board:
    """A simple board for routing tests."""
    board = Board(name="test_board")
    board.outline = BoardOutline(
        width=50.0,
        height=50.0,
        origin_x=0.0,
        origin_y=0.0,
        has_outline=True,
    )

    # Add a simple IC
    ic = Component(
        reference="U1",
        footprint="Package_SO:SOIC-8",
        value="IC",
        x=25.0,
        y=25.0,
        width=4.0,
        height=5.0,
    )
    ic.pads = [
        Pad(number="1", x=-1.5, y=-2.0, width=0.6, height=1.5, net="VCC"),
        Pad(number="2", x=-1.5, y=-0.67, width=0.6, height=1.5, net="USB_D+"),
        Pad(number="3", x=-1.5, y=0.67, width=0.6, height=1.5, net="USB_D-"),
        Pad(number="4", x=-1.5, y=2.0, width=0.6, height=1.5, net="GND"),
        Pad(number="5", x=1.5, y=2.0, width=0.6, height=1.5, net="DATA"),
        Pad(number="6", x=1.5, y=0.67, width=0.6, height=1.5, net="CLK"),
        Pad(number="7", x=1.5, y=-0.67, width=0.6, height=1.5, net="RST"),
        Pad(number="8", x=1.5, y=-2.0, width=0.6, height=1.5, net="VCC"),
    ]

    # Add a connector
    conn = Component(
        reference="J1",
        footprint="Connector_USB:USB_C",
        value="USB",
        x=10.0,
        y=25.0,
        width=9.0,
        height=7.0,
    )
    conn.pads = [
        Pad(number="1", x=-3.0, y=0, width=0.5, height=1.0, net="VCC"),
        Pad(number="2", x=-1.0, y=0, width=0.5, height=1.0, net="USB_D+"),
        Pad(number="3", x=1.0, y=0, width=0.5, height=1.0, net="USB_D-"),
        Pad(number="4", x=3.0, y=0, width=0.5, height=1.0, net="GND"),
    ]

    # Add a capacitor
    cap = Component(
        reference="C1",
        footprint="Capacitor_SMD:C_0603",
        value="100nF",
        x=35.0,
        y=25.0,
        width=1.6,
        height=0.8,
    )
    cap.pads = [
        Pad(number="1", x=-0.75, y=0, width=0.5, height=0.6, net="VCC"),
        Pad(number="2", x=0.75, y=0, width=0.5, height=0.6, net="GND"),
    ]

    board.components = {"U1": ic, "J1": conn, "C1": cap}

    # Create nets
    board.nets = {
        "VCC": Net(name="VCC", code=1, is_power=True),
        "GND": Net(name="GND", code=2, is_ground=True),
        "USB_D+": Net(name="USB_D+", code=3),
        "USB_D-": Net(name="USB_D-", code=4),
        "DATA": Net(name="DATA", code=5),
        "CLK": Net(name="CLK", code=6),
        "RST": Net(name="RST", code=7),
    }

    # Add connections to nets
    board.nets["VCC"].add_connection("U1", "1")
    board.nets["VCC"].add_connection("U1", "8")
    board.nets["VCC"].add_connection("J1", "1")
    board.nets["VCC"].add_connection("C1", "1")

    board.nets["GND"].add_connection("U1", "4")
    board.nets["GND"].add_connection("J1", "4")
    board.nets["GND"].add_connection("C1", "2")

    board.nets["USB_D+"].add_connection("U1", "2")
    board.nets["USB_D+"].add_connection("J1", "2")

    board.nets["USB_D-"].add_connection("U1", "3")
    board.nets["USB_D-"].add_connection("J1", "3")

    # Set component_ref on pads
    for ref, comp in board.components.items():
        for pad in comp.pads:
            pad.component_ref = ref

    return board


@pytest.fixture
def dfm_profile() -> DFMProfile:
    """Default DFM profile for tests."""
    return DFMProfile(
        name="test_profile",
        min_trace_width=0.15,
        min_spacing=0.15,
        min_via_drill=0.2,
        min_via_annular=0.1,
    )


@pytest.fixture
def manager_config() -> RoutingManagerConfig:
    """Default manager configuration."""
    return RoutingManagerConfig(
        enable_fanout=False,  # Disable for simpler tests
        enable_diff_pairs=True,
    )


# =============================================================================
# Enum Tests
# =============================================================================

class TestRoutingPhase:
    """Test RoutingPhase enum."""

    def test_phase_values(self):
        """Test that all expected phases exist with correct values."""
        assert RoutingPhase.FANOUT.value == "fanout"
        assert RoutingPhase.CRITICAL.value == "critical"
        assert RoutingPhase.GENERAL.value == "general"
        assert RoutingPhase.CLEANUP.value == "cleanup"

    def test_phase_count(self):
        """Test that we have exactly 4 phases."""
        assert len(RoutingPhase) == 4


class TestNetPriority:
    """Test NetPriority enum."""

    def test_priority_values(self):
        """Test priority values are in correct order."""
        assert NetPriority.CRITICAL.value == 100
        assert NetPriority.POWER.value == 80
        assert NetPriority.SIGNAL.value == 50
        assert NetPriority.LOW.value == 10

    def test_priority_ordering(self):
        """Test that priorities have correct relative ordering."""
        assert NetPriority.CRITICAL.value > NetPriority.POWER.value
        assert NetPriority.POWER.value > NetPriority.SIGNAL.value
        assert NetPriority.SIGNAL.value > NetPriority.LOW.value


# =============================================================================
# DiffPair Tests
# =============================================================================

class TestDiffPair:
    """Test DiffPair dataclass."""

    def test_diff_pair_creation(self):
        """Test creating a differential pair."""
        dp = DiffPair(net_p="USB_D+", net_n="USB_D-")

        assert dp.net_p == "USB_D+"
        assert dp.net_n == "USB_D-"
        assert dp.gap == 0.15  # default
        assert dp.width == 0.2  # default

    def test_diff_pair_custom_values(self):
        """Test differential pair with custom gap and width."""
        dp = DiffPair(net_p="HDMI_TX0_P", net_n="HDMI_TX0_N", gap=0.1, width=0.15)

        assert dp.gap == 0.1
        assert dp.width == 0.15

    def test_diff_pair_name_property(self):
        """Test the auto-generated name property."""
        dp = DiffPair(net_p="USB_D+", net_n="USB_D-")

        assert dp.name == "DIFF_USB_D+_USB_D-"


# =============================================================================
# Configuration Tests
# =============================================================================

class TestRoutingManagerConfig:
    """Test RoutingManagerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RoutingManagerConfig()

        assert config.enable_fanout is True
        assert config.fanout_strategy == FanoutStrategy.AUTO
        assert config.enable_diff_pairs is True
        assert config.diff_pairs == []
        assert config.greedy_weight == 2.0
        assert config.grid_size == 0.1
        assert config.max_iterations == 50000
        assert config.visualize is False
        assert config.animate is False

    def test_custom_config(self):
        """Test custom configuration values."""
        diff_pairs = [DiffPair(net_p="USB_D+", net_n="USB_D-")]
        config = RoutingManagerConfig(
            enable_fanout=False,
            fanout_strategy=FanoutStrategy.DOGBONE,
            enable_diff_pairs=False,
            diff_pairs=diff_pairs,
            greedy_weight=3.0,
            grid_size=0.05,
            max_iterations=100000,
            visualize=True,
            animate=True,
        )

        assert config.enable_fanout is False
        assert config.fanout_strategy == FanoutStrategy.DOGBONE
        assert config.enable_diff_pairs is False
        assert len(config.diff_pairs) == 1
        assert config.greedy_weight == 3.0
        assert config.grid_size == 0.05
        assert config.max_iterations == 100000
        assert config.visualize is True
        assert config.animate is True


# =============================================================================
# Result Tests
# =============================================================================

class TestRoutingManagerResult:
    """Test RoutingManagerResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = RoutingManagerResult(success=True)

        assert result.success is True
        assert result.results == {}
        assert result.fanout_results == {}
        assert result.unrouted_nets == []
        assert result.stats == {}

    def test_result_with_data(self):
        """Test result with populated data."""
        result = RoutingManagerResult(
            success=False,
            results={"NET1": Mock()},
            fanout_results={"U1": Mock()},
            unrouted_nets=["NET2", "NET3"],
            stats={"total_nets": 10, "routed_nets": 7}
        )

        assert result.success is False
        assert "NET1" in result.results
        assert "U1" in result.fanout_results
        assert result.unrouted_nets == ["NET2", "NET3"]
        assert result.stats["total_nets"] == 10

    def test_success_false_with_unrouted(self):
        """Test that success is false when there are unrouted nets."""
        result = RoutingManagerResult(
            success=False,
            unrouted_nets=["GND", "VCC"]
        )

        assert result.success is False
        assert len(result.unrouted_nets) == 2


# =============================================================================
# Initialization Tests
# =============================================================================

class TestRoutingManagerInit:
    """Test RoutingManager initialization."""

    def test_init_with_required_params(self, simple_board, dfm_profile):
        """Test initialization with required parameters."""
        manager = RoutingManager(simple_board, dfm_profile)

        assert manager.board is simple_board
        assert manager.dfm is dfm_profile
        assert manager.config is not None
        assert isinstance(manager.config, RoutingManagerConfig)

    def test_init_with_custom_config(self, simple_board, dfm_profile, manager_config):
        """Test initialization with custom config."""
        manager = RoutingManager(simple_board, dfm_profile, config=manager_config)

        assert manager.config.enable_fanout is False
        assert manager.config.enable_diff_pairs is True

    def test_init_with_visualizer(self, simple_board, dfm_profile):
        """Test initialization with visualizer."""
        mock_viz = Mock()
        manager = RoutingManager(simple_board, dfm_profile, visualizer=mock_viz)

        assert manager.viz is mock_viz

    def test_init_creates_subcomponents(self, simple_board, dfm_profile):
        """Test that initialization creates required subcomponents."""
        manager = RoutingManager(simple_board, dfm_profile)

        assert manager.fanout_gen is not None
        assert manager.obstacle_builder is not None

    def test_init_state(self, simple_board, dfm_profile):
        """Test initial state after initialization."""
        manager = RoutingManager(simple_board, dfm_profile)

        assert manager.obstacle_index is None
        assert manager.nets_to_route == set()
        assert manager.routed_nets == set()
        assert manager.results is not None
        assert manager.results.success is True


# =============================================================================
# Integration Tests (with mocking)
# =============================================================================

class TestRoutingManagerRun:
    """Test RoutingManager.run() method."""

    def test_run_identifies_nets(self, simple_board, dfm_profile):
        """Test that run() identifies nets to route."""
        config = RoutingManagerConfig(
            enable_fanout=False,
            enable_diff_pairs=False,
        )
        manager = RoutingManager(simple_board, dfm_profile, config=config)

        # Mock the general phase to avoid full routing
        with patch.object(manager, '_run_general_phase'):
            manager.run()

        # Should have identified nets with 2+ connections
        assert len(manager.nets_to_route) > 0
        assert "VCC" in manager.nets_to_route
        assert "GND" in manager.nets_to_route

    def test_run_returns_result(self, simple_board, dfm_profile):
        """Test that run() returns a RoutingManagerResult."""
        config = RoutingManagerConfig(
            enable_fanout=False,
            enable_diff_pairs=False,
        )
        manager = RoutingManager(simple_board, dfm_profile, config=config)

        # Mock the general phase to avoid full routing
        with patch.object(manager, '_run_general_phase'):
            result = manager.run()

        assert isinstance(result, RoutingManagerResult)
        assert "total_nets" in result.stats
        assert "routed_nets" in result.stats
        assert "completion_rate" in result.stats

    def test_run_with_fanout_disabled(self, simple_board, dfm_profile):
        """Test that fanout phase is skipped when disabled."""
        config = RoutingManagerConfig(
            enable_fanout=False,
            enable_diff_pairs=False,
        )
        manager = RoutingManager(simple_board, dfm_profile, config=config)

        with patch.object(manager, '_run_fanout_phase') as mock_fanout:
            with patch.object(manager, '_run_general_phase'):
                manager.run()

        # Fanout should not have been called
        mock_fanout.assert_not_called()

    def test_run_with_fanout_enabled(self, simple_board, dfm_profile):
        """Test that fanout phase runs when enabled."""
        config = RoutingManagerConfig(
            enable_fanout=True,
            enable_diff_pairs=False,
        )
        manager = RoutingManager(simple_board, dfm_profile, config=config)

        with patch.object(manager, '_run_fanout_phase') as mock_fanout:
            with patch.object(manager, '_run_general_phase'):
                manager.run()

        # Fanout should have been called
        mock_fanout.assert_called_once()
