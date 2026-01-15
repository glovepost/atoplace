"""
Tests for the RoutingManager multi-phase routing pipeline.

Tests cover:
- Configuration and initialization
- Net classification (critical, power, ground, signal)
- Differential pair registration
- Progress callback functionality
- Net ordering
"""

import pytest
from typing import List, Dict
from unittest.mock import Mock, patch

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
)
from atoplace.routing.astar_router import RouterConfig


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
def manager_config() -> RoutingManagerConfig:
    """Default manager configuration."""
    return RoutingManagerConfig(
        enable_fanout=False,  # Disable for simpler tests
        enable_critical_nets=True,
        enable_general_routing=True,
    )


# =============================================================================
# Initialization Tests
# =============================================================================

class TestRoutingManagerInit:
    """Test RoutingManager initialization."""

    def test_init_with_defaults(self, simple_board):
        """Test initialization with default config."""
        manager = RoutingManager(simple_board)

        assert manager.board is simple_board
        assert manager.config is not None
        assert manager.dfm is not None

    def test_init_with_custom_config(self, simple_board, manager_config):
        """Test initialization with custom config."""
        manager = RoutingManager(simple_board, config=manager_config)

        assert manager.config.enable_fanout is False
        assert manager.config.enable_critical_nets is True

    def test_init_with_router_config(self, simple_board):
        """Test that router config is properly initialized."""
        router_config = RouterConfig(
            greedy_weight=2.5,
            grid_size=0.05,
        )
        config = RoutingManagerConfig(router_config=router_config)
        manager = RoutingManager(simple_board, config=config)

        assert manager.config.router_config.greedy_weight == 2.5
        assert manager.config.router_config.grid_size == 0.05


# =============================================================================
# Net Classification Tests
# =============================================================================

class TestNetClassification:
    """Test net classification functionality."""

    def test_add_diff_pair(self, simple_board, manager_config):
        """Test differential pair registration."""
        manager = RoutingManager(simple_board, config=manager_config)
        manager.add_diff_pair("USB", "USB_D+", "USB_D-")

        # Diff pair should be registered
        assert len(manager._diff_pairs) == 1
        pair = manager._diff_pairs[0]
        assert pair.name == "USB"
        assert pair.positive_net == "USB_D+"
        assert pair.negative_net == "USB_D-"

        # Both nets should be marked as critical
        assert "USB_D+" in manager._critical_nets
        assert "USB_D-" in manager._critical_nets

    def test_add_diff_pair_chaining(self, simple_board, manager_config):
        """Test method chaining for diff pair registration."""
        manager = RoutingManager(simple_board, config=manager_config)
        result = manager.add_diff_pair("USB", "USB_D+", "USB_D-")

        # Should return self for chaining
        assert result is manager

    def test_set_critical_nets(self, simple_board, manager_config):
        """Test critical net registration."""
        manager = RoutingManager(simple_board, config=manager_config)
        manager.set_critical_nets(["CLK", "DATA"])

        assert "CLK" in manager._critical_nets
        assert "DATA" in manager._critical_nets

    def test_set_power_nets(self, simple_board, manager_config):
        """Test power net registration."""
        manager = RoutingManager(simple_board, config=manager_config)
        manager.set_power_nets(["VCC", "+3V3"])

        assert "VCC" in manager._power_nets
        assert "+3V3" in manager._power_nets

    def test_set_ground_nets(self, simple_board, manager_config):
        """Test ground net registration."""
        manager = RoutingManager(simple_board, config=manager_config)
        manager.set_ground_nets(["GND", "AGND"])

        assert "GND" in manager._ground_nets
        assert "AGND" in manager._ground_nets


# =============================================================================
# Progress Callback Tests
# =============================================================================

class TestProgressCallback:
    """Test progress callback functionality."""

    def test_set_progress_callback(self, simple_board, manager_config):
        """Test setting progress callback."""
        callback = Mock()
        manager = RoutingManager(simple_board, config=manager_config)
        result = manager.set_progress_callback(callback)

        # Should return self for chaining
        assert result is manager
        assert manager._progress_callback is callback

    def test_progress_callback_invocation(self, simple_board, manager_config):
        """Test that progress callback is invoked."""
        callback = Mock()
        manager = RoutingManager(simple_board, config=manager_config)
        manager.set_progress_callback(callback)

        # Manually invoke progress reporting
        manager._report_progress("test_phase", 0.5)

        callback.assert_called_once_with("test_phase", 0.5)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestRoutingManagerConfig:
    """Test RoutingManagerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RoutingManagerConfig()

        assert config.enable_fanout is True
        assert config.enable_critical_nets is True
        assert config.enable_general_routing is True
        assert config.visualize is False

    def test_custom_router_config(self):
        """Test custom router configuration."""
        router_config = RouterConfig(
            greedy_weight=3.0,
            grid_size=0.2,
            trace_width=0.15,
        )
        config = RoutingManagerConfig(router_config=router_config)

        assert config.router_config.greedy_weight == 3.0
        assert config.router_config.grid_size == 0.2
        assert config.router_config.trace_width == 0.15


# =============================================================================
# Result Tests
# =============================================================================

class TestRoutingManagerResult:
    """Test RoutingManagerResult dataclass."""

    def test_completion_rate_calculation(self):
        """Test completion rate property."""
        result = RoutingManagerResult(
            success=True,
            phases_completed=[RoutingPhase.GENERAL_ROUTING],
            net_results={},
            total_nets=10,
            routed_nets=8,
            failed_nets=2,
        )

        assert result.completion_rate == 80.0

    def test_completion_rate_no_nets(self):
        """Test completion rate with no nets."""
        result = RoutingManagerResult(
            success=True,
            phases_completed=[],
            net_results={},
            total_nets=0,
            routed_nets=0,
            failed_nets=0,
        )

        # Should return 100% when no nets
        assert result.completion_rate == 100.0

    def test_success_determination(self):
        """Test success flag."""
        result = RoutingManagerResult(
            success=False,
            phases_completed=[RoutingPhase.GENERAL_ROUTING],
            net_results={},
            total_nets=10,
            routed_nets=5,
            failed_nets=5,
        )

        assert result.success is False
