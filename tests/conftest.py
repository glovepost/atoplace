"""
Shared test fixtures for AtoPlace tests.

Provides reusable board, component, and session fixtures
for testing MCP server, API actions, and context generators.
"""

import pytest
from pathlib import Path
from typing import Dict, List

from atoplace.board.abstraction import (
    Board,
    BoardOutline,
    Component,
    Net,
    Pad,
    Layer,
)


@pytest.fixture
def simple_component() -> Component:
    """A simple component with basic properties."""
    return Component(
        reference="U1",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        value="ATtiny85",
        x=50.0,
        y=50.0,
        width=4.0,
        height=5.0,
        rotation=0.0,
    )


@pytest.fixture
def component_with_pads() -> Component:
    """A component with pads for net connectivity testing."""
    comp = Component(
        reference="U1",
        footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
        value="ATtiny85",
        x=50.0,
        y=50.0,
        width=4.0,
        height=5.0,
        rotation=0.0,
    )
    comp.pads = [
        Pad(number="1", x=-1.5, y=-2.0, width=0.6, height=1.5, net="VCC"),
        Pad(number="2", x=-1.5, y=-0.67, width=0.6, height=1.5, net="PB0"),
        Pad(number="3", x=-1.5, y=0.67, width=0.6, height=1.5, net="PB1"),
        Pad(number="4", x=-1.5, y=2.0, width=0.6, height=1.5, net="GND"),
        Pad(number="5", x=1.5, y=2.0, width=0.6, height=1.5, net="PB2"),
        Pad(number="6", x=1.5, y=0.67, width=0.6, height=1.5, net="PB3"),
        Pad(number="7", x=1.5, y=-0.67, width=0.6, height=1.5, net="PB4"),
        Pad(number="8", x=1.5, y=-2.0, width=0.6, height=1.5, net="VCC"),
    ]
    return comp


@pytest.fixture
def resistor_component() -> Component:
    """A resistor component."""
    comp = Component(
        reference="R1",
        footprint="Resistor_SMD:R_0603_1608Metric",
        value="10k",
        x=45.0,
        y=55.0,
        width=1.6,
        height=0.8,
        rotation=0.0,
    )
    comp.pads = [
        Pad(number="1", x=-0.75, y=0, width=0.5, height=0.6, net="VCC"),
        Pad(number="2", x=0.75, y=0, width=0.5, height=0.6, net="PB0"),
    ]
    return comp


@pytest.fixture
def capacitor_component() -> Component:
    """A capacitor component."""
    comp = Component(
        reference="C1",
        footprint="Capacitor_SMD:C_0603_1608Metric",
        value="100nF",
        x=45.0,
        y=45.0,
        width=1.6,
        height=0.8,
        rotation=0.0,
    )
    comp.pads = [
        Pad(number="1", x=-0.75, y=0, width=0.5, height=0.6, net="VCC"),
        Pad(number="2", x=0.75, y=0, width=0.5, height=0.6, net="GND"),
    ]
    return comp


@pytest.fixture
def connector_component() -> Component:
    """A connector component."""
    comp = Component(
        reference="J1",
        footprint="Connector_USB:USB_C_Receptacle",
        value="USB_C",
        x=10.0,
        y=50.0,
        width=9.0,
        height=7.0,
        rotation=0.0,
    )
    comp.pads = [
        Pad(number="A1", x=-3.5, y=-2.5, width=0.3, height=1.0, net="GND"),
        Pad(number="A4", x=-2.5, y=-2.5, width=0.3, height=1.0, net="VCC"),
        Pad(number="A5", x=-1.5, y=-2.5, width=0.3, height=1.0, net="CC1"),
    ]
    return comp


@pytest.fixture
def test_board(
    component_with_pads,
    resistor_component,
    capacitor_component,
    connector_component,
) -> Board:
    """A complete test board with multiple components and nets."""
    board = Board(name="test_board")
    board.outline = BoardOutline(
        width=100.0,
        height=100.0,
        origin_x=0.0,
        origin_y=0.0,
        has_outline=True,
    )

    # Add components
    board.components = {
        "U1": component_with_pads,
        "R1": resistor_component,
        "C1": capacitor_component,
        "J1": connector_component,
    }

    # Create nets
    vcc_net = Net(name="VCC", code=1, is_power=True)
    vcc_net.add_connection("U1", "1")
    vcc_net.add_connection("U1", "8")
    vcc_net.add_connection("R1", "1")
    vcc_net.add_connection("C1", "1")
    vcc_net.add_connection("J1", "A4")

    gnd_net = Net(name="GND", code=2, is_ground=True)
    gnd_net.add_connection("U1", "4")
    gnd_net.add_connection("C1", "2")
    gnd_net.add_connection("J1", "A1")

    pb0_net = Net(name="PB0", code=3)
    pb0_net.add_connection("U1", "2")
    pb0_net.add_connection("R1", "2")

    board.nets = {
        "VCC": vcc_net,
        "GND": gnd_net,
        "PB0": pb0_net,
    }

    # Create pad objects with component_ref for topology queries
    # These are needed for MCP tools like get_connected_components
    for ref, comp in board.components.items():
        for pad in comp.pads:
            pad.component_ref = ref

    return board


@pytest.fixture
def empty_board() -> Board:
    """An empty board with no components."""
    board = Board(name="empty_board")
    board.outline = BoardOutline(width=50.0, height=50.0)
    board.components = {}
    board.nets = {}
    return board


@pytest.fixture
def locked_component() -> Component:
    """A locked component that cannot be moved."""
    comp = Component(
        reference="U2",
        footprint="Package_QFP:LQFP-48",
        value="STM32F103",
        x=60.0,
        y=60.0,
        width=9.0,
        height=9.0,
        rotation=0.0,
        locked=True,
    )
    return comp


@pytest.fixture
def board_with_locked(test_board, locked_component) -> Board:
    """A board with a locked component."""
    test_board.components["U2"] = locked_component
    return test_board


@pytest.fixture
def overlapping_components() -> List[Component]:
    """Two components that overlap."""
    c1 = Component(
        reference="C1",
        footprint="Capacitor_SMD:C_0603",
        value="100nF",
        x=50.0,
        y=50.0,
        width=1.6,
        height=0.8,
    )
    c2 = Component(
        reference="C2",
        footprint="Capacitor_SMD:C_0603",
        value="100nF",
        x=50.5,  # Overlaps with C1
        y=50.0,
        width=1.6,
        height=0.8,
    )
    return [c1, c2]


# Session fixtures for MCP server testing
@pytest.fixture
def mcp_session(test_board):
    """A session with a loaded test board for MCP testing."""
    from atoplace.api.session import Session

    session = Session()
    session.board = test_board
    session.source_path = Path("/tmp/test.kicad_pcb")
    session._undo_stack = [session._take_snapshot("Initial")]
    return session


@pytest.fixture
def empty_session():
    """An unloaded session."""
    from atoplace.api.session import Session

    return Session()
