"""Swap group detection for identifying functionally equivalent pins.

Detects groups of pins that can be safely swapped without changing functionality:
- FPGA I/O banks (pins within same bank are often interchangeable)
- MCU GPIO ports (pins within same port, same function)
- Memory interfaces (address/data bus pins within same group)
- Connector pins (generic signal pins on ribbon cables, headers)

Detection uses heuristics based on:
1. Pin naming patterns (PA0-PA7, GPIO_0-GPIO_15, etc.)
2. Component footprint type (BGA, QFP, etc.)
3. Net naming patterns (helps identify bus signals)
4. Explicit annotations in component properties
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SwapGroupType(Enum):
    """Types of pin swap groups."""
    FPGA_BANK = "fpga_bank"       # FPGA I/O bank
    MCU_GPIO = "mcu_gpio"         # MCU GPIO port
    MEMORY_DATA = "memory_data"   # Memory data bus
    MEMORY_ADDR = "memory_addr"   # Memory address bus
    CONNECTOR = "connector"       # Generic connector pins
    MANUAL = "manual"             # Manually specified group


@dataclass
class SwappablePin:
    """A pin that can potentially be swapped."""
    pad_number: str
    net_name: Optional[str]
    x: float  # Absolute pad position
    y: float
    bank_id: Optional[str] = None  # For FPGA banks
    function: Optional[str] = None  # Pin function if known


@dataclass
class SwapGroup:
    """A group of functionally equivalent pins that can be swapped.

    All pins in a group are interchangeable from a functionality standpoint.
    Swapping pins within a group only affects routing complexity, not circuit behavior.
    """
    name: str
    group_type: SwapGroupType
    component_ref: str
    pins: List[SwappablePin] = field(default_factory=list)

    # Constraints on the group
    preserve_differential: bool = True  # Don't break diff pairs
    preserve_bus_order: bool = False    # Whether relative order matters

    # Metadata
    confidence: float = 1.0  # Detection confidence (0-1)
    bank_id: Optional[str] = None  # FPGA bank identifier

    @property
    def size(self) -> int:
        """Number of pins in the group."""
        return len(self.pins)

    @property
    def connected_pins(self) -> List[SwappablePin]:
        """Get pins that have net connections."""
        return [p for p in self.pins if p.net_name]

    def __repr__(self) -> str:
        return f"SwapGroup({self.name}, {self.group_type.value}, {self.size} pins)"


class SwapGroupDetector:
    """Detects groups of swappable pins on components.

    Uses pattern matching and heuristics to identify functionally equivalent pins
    that can be safely swapped without affecting circuit behavior.
    """

    # Patterns for MCU GPIO detection
    GPIO_PATTERNS = [
        # STM32 style: PA0, PA1, PB0, etc.
        (r'^P([A-K])(\d+)$', 'stm32'),
        # Generic GPIO: GPIO0, GPIO_0, GPIO[0], etc.
        (r'^GPIO[_\[]?(\d+)\]?$', 'generic'),
        # Arduino style: D0, D1, A0, A1
        (r'^([DA])(\d+)$', 'arduino'),
        # Raspberry Pi: GPIO_XX
        (r'^GPIO_?(\d{1,2})$', 'rpi'),
        # ESP32: IOxx
        (r'^IO(\d+)$', 'esp32'),
    ]

    # Patterns for FPGA I/O detection
    FPGA_PATTERNS = [
        # Xilinx: IOB_XX_YY, IOL_XX, IOR_XX
        (r'^IO[BLRT]_(\d+)(?:_.*)?$', 'xilinx'),
        # Intel/Altera: PIN_XX, IOx[y]
        (r'^PIN_([A-Z]+\d+)$', 'intel'),
        # Lattice: PIOX
        (r'^PIO([A-Z]?\d+)$', 'lattice'),
        # Generic bank: BANK_X_IOY
        (r'^BANK_?(\d+)_IO(\d+)$', 'generic_bank'),
    ]

    # Memory interface patterns
    MEMORY_PATTERNS = [
        # Data bus: D0-D31, DQ0-DQ63
        (r'^D(?:Q)?(\d+)$', 'data'),
        # Address bus: A0-A31, ADDR0-ADDR31
        (r'^A(?:DDR)?(\d+)$', 'address'),
        # DDR specific: DQS, DM
        (r'^DQ(?:S|M)(\d+)(?:[PN])?$', 'ddr_strobe'),
    ]

    # Connector patterns
    CONNECTOR_PATTERNS = [
        # Numbered pins: 1, 2, 3 or P1, P2
        (r'^P?(\d+)$', 'numbered'),
        # Signal naming: SIG_XX
        (r'^SIG(?:NAL)?_?(\d+)$', 'signal'),
    ]

    def __init__(self, board: "Board"):
        """
        Initialize detector with board data.

        Args:
            board: Board abstraction with components and nets
        """
        self.board = board
        self._detected_groups: Dict[str, List[SwapGroup]] = {}  # ref -> groups

    def detect_component(self, ref: str) -> List[SwapGroup]:
        """
        Detect swap groups for a specific component.

        Args:
            ref: Component reference designator

        Returns:
            List of detected swap groups
        """
        comp = self.board.get_component(ref)
        if not comp:
            logger.warning(f"Component {ref} not found")
            return []

        groups = []

        # Check component type based on reference prefix and footprint
        if self._is_fpga_mcu(comp):
            groups.extend(self._detect_fpga_mcu_groups(comp))
        elif self._is_connector(comp):
            groups.extend(self._detect_connector_groups(comp))
        elif self._is_memory(comp):
            groups.extend(self._detect_memory_groups(comp))

        # Cache results
        self._detected_groups[ref] = groups

        logger.info(f"Detected {len(groups)} swap groups for {ref}")
        return groups

    def detect_all(self) -> Dict[str, List[SwapGroup]]:
        """
        Detect swap groups for all components on the board.

        Returns:
            Dict mapping component refs to their swap groups
        """
        self._detected_groups = {}

        for ref, comp in self.board.components.items():
            groups = self.detect_component(ref)
            if groups:
                self._detected_groups[ref] = groups

        total_groups = sum(len(g) for g in self._detected_groups.values())
        logger.info(f"Detected {total_groups} total swap groups across {len(self._detected_groups)} components")

        return self._detected_groups

    def _is_fpga_mcu(self, comp: "Component") -> bool:
        """Check if component is an FPGA or MCU."""
        # Check reference prefix
        if comp.reference.startswith(('U', 'IC')):
            # Check footprint for BGA/QFP patterns typical of FPGAs/MCUs
            fp_lower = comp.footprint.lower()
            if any(pattern in fp_lower for pattern in
                   ['bga', 'qfp', 'lqfp', 'tqfp', 'qfn', 'lga']):
                return True

            # Check pad count (FPGAs/MCUs typically have many pins)
            if len(comp.pads) >= 32:
                return True

        return False

    def _is_connector(self, comp: "Component") -> bool:
        """Check if component is a connector."""
        ref_prefix = ''.join(c for c in comp.reference if c.isalpha())
        if ref_prefix in ('J', 'P', 'CN', 'CON', 'X'):
            return True

        fp_lower = comp.footprint.lower()
        if any(pattern in fp_lower for pattern in
               ['header', 'connector', 'pin', 'socket', 'usb', 'hdmi']):
            return True

        return False

    def _is_memory(self, comp: "Component") -> bool:
        """Check if component is a memory chip."""
        ref_prefix = ''.join(c for c in comp.reference if c.isalpha())
        if ref_prefix in ('U', 'IC'):
            fp_lower = comp.footprint.lower()
            if any(pattern in fp_lower for pattern in
                   ['ddr', 'sdram', 'sram', 'flash', 'eeprom', 'bga']):
                return True

            # Check for memory-like net patterns
            memory_nets = sum(
                1 for pad in comp.pads
                if pad.net and re.match(r'^D(?:Q)?\d+$', pad.net)
            )
            if memory_nets >= 8:
                return True

        return False

    def _detect_fpga_mcu_groups(self, comp: "Component") -> List[SwapGroup]:
        """Detect swap groups for FPGA/MCU components."""
        groups = []

        # Group pins by GPIO port (for MCUs)
        gpio_groups = self._group_by_gpio_port(comp)
        for port_name, pins in gpio_groups.items():
            if len(pins) >= 2:  # Need at least 2 pins to swap
                group = SwapGroup(
                    name=f"{comp.reference}_{port_name}",
                    group_type=SwapGroupType.MCU_GPIO,
                    component_ref=comp.reference,
                    pins=pins,
                    confidence=0.9
                )
                groups.append(group)

        # Group pins by FPGA bank
        fpga_groups = self._group_by_fpga_bank(comp)
        for bank_id, pins in fpga_groups.items():
            if len(pins) >= 2:
                group = SwapGroup(
                    name=f"{comp.reference}_BANK{bank_id}",
                    group_type=SwapGroupType.FPGA_BANK,
                    component_ref=comp.reference,
                    pins=pins,
                    bank_id=bank_id,
                    confidence=0.85
                )
                groups.append(group)

        # If no specific groups found, try generic IO grouping
        if not groups:
            generic_group = self._detect_generic_io_group(comp)
            if generic_group:
                groups.append(generic_group)

        return groups

    def _detect_connector_groups(self, comp: "Component") -> List[SwapGroup]:
        """Detect swap groups for connector components."""
        groups = []

        # Get all signal pins (exclude power/ground)
        signal_pins = []
        for pad in comp.pads:
            if not pad.net:
                continue

            net = self.board.get_net(pad.net)
            if net and (net.is_power or net.is_ground):
                continue

            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            signal_pins.append(SwappablePin(
                pad_number=pad.number,
                net_name=pad.net,
                x=abs_x,
                y=abs_y,
                function="signal"
            ))

        if len(signal_pins) >= 2:
            group = SwapGroup(
                name=f"{comp.reference}_SIGNALS",
                group_type=SwapGroupType.CONNECTOR,
                component_ref=comp.reference,
                pins=signal_pins,
                preserve_differential=True,
                confidence=0.7  # Lower confidence for connectors
            )
            groups.append(group)

        return groups

    def _detect_memory_groups(self, comp: "Component") -> List[SwapGroup]:
        """Detect swap groups for memory components."""
        groups = []

        # Group data bus pins
        data_pins = []
        addr_pins = []

        for pad in comp.pads:
            if not pad.net:
                continue

            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)

            # Check for data pin
            if re.match(r'^D(?:Q)?\d+$', pad.net):
                data_pins.append(SwappablePin(
                    pad_number=pad.number,
                    net_name=pad.net,
                    x=abs_x,
                    y=abs_y,
                    function="data"
                ))
            # Check for address pin
            elif re.match(r'^A(?:DDR)?\d+$', pad.net):
                addr_pins.append(SwappablePin(
                    pad_number=pad.number,
                    net_name=pad.net,
                    x=abs_x,
                    y=abs_y,
                    function="address"
                ))

        if len(data_pins) >= 4:  # At least 4-bit bus
            group = SwapGroup(
                name=f"{comp.reference}_DATA",
                group_type=SwapGroupType.MEMORY_DATA,
                component_ref=comp.reference,
                pins=data_pins,
                preserve_bus_order=False,  # Data bits can be swapped
                confidence=0.95
            )
            groups.append(group)

        if len(addr_pins) >= 4:
            group = SwapGroup(
                name=f"{comp.reference}_ADDR",
                group_type=SwapGroupType.MEMORY_ADDR,
                component_ref=comp.reference,
                pins=addr_pins,
                preserve_bus_order=False,  # With software update
                confidence=0.95
            )
            groups.append(group)

        return groups

    def _group_by_gpio_port(self, comp: "Component") -> Dict[str, List[SwappablePin]]:
        """Group pins by GPIO port (e.g., PA, PB for STM32)."""
        port_groups: Dict[str, List[SwappablePin]] = {}

        for pad in comp.pads:
            if not pad.net:
                continue

            # Try each GPIO pattern
            for pattern, style in self.GPIO_PATTERNS:
                match = re.match(pattern, pad.net, re.IGNORECASE)
                if match:
                    if style == 'stm32':
                        port = f"PORT_{match.group(1)}"
                    elif style in ('generic', 'rpi', 'esp32'):
                        # Group in blocks of 8 or 16
                        pin_num = int(match.group(1))
                        port = f"GPIO_{(pin_num // 8) * 8}-{(pin_num // 8) * 8 + 7}"
                    elif style == 'arduino':
                        port = f"PORT_{match.group(1)}"
                    else:
                        port = "GPIO"

                    if port not in port_groups:
                        port_groups[port] = []

                    abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                    port_groups[port].append(SwappablePin(
                        pad_number=pad.number,
                        net_name=pad.net,
                        x=abs_x,
                        y=abs_y,
                        function="gpio"
                    ))
                    break

        return port_groups

    def _group_by_fpga_bank(self, comp: "Component") -> Dict[str, List[SwappablePin]]:
        """Group pins by FPGA I/O bank."""
        bank_groups: Dict[str, List[SwappablePin]] = {}

        for pad in comp.pads:
            if not pad.net:
                continue

            # Try FPGA patterns
            for pattern, style in self.FPGA_PATTERNS:
                match = re.match(pattern, pad.net, re.IGNORECASE)
                if match:
                    if style == 'generic_bank':
                        bank_id = match.group(1)
                    else:
                        # Extract bank from pin position or use generic grouping
                        # For now, group by first digit
                        try:
                            num = int(re.search(r'\d+', match.group(1)).group())
                            bank_id = str(num // 16)  # Group in banks of 16
                        except (AttributeError, ValueError):
                            bank_id = "0"

                    if bank_id not in bank_groups:
                        bank_groups[bank_id] = []

                    abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
                    bank_groups[bank_id].append(SwappablePin(
                        pad_number=pad.number,
                        net_name=pad.net,
                        x=abs_x,
                        y=abs_y,
                        bank_id=bank_id,
                        function="io"
                    ))
                    break

        return bank_groups

    def _detect_generic_io_group(self, comp: "Component") -> Optional[SwapGroup]:
        """Detect a generic IO group when specific patterns don't match."""
        io_pins = []

        for pad in comp.pads:
            if not pad.net:
                continue

            # Skip power and ground
            net = self.board.get_net(pad.net)
            if net and (net.is_power or net.is_ground):
                continue

            # Skip obvious non-swappable pins
            net_upper = pad.net.upper()
            if any(skip in net_upper for skip in
                   ['VCC', 'VDD', 'GND', 'VSS', 'CLK', 'RESET', 'BOOT', 'JTAG']):
                continue

            abs_x, abs_y = pad.absolute_position(comp.x, comp.y, comp.rotation)
            io_pins.append(SwappablePin(
                pad_number=pad.number,
                net_name=pad.net,
                x=abs_x,
                y=abs_y,
                function="io"
            ))

        if len(io_pins) >= 4:
            return SwapGroup(
                name=f"{comp.reference}_IO",
                group_type=SwapGroupType.MCU_GPIO,
                component_ref=comp.reference,
                pins=io_pins,
                confidence=0.5  # Lower confidence for generic detection
            )

        return None

    @property
    def detected_groups(self) -> Dict[str, List[SwapGroup]]:
        """Get all detected swap groups."""
        return self._detected_groups
