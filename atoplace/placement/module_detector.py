"""
Module Detector

Automatically identifies functional modules (power, RF, digital, analog)
from component connectivity and characteristics. This enables intelligent
grouping during placement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import re

from ..board.abstraction import Board, Component, Net


class ModuleType(Enum):
    """Types of functional modules."""
    POWER_SUPPLY = "power_supply"
    POWER_REGULATOR = "power_regulator"
    MICROCONTROLLER = "microcontroller"
    RF_FRONTEND = "rf_frontend"
    ANALOG_SIGNAL = "analog_signal"
    DIGITAL_LOGIC = "digital_logic"
    SENSOR = "sensor"
    CONNECTOR = "connector"
    DECOUPLING = "decoupling"
    ESD_PROTECTION = "esd_protection"
    CRYSTAL_OSCILLATOR = "crystal"
    LED_INDICATOR = "led"
    UNKNOWN = "unknown"


@dataclass
class FunctionalModule:
    """A detected functional module in the design."""
    module_type: ModuleType
    name: str
    components: Set[str] = field(default_factory=set)
    primary_component: Optional[str] = None  # Main IC/component
    priority: int = 0  # Placement priority (higher = place first)
    placement_hints: Dict[str, str] = field(default_factory=dict)

    def add_component(self, ref: str):
        """Add a component to this module."""
        self.components.add(ref)


class ModuleDetector:
    """
    Detects functional modules from board connectivity and component types.

    Uses a combination of:
    1. Component reference prefix (U, R, C, L, etc.)
    2. Footprint patterns (QFN, SOT-23, 0402, etc.)
    3. Net connectivity (power rails, signal paths)
    4. Heuristics for common patterns (decoupling, ESD, etc.)
    """

    # Patterns for component identification
    COMPONENT_PATTERNS = {
        'microcontroller': [
            r'STM32', r'ESP32', r'ATMEGA', r'PIC', r'NRF5', r'RP2040',
            r'SAMD', r'EFM32', r'IMXRT', r'RAK3172',
        ],
        'rf': [
            r'RF', r'LORA', r'WIFI', r'BLE', r'ANT', r'BALUN',
            r'LNA', r'PA_', r'MIXER',
        ],
        'power_regulator': [
            r'LDO', r'BUCK', r'BOOST', r'REG', r'TPS7', r'TLV75',
            r'AP2112', r'AMS1117', r'MCP1700', r'RT9080',
        ],
        'sensor': [
            r'BME', r'BMP', r'MPU', r'ICM', r'LIS', r'LSM', r'HDC',
            r'SHT', r'MAX', r'TMP', r'LMT', r'VEML', r'TSL', r'QMI',
        ],
        'esd': [
            r'ESD', r'TVS', r'SP0503', r'USBLC', r'PRTR',
        ],
        'opamp': [
            r'OPA', r'LM358', r'TL08', r'AD8', r'MCP6',
        ],
        'connector': [
            r'USB', r'HDR', r'CONN', r'JST', r'MOLEX', r'SMA',
        ],
        'crystal': [
            r'XTAL', r'OSC', r'CRYSTAL', r'32K', r'ABM',
        ],
    }

    def __init__(self, board: Board):
        self.board = board
        self.modules: List[FunctionalModule] = []
        self._component_to_module: Dict[str, FunctionalModule] = {}

    def detect(self) -> List[FunctionalModule]:
        """
        Detect all functional modules in the design.

        Returns:
            List of detected FunctionalModule instances
        """
        self.modules = []
        self._component_to_module = {}

        # 1. Identify main ICs and their supporting components
        self._detect_ics()

        # 2. Detect power supply chains
        self._detect_power_modules()

        # 3. Detect RF sections
        self._detect_rf_modules()

        # 4. Detect decoupling networks
        self._detect_decoupling()

        # 5. Detect connectors
        self._detect_connectors()

        # 6. Detect crystals/oscillators
        self._detect_crystals()

        # 7. Detect LEDs
        self._detect_leds()

        # 8. Assign remaining components
        self._assign_remaining()

        # Set placement priorities
        self._assign_priorities()

        return self.modules

    def _detect_ics(self):
        """Detect main ICs (MCU, sensors, etc.)."""
        for ref, comp in self.board.components.items():
            if not ref.startswith('U'):
                continue

            fp = comp.footprint.upper()
            value = comp.value.upper()

            module_type = ModuleType.UNKNOWN

            # Check against known patterns
            for mtype, patterns in self.COMPONENT_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, value) or re.search(pattern, fp):
                        module_type = ModuleType[mtype.upper()]
                        break
                if module_type != ModuleType.UNKNOWN:
                    break

            # Create module for each IC
            if module_type != ModuleType.UNKNOWN:
                module = FunctionalModule(
                    module_type=module_type,
                    name=f"{module_type.value}_{ref}",
                    primary_component=ref,
                )
                module.add_component(ref)
                self.modules.append(module)
                self._component_to_module[ref] = module

    def _detect_power_modules(self):
        """Detect power supply modules."""
        # Find regulators
        for ref, comp in self.board.components.items():
            if ref in self._component_to_module:
                continue

            fp = comp.footprint.upper()
            value = comp.value.upper()

            is_regulator = False
            for pattern in self.COMPONENT_PATTERNS['power_regulator']:
                if re.search(pattern, value) or re.search(pattern, fp):
                    is_regulator = True
                    break

            if is_regulator:
                module = FunctionalModule(
                    module_type=ModuleType.POWER_REGULATOR,
                    name=f"power_{ref}",
                    primary_component=ref,
                    placement_hints={"near_input": "true"},
                )
                module.add_component(ref)

                # Find associated capacitors on input/output
                self._add_connected_passives(module, ref, ['C'])

                self.modules.append(module)
                self._component_to_module[ref] = module

    def _detect_rf_modules(self):
        """Detect RF front-end modules."""
        # Find RF components
        rf_components = set()

        for ref, comp in self.board.components.items():
            fp = comp.footprint.upper()
            value = comp.value.upper()

            for pattern in self.COMPONENT_PATTERNS['rf']:
                if re.search(pattern, value) or re.search(pattern, fp):
                    rf_components.add(ref)
                    break

            # Check for antenna
            if 'ANT' in fp or 'ANTENNA' in fp:
                rf_components.add(ref)

        if rf_components:
            module = FunctionalModule(
                module_type=ModuleType.RF_FRONTEND,
                name="rf_frontend",
                placement_hints={
                    "edge_placement": "preferred",
                    "ground_plane": "required",
                },
            )

            for ref in rf_components:
                module.add_component(ref)
                self._component_to_module[ref] = module

            # Find RF matching network components
            for ref in rf_components:
                self._add_connected_passives(module, ref, ['L', 'C', 'R'])

            self.modules.append(module)

    def _detect_decoupling(self):
        """Detect decoupling capacitor networks."""
        # Find capacitors connected to power nets
        for ref, comp in self.board.components.items():
            if not ref.startswith('C'):
                continue
            if ref in self._component_to_module:
                continue

            # Check if connected to power/ground
            nets = comp.get_connected_nets()
            has_power = False
            has_ground = False
            ic_ref = None

            for net_name in nets:
                net = self.board.nets.get(net_name)
                if net:
                    if net.is_power:
                        has_power = True
                    if net.is_ground:
                        has_ground = True

                    # Find connected IC
                    for conn_ref in net.get_component_refs():
                        if conn_ref.startswith('U') and conn_ref in self._component_to_module:
                            ic_ref = conn_ref
                            break

            if has_power and has_ground and ic_ref:
                # This is a decoupling cap - add to IC's module
                ic_module = self._component_to_module.get(ic_ref)
                if ic_module:
                    ic_module.add_component(ref)
                    self._component_to_module[ref] = ic_module

    def _detect_connectors(self):
        """Detect connector modules."""
        for ref, comp in self.board.components.items():
            if ref in self._component_to_module:
                continue

            fp = comp.footprint.upper()
            value = comp.value.upper()

            is_connector = ref.startswith('J') or ref.startswith('P')
            for pattern in self.COMPONENT_PATTERNS['connector']:
                if re.search(pattern, value) or re.search(pattern, fp):
                    is_connector = True
                    break

            if is_connector:
                module = FunctionalModule(
                    module_type=ModuleType.CONNECTOR,
                    name=f"connector_{ref}",
                    primary_component=ref,
                    placement_hints={"edge_placement": "required"},
                )
                module.add_component(ref)

                # Add ESD protection if connected
                self._add_connected_by_type(module, ref, ['D', 'U'],
                                            self.COMPONENT_PATTERNS.get('esd', []))

                self.modules.append(module)
                self._component_to_module[ref] = module

    def _detect_crystals(self):
        """Detect crystal oscillator circuits."""
        for ref, comp in self.board.components.items():
            if ref in self._component_to_module:
                continue

            fp = comp.footprint.upper()
            value = comp.value.upper()

            is_crystal = ref.startswith('Y') or ref.startswith('X')
            for pattern in self.COMPONENT_PATTERNS['crystal']:
                if re.search(pattern, value) or re.search(pattern, fp):
                    is_crystal = True
                    break

            if is_crystal:
                module = FunctionalModule(
                    module_type=ModuleType.CRYSTAL_OSCILLATOR,
                    name=f"crystal_{ref}",
                    primary_component=ref,
                    placement_hints={"close_to_mcu": "required"},
                )
                module.add_component(ref)

                # Add load capacitors
                self._add_connected_passives(module, ref, ['C'])

                self.modules.append(module)
                self._component_to_module[ref] = module

    def _detect_leds(self):
        """Detect LED indicator circuits."""
        for ref, comp in self.board.components.items():
            if not ref.startswith('D'):
                continue
            if ref in self._component_to_module:
                continue

            fp = comp.footprint.upper()
            if 'LED' in fp:
                module = FunctionalModule(
                    module_type=ModuleType.LED_INDICATOR,
                    name=f"led_{ref}",
                    primary_component=ref,
                )
                module.add_component(ref)

                # Add current limiting resistor
                self._add_connected_passives(module, ref, ['R'])

                self.modules.append(module)
                self._component_to_module[ref] = module

    def _add_connected_passives(self, module: FunctionalModule,
                                 ref: str, prefixes: List[str]):
        """Add passive components connected to a reference component."""
        comp = self.board.components.get(ref)
        if not comp:
            return

        connected_nets = comp.get_connected_nets()
        for net_name in connected_nets:
            net = self.board.nets.get(net_name)
            if not net:
                continue

            for conn_ref in net.get_component_refs():
                if conn_ref == ref:
                    continue
                if conn_ref in self._component_to_module:
                    continue

                for prefix in prefixes:
                    if conn_ref.startswith(prefix):
                        module.add_component(conn_ref)
                        self._component_to_module[conn_ref] = module
                        break

    def _add_connected_by_type(self, module: FunctionalModule,
                                ref: str, prefixes: List[str],
                                patterns: List[str]):
        """Add components matching patterns that are connected."""
        comp = self.board.components.get(ref)
        if not comp:
            return

        connected_nets = comp.get_connected_nets()
        for net_name in connected_nets:
            net = self.board.nets.get(net_name)
            if not net:
                continue

            for conn_ref in net.get_component_refs():
                if conn_ref == ref:
                    continue
                if conn_ref in self._component_to_module:
                    continue

                conn_comp = self.board.components.get(conn_ref)
                if not conn_comp:
                    continue

                # Check prefix
                prefix_match = any(conn_ref.startswith(p) for p in prefixes)
                if not prefix_match:
                    continue

                # Check patterns
                fp = conn_comp.footprint.upper()
                value = conn_comp.value.upper()
                for pattern in patterns:
                    if re.search(pattern, value) or re.search(pattern, fp):
                        module.add_component(conn_ref)
                        self._component_to_module[conn_ref] = module
                        break

    def _assign_remaining(self):
        """Assign remaining unclassified components."""
        # Create a catch-all module for unassigned components
        unassigned = FunctionalModule(
            module_type=ModuleType.UNKNOWN,
            name="unassigned",
        )

        for ref in self.board.components:
            if ref not in self._component_to_module:
                unassigned.add_component(ref)
                self._component_to_module[ref] = unassigned

        if unassigned.components:
            self.modules.append(unassigned)

    def _assign_priorities(self):
        """Assign placement priorities to modules."""
        priority_map = {
            ModuleType.CONNECTOR: 100,  # Place first (edge constraints)
            ModuleType.RF_FRONTEND: 90,
            ModuleType.MICROCONTROLLER: 80,
            ModuleType.CRYSTAL_OSCILLATOR: 75,
            ModuleType.POWER_REGULATOR: 70,
            ModuleType.SENSOR: 60,
            ModuleType.DECOUPLING: 50,
            ModuleType.LED_INDICATOR: 40,
            ModuleType.ESD_PROTECTION: 30,
            ModuleType.UNKNOWN: 10,
        }

        for module in self.modules:
            module.priority = priority_map.get(module.module_type, 0)

    def get_module_for_component(self, ref: str) -> Optional[FunctionalModule]:
        """Get the module containing a specific component."""
        return self._component_to_module.get(ref)

    def get_modules_by_type(self, module_type: ModuleType) -> List[FunctionalModule]:
        """Get all modules of a specific type."""
        return [m for m in self.modules if m.module_type == module_type]
