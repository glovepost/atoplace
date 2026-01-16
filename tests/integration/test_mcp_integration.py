#!/usr/bin/env python3
"""
Comprehensive MCP server integration test.

Tests all major tool categories after BoardInspector refactoring
and launcher improvements to ensure functionality is intact.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from atoplace.mcp import server
from atoplace.board import Board, Component, Pad, Net


class TestSession:
    """Mock MCP session for testing."""
    def __init__(self):
        self.board = self._create_test_board()

    def _create_test_board(self):
        """Create a test board with various components."""
        board = Board()
        board.width = 100.0
        board.height = 100.0

        # Microcontroller
        u1 = Component("U1", "IC", "TQFP-44")
        u1.x, u1.y = 50.0, 50.0
        u1.width, u1.height = 10.0, 10.0
        u1.value = "STM32F103"
        u1.layer = "F.Cu"
        for i in range(4):
            u1.pads.append(Pad(f"{i+1}", 48.0 + i*0.8, 45.0, 0.3, 0.6))

        # Resistors
        r1 = Component("R1", "Resistor", "0603")
        r1.x, r1.y = 60.0, 50.0
        r1.width, r1.height = 1.6, 0.8
        r1.value = "10k"
        r1.layer = "F.Cu"
        r1.pads.append(Pad("1", 59.2, 50.0, 0.8, 0.9))
        r1.pads.append(Pad("2", 60.8, 50.0, 0.8, 0.9))

        r2 = Component("R2", "Resistor", "0603")
        r2.x, r2.y = 65.0, 50.0
        r2.width, r2.height = 1.6, 0.8
        r2.value = "10k"
        r2.layer = "F.Cu"
        r2.pads.append(Pad("1", 64.2, 50.0, 0.8, 0.9))
        r2.pads.append(Pad("2", 65.8, 50.0, 0.8, 0.9))

        # Capacitors
        c1 = Component("C1", "Capacitor", "0603")
        c1.x, c1.y = 70.0, 50.0
        c1.width, c1.height = 1.6, 0.8
        c1.value = "100nF"
        c1.layer = "F.Cu"
        c1.pads.append(Pad("1", 69.2, 50.0, 0.8, 0.9))
        c1.pads.append(Pad("2", 70.8, 50.0, 0.8, 0.9))

        # Connector
        j1 = Component("J1", "Connector", "USB-C")
        j1.x, j1.y = 10.0, 50.0
        j1.width, j1.height = 8.0, 3.0
        j1.value = "USB4105"
        j1.layer = "F.Cu"
        for i in range(4):
            j1.pads.append(Pad(f"{i+1}", 8.0 + i*1.0, 50.0, 0.6, 1.5))

        # LED
        d1 = Component("D1", "LED", "0603")
        d1.x, d1.y = 75.0, 50.0
        d1.width, d1.height = 1.6, 0.8
        d1.value = "RED"
        d1.layer = "F.Cu"
        d1.pads.append(Pad("A", 74.2, 50.0, 0.8, 0.9))
        d1.pads.append(Pad("K", 75.8, 50.0, 0.8, 0.9))

        board.components = [u1, r1, r2, c1, j1, d1]

        # Create nets
        vcc = Net("VCC")
        gnd = Net("GND")
        led = Net("LED")

        vcc.pads = [u1.pads[0], r1.pads[0], c1.pads[0]]
        gnd.pads = [u1.pads[1], r1.pads[1], c1.pads[1], j1.pads[0]]
        led.pads = [u1.pads[2], r2.pads[0], d1.pads[0]]

        board.nets = [vcc, gnd, led]

        return board


def run_tests():
    """Run comprehensive MCP tool tests."""

    print("=" * 80)
    print("MCP Server Integration Test Suite")
    print("=" * 80)
    print()

    # Set up mock session
    server.session = TestSession()

    results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }

    # Category 1: Board Loading & Inspection
    print("Category 1: Board Loading & Inspection")
    print("-" * 80)

    tests = [
        ("get_board_summary", lambda: server.get_board_summary()),
        ("inspect_region", lambda: server.inspect_region(["U1", "R1"])),
        ("find_components (by ref)", lambda: server.find_components("R", "ref")),
        ("find_components (by value)", lambda: server.find_components("10k", "value")),
        ("find_components (by footprint)", lambda: server.find_components("0603", "footprint")),
        ("check_overlaps (all)", lambda: server.check_overlaps()),
        ("check_overlaps (subset)", lambda: server.check_overlaps(["U1", "R1"])),
        ("get_unplaced_components", lambda: server.get_unplaced_components()),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            data = json.loads(result)

            # Check for error response
            if "error" in data:
                print(f"❌ {test_name}: {data['error']}")
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {data['error']}")
            else:
                print(f"✅ {test_name}")
                results["passed"] += 1

        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")

    print()

    # Category 2: Placement Operations
    print("Category 2: Placement Operations")
    print("-" * 80)

    tests = [
        ("move_absolute", lambda: server.move_absolute("R1", 65.0, 55.0)),
        ("move_relative", lambda: server.move_relative("R2", 2.0, 0.0)),
        ("rotate", lambda: server.rotate("C1", 90.0)),
        ("place_next_to", lambda: server.place_next_to("D1", "R2", "right", 2.0)),
        ("align_components", lambda: server.align_components(["R1", "R2", "C1"], "x")),
        ("distribute_evenly", lambda: server.distribute_evenly(["R1", "R2", "C1"])),
        ("stack_components", lambda: server.stack_components(["R1", "R2", "C1"], "down", 2.0)),
        ("lock_components", lambda: server.lock_components(["U1"])),
        ("arrange_pattern (grid)", lambda: server.arrange_pattern(["R1", "R2", "C1", "D1"], "grid", cols=2)),
        ("cluster_around", lambda: server.cluster_around("U1", ["C1", "R1"])),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            data = json.loads(result)

            if "error" in data:
                print(f"❌ {test_name}: {data['error']}")
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {data['error']}")
            else:
                print(f"✅ {test_name}")
                results["passed"] += 1

        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")

    print()

    # Category 3: Validation
    print("Category 3: Validation")
    print("-" * 80)

    tests = [
        ("validate_placement", lambda: server.validate_placement()),
        ("run_drc", lambda: server.run_drc(use_kicad=False)),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            data = json.loads(result)

            if "error" in data:
                print(f"❌ {test_name}: {data['error']}")
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {data['error']}")
            else:
                print(f"✅ {test_name}")
                results["passed"] += 1

        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")

    print()

    # Category 4: Module Detection
    print("Category 4: Module Detection")
    print("-" * 80)

    tests = [
        ("detect_modules", lambda: server.detect_modules()),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            data = json.loads(result)

            if "error" in data:
                print(f"❌ {test_name}: {data['error']}")
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {data['error']}")
            else:
                print(f"✅ {test_name}")
                results["passed"] += 1

        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")

    print()

    # Category 5: Constraint Parsing
    print("Category 5: Constraint Parsing")
    print("-" * 80)

    tests = [
        ("parse_constraint (proximity)", lambda: server.parse_constraint("Keep C1 close to U1")),
        ("parse_constraint (edge)", lambda: server.parse_constraint("USB connector on left edge")),
        ("parse_constraint (zone)", lambda: server.parse_constraint("Keep RF in top-left")),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            data = json.loads(result)

            if "error" in data:
                print(f"❌ {test_name}: {data['error']}")
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {data['error']}")
            else:
                print(f"✅ {test_name}")
                results["passed"] += 1

        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")

    print()

    # Category 6: BGA/Routing
    print("Category 6: BGA/Routing")
    print("-" * 80)

    tests = [
        ("detect_bga_components", lambda: server.detect_bga_components()),
        ("detect_diff_pairs", lambda: server.detect_diff_pairs()),
        ("get_routing_preview", lambda: server.get_routing_preview()),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            data = json.loads(result)

            if "error" in data:
                print(f"❌ {test_name}: {data['error']}")
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {data['error']}")
            else:
                print(f"✅ {test_name}")
                results["passed"] += 1

        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")

    print()

    # Summary
    print("=" * 80)
    print("Test Results")
    print("=" * 80)
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"Total: {results['passed'] + results['failed']}")
    print()

    if results["errors"]:
        print("Errors:")
        for error in results["errors"]:
            print(f"  - {error}")
        print()

    success_rate = (results['passed'] / (results['passed'] + results['failed'])) * 100 if (results['passed'] + results['failed']) > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print()

    return results['failed'] == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
