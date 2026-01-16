"""Tests for atopile integration adapter."""

import pytest
from pathlib import Path
from atoplace.board.atopile_adapter import (
    AtopileProjectLoader,
    AtopileModuleParser,
    ComponentMetadata,
    ModuleHierarchy,
    detect_board_source,
)
from atoplace.board.abstraction import Board, Component


@pytest.fixture
def mock_atopile_project(tmp_path):
    """Create a mock atopile project structure."""
    # Create ato.yaml
    ato_yaml = tmp_path / "ato.yaml"
    ato_yaml.write_text("""
ato-version: ^0.2.0
builds:
  default:
    entry: elec/src/sensor.ato:SensorBoard
  custom:
    entry: elec/src/sensor.ato:CustomVariant
dependencies:
  - generics^v1.0.0
""")

    # Create ato-lock.yaml
    lock_yaml = tmp_path / "ato-lock.yaml"
    lock_yaml.write_text("""
components:
  C1:
    mpn: "GRM155R71C104KA88D"
    value: "100nF"
    package: "0402"
    manufacturer: "Murata"
  C2:
    mpn: "GRM155R61A105KE15D"
    value: "1uF"
    package: "0402"
  U1:
    mpn: "ESP32-S3-WROOM-1"
    package: "QFN-56"
    description: "WiFi/BLE MCU module"
  R1:
    mpn: "RC0402FR-0710KL"
    value: "10K"
    package: "0402"
""")

    # Create source directory structure
    elec_src = tmp_path / "elec" / "src"
    elec_src.mkdir(parents=True)

    # Create .ato source file
    ato_file = elec_src / "sensor.ato"
    ato_file.write_text("""
# Main sensor board module
module SensorBoard:
    # Power section
    power = new PowerSupply

    # MCU section
    mcu = new ESP32Module

    # Sensor section
    sensor = new TempSensor

    # Connect power
    power.vout ~ mcu.vin
    power.gnd ~ mcu.gnd

module PowerSupply:
    regulator = new LDO3V3
    c_in = new Capacitor
    c_out = new Capacitor

module ESP32Module:
    mcu = new ESP32
    decoupling = new Capacitor
    crystal = new Crystal

module TempSensor:
    sensor = new SHT40
    pullup = new Resistor
""")

    # Create layout directory structure
    layout_dir = tmp_path / "elec" / "layout" / "default"
    layout_dir.mkdir(parents=True)

    return tmp_path


@pytest.fixture
def mock_kicad_board(tmp_path):
    """Create a mock KiCad board file (minimal valid content)."""
    board_path = tmp_path / "test.kicad_pcb"
    # Note: This is a minimal stub - actual loading requires pcbnew
    board_path.write_text("(kicad_pcb (version 20211014))")
    return board_path


class TestAtopileProjectLoader:
    """Tests for AtopileProjectLoader."""

    def test_is_atopile_project_true(self, mock_atopile_project):
        """Should detect valid atopile project."""
        assert AtopileProjectLoader.is_atopile_project(mock_atopile_project)

    def test_is_atopile_project_false(self, tmp_path):
        """Should return False for non-atopile directory."""
        assert not AtopileProjectLoader.is_atopile_project(tmp_path)

    def test_find_project_root_from_subdir(self, mock_atopile_project):
        """Should find project root from subdirectory."""
        subdir = mock_atopile_project / "elec" / "src"
        root = AtopileProjectLoader.find_project_root(subdir)
        assert root == mock_atopile_project

    def test_find_project_root_none(self, tmp_path):
        """Should return None when no project found."""
        root = AtopileProjectLoader.find_project_root(tmp_path)
        assert root is None

    def test_load_ato_yaml(self, mock_atopile_project):
        """Should parse ato.yaml correctly."""
        loader = AtopileProjectLoader(mock_atopile_project)

        assert loader.project.ato_version == "^0.2.0"
        assert "default" in loader.project.builds
        assert "custom" in loader.project.builds
        assert len(loader.project.dependencies) == 1

    def test_get_build_names(self, mock_atopile_project):
        """Should list available builds."""
        loader = AtopileProjectLoader(mock_atopile_project)
        builds = loader.get_build_names()

        assert "default" in builds
        assert "custom" in builds

    def test_get_entry_point(self, mock_atopile_project):
        """Should return correct entry point."""
        loader = AtopileProjectLoader(mock_atopile_project)

        entry = loader.get_entry_point("default")
        assert entry == "elec/src/sensor.ato:SensorBoard"

        entry = loader.get_entry_point("custom")
        assert entry == "elec/src/sensor.ato:CustomVariant"

    def test_get_entry_point_invalid_build(self, mock_atopile_project):
        """Should raise error for invalid build name."""
        loader = AtopileProjectLoader(mock_atopile_project)

        with pytest.raises(ValueError, match="not found"):
            loader.get_entry_point("nonexistent")

    def test_get_board_path(self, mock_atopile_project):
        """Should construct correct board path."""
        loader = AtopileProjectLoader(mock_atopile_project)
        board_path = loader.get_board_path("default")

        expected = mock_atopile_project / "elec" / "layout" / "default" / "sensor.kicad_pcb"
        assert board_path == expected

    def test_load_lock_file(self, mock_atopile_project):
        """Should parse ato-lock.yaml correctly."""
        loader = AtopileProjectLoader(mock_atopile_project)

        assert "C1" in loader.lock_data["components"]
        assert loader.lock_data["components"]["C1"]["value"] == "100nF"
        assert loader.lock_data["components"]["U1"]["mpn"] == "ESP32-S3-WROOM-1"

    def test_get_component_metadata(self, mock_atopile_project):
        """Should return component metadata."""
        loader = AtopileProjectLoader(mock_atopile_project)

        meta = loader.get_component_metadata("C1")
        assert meta is not None
        assert meta.reference == "C1"
        assert meta.value == "100nF"
        assert meta.package == "0402"
        assert meta.manufacturer == "Murata"

    def test_get_component_metadata_missing(self, mock_atopile_project):
        """Should return None for missing component."""
        loader = AtopileProjectLoader(mock_atopile_project)

        meta = loader.get_component_metadata("C99")
        assert meta is None

    def test_get_ato_source_path(self, mock_atopile_project):
        """Should return correct .ato source path."""
        loader = AtopileProjectLoader(mock_atopile_project)
        ato_path = loader.get_ato_source_path("default")

        expected = mock_atopile_project / "elec" / "src" / "sensor.ato"
        assert ato_path == expected

    def test_invalid_project_raises(self, tmp_path):
        """Should raise error for invalid project."""
        with pytest.raises(ValueError, match="Not an atopile project"):
            AtopileProjectLoader(tmp_path)


class TestAtopileModuleParser:
    """Tests for AtopileModuleParser."""

    def test_parse_file(self, mock_atopile_project):
        """Should parse .ato file and extract modules."""
        parser = AtopileModuleParser()
        ato_path = mock_atopile_project / "elec" / "src" / "sensor.ato"
        modules = parser.parse_file(ato_path)

        assert "SensorBoard" in modules
        assert "PowerSupply" in modules
        assert "ESP32Module" in modules
        assert "TempSensor" in modules

    def test_parse_module_hierarchy(self, mock_atopile_project):
        """Should extract component instantiations."""
        parser = AtopileModuleParser()
        ato_path = mock_atopile_project / "elec" / "src" / "sensor.ato"
        modules = parser.parse_file(ato_path)

        # SensorBoard should have submodule instantiations
        sensor_board = modules["SensorBoard"]
        assert "power" in sensor_board.components
        assert "mcu" in sensor_board.components
        assert "sensor" in sensor_board.components

    def test_infer_module_type(self):
        """Should infer module type from name."""
        parser = AtopileModuleParser()

        assert parser._infer_module_type("PowerSupply") == "power"
        assert parser._infer_module_type("TempSensor") == "sensor"
        assert parser._infer_module_type("ESP32Module") == "mcu"
        assert parser._infer_module_type("USBConnector") == "connector"
        assert parser._infer_module_type("RandomModule") is None

    def test_parse_empty_file(self, tmp_path):
        """Should handle empty file."""
        parser = AtopileModuleParser()
        empty_file = tmp_path / "empty.ato"
        empty_file.write_text("")

        modules = parser.parse_file(empty_file)
        assert modules == {}

    def test_parse_nonexistent_file(self, tmp_path):
        """Should handle nonexistent file."""
        parser = AtopileModuleParser()
        modules = parser.parse_file(tmp_path / "nonexistent.ato")
        assert modules == {}

    def test_parse_content_with_comments(self):
        """Should skip comments."""
        parser = AtopileModuleParser()
        content = """
# This is a comment
module TestModule:
    # Another comment
    comp = new Component
"""
        modules = parser.parse_content(content)
        assert "TestModule" in modules


class TestDetectBoardSource:
    """Tests for detect_board_source function."""

    def test_detect_kicad_file(self, mock_kicad_board):
        """Should detect .kicad_pcb file."""
        source_type, path = detect_board_source(mock_kicad_board)
        assert source_type == "kicad"
        assert path == mock_kicad_board

    def test_detect_atopile_project(self, mock_atopile_project):
        """Should detect atopile project directory."""
        # Create a mock board file so the path exists
        board_path = mock_atopile_project / "elec" / "layout" / "default" / "sensor.kicad_pcb"
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board_path.write_text("(kicad_pcb)")

        source_type, path = detect_board_source(mock_atopile_project)
        assert source_type == "atopile"
        assert path == board_path

    def test_detect_directory_with_kicad(self, tmp_path):
        """Should find .kicad_pcb in directory."""
        board_file = tmp_path / "myboard.kicad_pcb"
        board_file.write_text("(kicad_pcb)")

        source_type, path = detect_board_source(tmp_path)
        assert source_type == "kicad"
        assert path == board_file

    def test_detect_invalid_path(self, tmp_path):
        """Should raise error for unrecognized path."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="Cannot determine board source"):
            detect_board_source(empty_dir)


class TestComponentMetadataApplication:
    """Tests for applying component metadata to boards."""

    def test_apply_component_values(self, mock_atopile_project):
        """Should apply metadata to board components."""
        loader = AtopileProjectLoader(mock_atopile_project)

        # Create a mock board with components
        board = Board(name="test")
        board.add_component(Component(reference="C1", footprint="C_0402"))
        board.add_component(Component(reference="U1", footprint="QFN-56"))

        # Apply metadata
        loader._apply_component_metadata(board)

        # Check values were applied
        assert board.components["C1"].properties.get("mpn") == "GRM155R71C104KA88D"
        assert board.components["C1"].properties.get("package") == "0402"
        assert board.components["U1"].properties.get("mpn") == "ESP32-S3-WROOM-1"


class TestConstraintInference:
    """Tests for constraint inference from atopile patterns."""

    def test_infer_decoupling_constraints(self, mock_atopile_project):
        """Should infer proximity constraints for decoupling caps."""
        loader = AtopileProjectLoader(mock_atopile_project)

        # Create board with decoupling cap near IC
        from atoplace.board.abstraction import Board, Component, Net, Pad

        board = Board(name="test")

        # Add IC with power pin
        u1 = Component(reference="U1", footprint="QFN-56", x=50, y=50)
        u1.pads.append(Pad(number="1", x=0, y=0, width=0.5, height=0.5, net="VCC"))
        board.add_component(u1)

        # Add decoupling cap
        c1 = Component(reference="C1", footprint="C_0402", value="100nF", x=52, y=50)
        c1.pads.append(Pad(number="1", x=0, y=0, width=0.3, height=0.3, net="VCC"))
        board.add_component(c1)

        # Add net
        net = Net(name="VCC")
        net.add_connection("U1", "1")
        net.add_connection("C1", "1")
        board.add_net(net)

        # Infer constraints
        constraints = loader.infer_constraints(board)

        # Should find decoupling constraint
        assert len(constraints) >= 1
        assert any("decoupling" in c[1].lower() for c in constraints)


class TestModuleHierarchyDataClass:
    """Tests for ModuleHierarchy dataclass."""

    def test_create_hierarchy(self):
        """Should create module hierarchy."""
        module = ModuleHierarchy(
            name="PowerSupply",
            module_type="power",
            components=["regulator", "c_in", "c_out"],
        )

        assert module.name == "PowerSupply"
        assert module.module_type == "power"
        assert len(module.components) == 3

    def test_nested_hierarchy(self):
        """Should support nested modules."""
        parent = ModuleHierarchy(name="MainBoard")
        child = ModuleHierarchy(name="PowerSection", parent="MainBoard")
        parent.submodules.append(child)

        assert child.parent == "MainBoard"
        assert len(parent.submodules) == 1


class TestComponentMetadataDataClass:
    """Tests for ComponentMetadata dataclass."""

    def test_create_metadata(self):
        """Should create component metadata."""
        meta = ComponentMetadata(
            reference="C1",
            mpn="GRM155R71C104KA88D",
            value="100nF",
            package="0402",
            manufacturer="Murata",
            description="Ceramic capacitor",
        )

        assert meta.reference == "C1"
        assert meta.value == "100nF"
        assert meta.manufacturer == "Murata"

    def test_optional_fields(self):
        """Should allow optional fields."""
        meta = ComponentMetadata(reference="R1")

        assert meta.mpn is None
        assert meta.value is None
