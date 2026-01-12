# Freerouting Integration Specification

## Overview

[Freerouting](https://github.com/freerouting/freerouting) is an advanced open-source PCB autorouter that works with any EDA tool supporting the Specctra DSN format. AtoPlace integrates Freerouting to provide automated routing after placement optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATOPLACE CORE                           │
│                                                                 │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │   KiCad     │     │   Freerouting│     │   KiCad         │  │
│  │  PCB Board  │     │     JAR      │     │  PCB Board      │  │
│  └──────┬──────┘     └──────┬───────┘     └──────┬──────────┘  │
│         │                   │                     │             │
│         ▼                   ▼                     ▼             │
│    Export DSN          Run JAR             Import SES          │
│    (pcbnew API)    (subprocess/Docker)    (pcbnew API)         │
└─────────────────────────────────────────────────────────────────┘
```

## File Format Flow

```
.kicad_pcb  ->  ExportSpecctraDSN()  ->  .dsn
                                            |
                                    Freerouting JAR
                                            |
                                            v
.kicad_pcb  <-  ImportSpecctraSession() <-  .ses
```

**DSN (Design Specctra Notation):** ASCII format containing board outline, components, pads, nets, and design rules.

**SES (Session File):** Contains routing results (wires and vias) to import back into the PCB.

## Freerouting Runner

```python
# atoplace/routing/freerouting.py

from pathlib import Path
from dataclasses import dataclass
import subprocess
import tempfile
import shutil
import json

@dataclass
class RoutingResult:
    """Result of a Freerouting run."""
    success: bool
    completion_percentage: float
    routed_nets: int
    total_nets: int
    unrouted_nets: list[str]
    via_count: int
    total_wire_length: float  # mm
    passes_completed: int
    duration_seconds: float
    drc_errors: int
    score: dict | None  # Freerouting 2.1+ scoring

class FreeroutingRunner:
    """
    Run Freerouting autorouter on KiCad boards.

    Supports three execution modes:
    1. Local JAR (bundled or system)
    2. Docker container
    3. Freerouting API (cloud)
    """

    # Bundled JAR version
    BUNDLED_VERSION = "2.1.0"
    JAR_NAME = f"freerouting-{BUNDLED_VERSION}.jar"

    def __init__(
        self,
        mode: str = "jar",  # "jar", "docker", "api"
        jar_path: Path | None = None,
        java_path: str = "java",
        docker_image: str = "ghcr.io/freerouting/freerouting:2.1.0"
    ):
        self.mode = mode
        self.jar_path = jar_path or self._find_bundled_jar()
        self.java_path = java_path
        self.docker_image = docker_image

    def _find_bundled_jar(self) -> Path:
        """Find bundled Freerouting JAR."""
        # Check package data directory
        pkg_dir = Path(__file__).parent / "data"
        bundled = pkg_dir / self.JAR_NAME
        if bundled.exists():
            return bundled
        raise FileNotFoundError(
            f"Freerouting JAR not found. Install with: "
            f"atoplace install-freerouting"
        )

    def route(
        self,
        board_path: Path,
        max_passes: int = 20,
        num_threads: int | None = None,
        ignore_net_classes: list[str] | None = None,
        timeout_seconds: int = 600
    ) -> RoutingResult:
        """
        Route a KiCad board using Freerouting.

        Args:
            board_path: Path to .kicad_pcb file
            max_passes: Maximum optimization passes
            num_threads: Thread count (default: CPU count - 1)
            ignore_net_classes: Net classes to skip (e.g., ["GND", "VCC"])
            timeout_seconds: Maximum routing time

        Returns:
            RoutingResult with completion stats
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dsn_path = tmp / "board.dsn"
            ses_path = tmp / "board.ses"
            result_path = tmp / "result.json"

            # Export to DSN
            self._export_dsn(board_path, dsn_path)

            # Run Freerouting
            if self.mode == "jar":
                self._run_jar(
                    dsn_path, ses_path, result_path,
                    max_passes, num_threads, ignore_net_classes,
                    timeout_seconds
                )
            elif self.mode == "docker":
                self._run_docker(
                    dsn_path, ses_path, result_path,
                    max_passes, num_threads, ignore_net_classes,
                    timeout_seconds
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # Import session back to KiCad
            if ses_path.exists():
                self._import_session(board_path, ses_path)

            # Parse results
            return self._parse_result(result_path, ses_path)

    def _export_dsn(self, board_path: Path, dsn_path: Path):
        """Export KiCad board to Specctra DSN format."""
        try:
            import pcbnew
            board = pcbnew.LoadBoard(str(board_path))
            pcbnew.ExportSpecctraDSN(board, str(dsn_path))
        except ImportError:
            # Fallback: use KiCad CLI
            subprocess.run([
                "kicad-cli", "pcb", "export", "specctra",
                str(board_path), "-o", str(dsn_path)
            ], check=True)

    def _import_session(self, board_path: Path, ses_path: Path):
        """Import Specctra session back to KiCad board."""
        try:
            import pcbnew
            board = pcbnew.LoadBoard(str(board_path))
            pcbnew.ImportSpecctraSession(board, str(ses_path))
            pcbnew.SaveBoard(str(board_path), board)
        except ImportError:
            # Fallback: use KiCad CLI
            subprocess.run([
                "kicad-cli", "pcb", "import", "specctra",
                str(board_path), "-i", str(ses_path)
            ], check=True)

    def _run_jar(
        self,
        dsn_path: Path,
        ses_path: Path,
        result_path: Path,
        max_passes: int,
        num_threads: int | None,
        ignore_net_classes: list[str] | None,
        timeout: int
    ):
        """Run Freerouting JAR."""
        cmd = [
            self.java_path, "-jar", str(self.jar_path),
            "-de", str(dsn_path),
            "-do", str(ses_path),
            "-mp", str(max_passes),
            "-dr", str(result_path),  # JSON result output
            "--gui.enabled=false"  # Headless mode
        ]

        if num_threads:
            cmd.extend(["-mt", str(num_threads)])

        if ignore_net_classes:
            cmd.extend(["-inc", ",".join(ignore_net_classes)])

        subprocess.run(cmd, timeout=timeout, check=True)

    def _run_docker(
        self,
        dsn_path: Path,
        ses_path: Path,
        result_path: Path,
        max_passes: int,
        num_threads: int | None,
        ignore_net_classes: list[str] | None,
        timeout: int
    ):
        """Run Freerouting in Docker container."""
        work_dir = dsn_path.parent

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/work",
            self.docker_image,
            "-de", "/work/board.dsn",
            "-do", "/work/board.ses",
            "-mp", str(max_passes),
            "--gui.enabled=false"
        ]

        if num_threads:
            cmd.extend(["-mt", str(num_threads)])

        if ignore_net_classes:
            cmd.extend(["-inc", ",".join(ignore_net_classes)])

        subprocess.run(cmd, timeout=timeout, check=True)

    def _parse_result(
        self,
        result_path: Path,
        ses_path: Path
    ) -> RoutingResult:
        """Parse Freerouting result JSON and session file."""
        # Default result
        result = RoutingResult(
            success=ses_path.exists(),
            completion_percentage=0.0,
            routed_nets=0,
            total_nets=0,
            unrouted_nets=[],
            via_count=0,
            total_wire_length=0.0,
            passes_completed=0,
            duration_seconds=0.0,
            drc_errors=0,
            score=None
        )

        # Parse JSON result if available (Freerouting 2.1+)
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
                result.completion_percentage = data.get("completion", 0) * 100
                result.routed_nets = data.get("routed_nets", 0)
                result.total_nets = data.get("total_nets", 0)
                result.unrouted_nets = data.get("unrouted", [])
                result.via_count = data.get("via_count", 0)
                result.passes_completed = data.get("passes", 0)
                result.duration_seconds = data.get("duration_ms", 0) / 1000
                result.score = data.get("score")

        return result
```

## Net Class Assignment

Before routing, AtoPlace assigns appropriate net classes based on signal type:

```python
# atoplace/routing/net_classes.py

from dataclasses import dataclass
from atoplace.board import Board, Net

@dataclass
class NetClass:
    """PCB net class definition."""
    name: str
    clearance: float  # mm
    track_width: float  # mm
    via_diameter: float  # mm
    via_drill: float  # mm
    diff_pair_width: float | None = None
    diff_pair_gap: float | None = None

# Standard net classes
DEFAULT_NET_CLASSES = {
    "Default": NetClass("Default", 0.2, 0.25, 0.8, 0.4),
    "Power": NetClass("Power", 0.3, 0.5, 1.0, 0.5),
    "GND": NetClass("GND", 0.3, 0.5, 1.0, 0.5),
    "Signal": NetClass("Signal", 0.2, 0.2, 0.8, 0.4),
    "USB": NetClass("USB", 0.2, 0.2, 0.6, 0.3, 0.2, 0.15),
    "HighSpeed": NetClass("HighSpeed", 0.15, 0.15, 0.6, 0.3),
}

class NetClassAssigner:
    """Automatically assign net classes based on net properties."""

    def __init__(self, net_classes: dict[str, NetClass] | None = None):
        self.net_classes = net_classes or DEFAULT_NET_CLASSES

    def assign(self, board: Board):
        """Assign net classes to all nets on the board."""
        for net in board.nets:
            net_class = self._classify_net(net, board)
            net.net_class = net_class.name

    def _classify_net(self, net: Net, board: Board) -> NetClass:
        """Determine appropriate net class for a net."""
        name = net.name.upper()

        # Power nets
        if name in ("VCC", "VDD", "VBUS", "5V", "3V3", "12V") or \
           name.startswith("V") and name[1:].replace(".", "").isdigit():
            return self.net_classes["Power"]

        # Ground nets
        if name in ("GND", "AGND", "DGND", "PGND", "VSS"):
            return self.net_classes["GND"]

        # USB differential pairs
        if "USB" in name and ("D+" in name or "D-" in name or
                              "DP" in name or "DM" in name):
            return self.net_classes["USB"]

        # High-speed signals (clock, data buses)
        if any(kw in name for kw in ("CLK", "SCK", "MISO", "MOSI", "SDA", "SCL")):
            return self.net_classes["HighSpeed"]

        return self.net_classes["Default"]
```

## Differential Pair Detection

```python
# atoplace/routing/diff_pairs.py

from dataclasses import dataclass
from atoplace.board import Board, Net
import re

@dataclass
class DifferentialPair:
    """A differential signal pair."""
    name: str
    positive_net: Net
    negative_net: Net
    impedance: float | None = None  # Target impedance in ohms

class DiffPairDetector:
    """Detect differential pairs from net names."""

    # Common differential pair naming patterns
    PATTERNS = [
        # USB: D+/D-, DP/DM, USB_P/USB_N
        (r"(.*)D\+$", r"\1D-"),
        (r"(.*)DP$", r"\1DM"),
        (r"(.*)_P$", r"\1_N"),
        (r"(.*)_\+$", r"\1_-"),
        (r"(.*)\+$", r"\1-"),
        # LVDS, Ethernet, etc.
        (r"(.*)_P(\d*)$", r"\1_N\2"),
        (r"(.*)P(\d+)$", r"\1N\2"),
    ]

    def detect(self, board: Board) -> list[DifferentialPair]:
        """Find all differential pairs on the board."""
        pairs = []
        net_names = {net.name: net for net in board.nets}
        matched = set()

        for net in board.nets:
            if net.name in matched:
                continue

            for pos_pattern, neg_pattern in self.PATTERNS:
                match = re.match(pos_pattern, net.name)
                if match:
                    # Construct negative net name
                    neg_name = re.sub(pos_pattern, neg_pattern, net.name)
                    if neg_name in net_names:
                        pair = DifferentialPair(
                            name=match.group(1) or net.name,
                            positive_net=net,
                            negative_net=net_names[neg_name]
                        )
                        pairs.append(pair)
                        matched.add(net.name)
                        matched.add(neg_name)
                        break

        return pairs
```

## CLI Integration

```bash
# Route a board
atoplace route board.kicad_pcb

# With options
atoplace route board.kicad_pcb \
  --max-passes 30 \
  --threads 4 \
  --ignore-nets GND,VCC \
  --timeout 300

# Route after placement in one command
atoplace place-and-route board.kicad_pcb \
  --constraints "USB on left edge" \
  --dfm jlcpcb

# Use Docker instead of local JAR
atoplace route board.kicad_pcb --mode docker
```

## Integration with Placement Workflow

```python
# atoplace/workflow.py

from atoplace.board import Board
from atoplace.placement import ForceDirectedRefiner
from atoplace.routing import FreeroutingRunner, NetClassAssigner, DiffPairDetector
from atoplace.validation import ConfidenceScorer, DRCChecker

def place_and_route(
    board_path: str,
    constraints: list[str] | None = None,
    dfm_profile: str = "jlcpcb",
    max_routing_passes: int = 20
) -> dict:
    """Complete placement and routing workflow."""

    # Load board
    board = Board.from_kicad(board_path)

    # Phase 1: Placement
    refiner = ForceDirectedRefiner(dfm_profile=dfm_profile)
    refiner.refine(board, constraints=constraints)

    # Pre-route validation
    scorer = ConfidenceScorer()
    pre_route_report = scorer.assess(board)

    if pre_route_report.overall_score < 0.7:
        return {
            "status": "placement_failed",
            "confidence": pre_route_report.overall_score,
            "flags": pre_route_report.flags
        }

    # Phase 2: Net class assignment
    assigner = NetClassAssigner()
    assigner.assign(board)

    # Phase 3: Differential pair detection
    detector = DiffPairDetector()
    diff_pairs = detector.detect(board)

    # Save before routing
    board.to_kicad(board_path)

    # Phase 4: Routing
    router = FreeroutingRunner()
    routing_result = router.route(
        board_path,
        max_passes=max_routing_passes,
        ignore_net_classes=["GND", "VCC"]  # Pour these as planes
    )

    # Phase 5: Post-route validation
    board = Board.from_kicad(board_path)  # Reload with routes
    drc = DRCChecker(dfm_profile=dfm_profile)
    drc_result = drc.check(board)

    return {
        "status": "success" if routing_result.success else "partial",
        "placement_confidence": pre_route_report.overall_score,
        "routing_completion": routing_result.completion_percentage,
        "unrouted_nets": routing_result.unrouted_nets,
        "via_count": routing_result.via_count,
        "diff_pairs_detected": len(diff_pairs),
        "drc_errors": drc_result.error_count,
        "drc_warnings": drc_result.warning_count
    }
```

## Freerouting Installation

```python
# atoplace/routing/install.py

from pathlib import Path
import urllib.request
import hashlib

FREEROUTING_VERSION = "2.1.0"
FREEROUTING_URL = (
    f"https://github.com/freerouting/freerouting/releases/download/"
    f"v{FREEROUTING_VERSION}/freerouting-{FREEROUTING_VERSION}.jar"
)
FREEROUTING_SHA256 = "..."  # Checksum for verification

def install_freerouting(target_dir: Path | None = None) -> Path:
    """Download and install Freerouting JAR."""
    if target_dir is None:
        target_dir = Path(__file__).parent / "data"

    target_dir.mkdir(parents=True, exist_ok=True)
    jar_path = target_dir / f"freerouting-{FREEROUTING_VERSION}.jar"

    if jar_path.exists():
        # Verify existing file
        if _verify_checksum(jar_path, FREEROUTING_SHA256):
            return jar_path

    # Download
    print(f"Downloading Freerouting {FREEROUTING_VERSION}...")
    urllib.request.urlretrieve(FREEROUTING_URL, jar_path)

    # Verify
    if not _verify_checksum(jar_path, FREEROUTING_SHA256):
        jar_path.unlink()
        raise RuntimeError("Checksum verification failed")

    print(f"Installed to {jar_path}")
    return jar_path

def _verify_checksum(path: Path, expected: str) -> bool:
    """Verify file SHA256 checksum."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected
```

## Implementation Plan

### Phase 3A: Basic Freerouting Integration
1. `FreeroutingRunner` with JAR execution
2. DSN export via pcbnew API
3. SES import back to KiCad
4. Basic CLI command `atoplace route`

### Phase 3B: Smart Routing
1. `NetClassAssigner` - automatic net classification
2. `DiffPairDetector` - USB, LVDS, Ethernet pairs
3. Pre-route net class assignment
4. Post-route DRC integration

### Phase 3C: Advanced Features
1. Docker execution mode
2. Freerouting JAR bundling and auto-install
3. Progress streaming for long routes
4. Routing result scoring (Freerouting 2.1+)
5. `place-and-route` combined workflow

## Testing Strategy

```python
# tests/test_freerouting.py

import pytest
from pathlib import Path
from atoplace.routing import FreeroutingRunner, NetClassAssigner, DiffPairDetector

@pytest.fixture
def simple_board(tmp_path):
    """Create a simple test board for routing."""
    # Create minimal board with a few components and nets
    ...

def test_dsn_export(simple_board):
    """Test DSN file generation."""
    runner = FreeroutingRunner()
    dsn_path = simple_board.parent / "test.dsn"
    runner._export_dsn(simple_board, dsn_path)
    assert dsn_path.exists()
    content = dsn_path.read_text()
    assert "(pcb" in content

def test_net_class_assignment(simple_board):
    """Test automatic net class assignment."""
    board = Board.from_kicad(simple_board)
    assigner = NetClassAssigner()
    assigner.assign(board)

    vcc_net = next(n for n in board.nets if n.name == "VCC")
    assert vcc_net.net_class == "Power"

def test_diff_pair_detection():
    """Test differential pair detection."""
    detector = DiffPairDetector()
    # Create mock board with USB D+/D- nets
    board = create_mock_board_with_usb()
    pairs = detector.detect(board)
    assert len(pairs) == 1
    assert pairs[0].positive_net.name == "USB_D+"

@pytest.mark.integration
def test_full_routing(simple_board):
    """Integration test: full routing workflow."""
    runner = FreeroutingRunner()
    result = runner.route(simple_board, max_passes=5)
    assert result.success
    assert result.completion_percentage > 90
```

## Error Handling

```python
class RoutingError(Exception):
    """Base class for routing errors."""
    pass

class FreeroutingNotFoundError(RoutingError):
    """Freerouting JAR not found."""
    pass

class DSNExportError(RoutingError):
    """Failed to export DSN file."""
    pass

class RoutingTimeoutError(RoutingError):
    """Routing exceeded timeout."""
    pass

class RoutingFailedError(RoutingError):
    """Routing completed but with errors."""
    def __init__(self, result: RoutingResult):
        self.result = result
        super().__init__(
            f"Routing failed: {result.completion_percentage}% complete, "
            f"{len(result.unrouted_nets)} unrouted nets"
        )
```

