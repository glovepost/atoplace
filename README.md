<p align="center">
  <img src="images/atoplace.png" alt="AtoPlace" width="420">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
  <img src="https://img.shields.io/badge/KiCad-6.0%2B-blue" alt="KiCad Support">
</p>

<p align="center">
  <strong>The AI Pair Designer for Professional PCB Layout</strong>
</p>

---

**AtoPlace** is an intelligent orchestration layer for PCB design that bridges the gap between schematic and physical layout. It automates the tedious 80% of design—placement optimization and DFM validation—while strictly adhering to "Manhattan" aesthetics and Signal Integrity (SI) best practices.

Designed to work seamlessly with **[atopile](https://atopile.io)** and **[KiCad](https://kicad.org)**.

## 🚀 Why AtoPlace?

PCB layout automation has historically been "black box" and "messy"—producing organic, unreadable layouts that professional engineers reject. AtoPlace takes a different approach:

*   **Human-Grade Aesthetics**: We don't just minimize wirelength. We enforce **grids**, **alignment**, and **orthogonal routing** so the result looks like *you* designed it.
*   **Physics-First**: We model high-degree nets (GND/VCC) correctly to prevent component collapse, and prioritize critical signals (USB, RF) before general routing.
*   **Transparent & Interactive**: No lock-in. The source of truth is always your `.kicad_pcb` file. You can take over manually at any second.

## ✨ Key Features

- **🧩 Manhattan Legalizer**: Transforms "organic" force-directed placements into professional, grid-snapped, and row-aligned layouts.
- **🧠 Intelligent Placement**: Force-directed annealing engine with a "Star Model" for stable power/ground net handling.
- **🔍 Confidence Scoring**: Automated assessment of your board's routability, signal integrity risks, and DFM compliance.
- **💬 Natural Language Control**: "Move the USB connector to the left edge", "Align these capacitors", "Keep the crystal near the MCU".
- **🔌 Atopile Native**: First-class support for `atopile` projects with module-aware grouping constraints.

## 🛠️ Installation

```bash
pip install atoplace
```

Alternative (no install): run from a local clone:
```bash
git clone https://github.com/glovepost/atoplace
cd atoplace
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m atoplace.cli --help
```

> **Note:** AtoPlace requires access to KiCad's Python API (`pcbnew`).
> You typically need to run it using the Python interpreter bundled with KiCad:

**macOS:**
```bash
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install atoplace
```

**Windows:**
```powershell
"C:\Program Files\KiCad\8.0\bin\python.exe" -m pip install atoplace
```

## ⚡ Quick Start

### 1. Optimize Placement
Automatically place components with "Manhattan" legalization:
```bash
atoplace place board.kicad_pcb --grid 0.5 --constraints "USB on left edge"
```

### 2. Validate Layout
Check your board against DFM rules (JLCPCB, OSH Park, etc.):
```bash
atoplace validate board.kicad_pcb --dfm jlcpcb_standard
```

### 3. Interactive Mode
Refine your design using natural language:
```bash
atoplace interactive board.kicad_pcb
# > "Rotate U1 45 degrees"
# > "Move C1 closer to U1"
# > "Save"
```

## 🧠 Python API

For deeper integration or custom workflows:

```python
from atoplace.board import Board
from atoplace.placement import ForceDirectedRefiner, PlacementLegalizer

# 1. Load Board
board = Board.from_kicad("board.kicad_pcb")

# 2. Physics Refinement (Global Optimization)
refiner = ForceDirectedRefiner(board)
refiner.refine()

# 3. Legalization (Manhattan Polish)
legalizer = PlacementLegalizer(board)
legalizer.legalize()

# 4. Save Result
board.to_kicad("board_polished.kicad_pcb")
```

## 🗺️ Roadmap

- **Milestone A (Q1 2026): Solid Foundation** ✅
  - [x] Manhattan Legalizer (Grid snapping & Alignment).
  - [x] Physics Engine scaling fixes ($O(N)$ Star Model).
  - [x] Rotated Pad geometry modeling.
- **Milestone B (Q1-Q2 2026): Routing Assistant** 🚧
  - [ ] Critical Path Geometric Planner (A* Dual-Grid).
  - [ ] BGA/QFN Fanout Generator.
  - [ ] Atopile `ato-lock.yaml` persistence.
- **Milestone C (Q2 2026): Professional Agent** 🔮
  - [ ] MCP Server for full conversational design.
  - [ ] Automated Manufacturing Outputs (Gerbers/BOM/PNP).

## 📂 Architecture

```
atoplace/
├── board/          # Board abstraction & KiCad/Atopile adapters
├── placement/      # Force-directed physics & Manhattan Legalizer
│   ├── force_directed.py   # Physics Engine (Star Model)
│   ├── legalizer.py        # Manhattan Pipeline (REQ-P-03)
│   ├── module_detector.py  # Hierarchy Analysis
│   └── constraints.py      # Placement Constraints
├── nlp/            # Natural Language & Intent Engine
├── routing/        # Routing integration (planned: Freerouting)
├── validation/     # Confidence Scorer & DFM/DRC Checker
├── dfm/            # Fab-specific design rules
├── mcp/            # MCP Server for Claude/LLM integration
└── cli.py          # CLI entry point
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

*   **[atopile](https://github.com/atopile/atopile)**: The declarative language that makes code-driven hardware possible.
*   **[KiCad](https://kicad.org)**: The open-source EDA standard we build upon.
*   **[Freerouting](https://github.com/freerouting/freerouting)**: The open-source autorouting engine.
