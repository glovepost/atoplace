<p align="center">
  <a href="#"><img src="images/atoplace.png" alt="atoplace" width="420"></a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
  <img src="https://img.shields.io/badge/KiCad-8.0%2B-blue" alt="KiCad Support">
</p>

<p align="center">
  <strong>The AI Pair Designer for Professional PCB Layout</strong>
</p>

**atoplace** is an intelligent orchestration layer for PCB design that bridges the gap between schematic and physical layout. It automates the tedious 80% of design—placement optimization and DFM validation—while strictly adhering to "Manhattan" aesthetics and Signal Integrity (SI) best practices.

Designed to work seamlessly with **[atopile](https://atopile.io)** and **[KiCad](https://kicad.org)**.

<p align="center">
  <img src="images/placement_demo.gif" alt="atoplace placement visualization" width="720">
</p>

## 🚀 Why atoplace?

PCB layout automation has historically been "black box" and "messy"—producing organic, unreadable layouts that professional engineers reject. atoplace takes a different approach:

*   **Human-Grade Aesthetics**: We don't just minimize wirelength. We enforce **grids**, **alignment**, and **orthogonal routing** so the result looks like *you* designed it.
*   **Physics-First**: We model high-degree nets (GND/VCC) correctly to prevent component collapse, and prioritize critical signals (USB, RF) before general routing.
*   **Transparent & Interactive**: No lock-in. The source of truth is always your `.kicad_pcb` file. You can take over manually at any second.

## ✨ Key Features

- **🧩 Manhattan Legalizer**: Transforms "organic" force-directed placements into professional, grid-snapped, and row-aligned layouts using PCA axis detection and Abacus-style overlap resolution.
- **🧠 Intelligent Placement**: Force-directed annealing engine with a **Star Model** for stable high-degree net (GND/VCC) handling and adaptive damping for oscillation control.
- **🚀 A* Routing**: High-performance, deterministic geometric router using a **Greedy Multiplier** ($w=2-3$) and **Spatial Hash Indexing** for O(~1) collision detection.
- **🔍 Confidence Scoring**: Automated assessment of your board's routability, signal integrity risks, and DFM compliance.
- **💬 Natural Language Control**: "Move the USB connector to the left edge", "Align these capacitors", "Keep the crystal near the MCU".
- **🔌 Atopile Native**: First-class support for `atopile` projects with `ato-lock.yaml` parsing and module-aware grouping.

## 🛠️ Installation

```bash
pip install atoplace
```

### Requirements
- **Python 3.10+**
- **KiCad 8.0+** (atoplace uses the `pcbnew` Python API)

> **Pro Tip:** atoplace works best when run using the Python interpreter bundled with KiCad:

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

### 2. Route the Board
Route all nets using the internal A* geometric planner:
```bash
atoplace route board.kicad_pcb --visualize
```

### 3. Validate Layout
Check your board against DFM rules (JLCPCB, OSH Park, etc.):
```bash
atoplace validate board.kicad_pcb --dfm jlcpcb_standard
```

### 4. Interactive Mode
Refine your design using natural language:
```bash
atoplace interactive board.kicad_pcb
# > "Rotate U1 45 degrees"
# > "Move C1 closer to U1"
# > "Save"
```

## 🗺️ Roadmap

- **Milestone A (Q1 2026): Solid Foundation** ✅
  - [x] Manhattan Legalizer (Grid snapping & PCA Alignment).
  - [x] Physics Engine scaling (Star Model & Adaptive Damping).
  - [x] Atopile `ato-lock.yaml` and module hierarchy integration.
- **Milestone B (Q1-Q2 2026): Routing & Persistence** 🚧
  - [x] **A* Geometric Planner** (Greedy Multiplier & Spatial Indexing).
  - [ ] `atoplace.lock` Sidecar Persistence for Atopile.
  - [ ] BGA/QFN Fanout Generator.
  - [ ] Differential Pair Path Planning.
- **Milestone C (Q2 2026): Professional Agent** 🔮
  - [ ] MCP Server for full conversational design.
  - [ ] Deep Signal Integrity Analysis (Crosstalk/Impedance).
  - [ ] Automated Manufacturing Outputs (Gerbers/BOM/PNP).

## 📂 Architecture

```
atoplace/
├── board/          # Board abstraction & KiCad/Atopile adapters
├── placement/      # Force-directed physics & Manhattan Legalizer
│   ├── force_directed.py   # Physics Engine (Star Model)
│   ├── legalizer.py        # Manhattan Pipeline (REQ-P-03)
│   └── module_detector.py  # Hierarchy Analysis
├── routing/        # A* Geometric Router
│   ├── astar_router.py     # Core A* with Greedy Multiplier
│   ├── spatial_index.py    # O(~1) collision detection
│   └── obstacle_map.py     # Obstacle generation
├── nlp/            # Natural Language & Intent Engine
├── validation/     # Confidence Scorer & DFM/DRC Checker
├── mcp/            # MCP Server (Planned)
└── cli.py          # CLI entry point
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

*   **[atopile](https://github.com/atopile/atopile)**: The declarative language that makes code-driven hardware possible.
*   **[KiCad](https://kicad.org)**: The open-source EDA standard we build upon.
*   **[Freerouting](https://github.com/freerouting/freerouting)**: The open-source autorouting engine.
