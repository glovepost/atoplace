<p align="center">
  <a href="#"><img src="images/atoplace.png" alt="atoplace" width="420"></a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
  <img src="https://img.shields.io/badge/KiCad-8.0%2B-blue" alt="KiCad Support">
  <img src="https://img.shields.io/badge/KiCad%209-Live%20IPC-green" alt="KiCad 9 Live IPC">
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
- **🔌 Atopile Native**: First-class support for `atopile` projects with `ato-lock.yaml` parsing, module-aware grouping, and **`atoplace.lock` sidecar persistence** to preserve placements across rebuilds.
- **📊 Interactive Visualization**: Unified SVG delta viewer with real-time playback, layer toggles, grid customization, force vector display, routing trace visualization, and A* debug mode for step-by-step algorithm analysis.

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

### 5. MCP Server (LLM Integration)
Expose atoplace tools to AI agents via the Model Context Protocol:
```bash
atoplace mcp --launch
```

See [MCP Server](#-mcp-server) section below for details.

## 🗺️ Roadmap

- **Milestone A (Q1 2026): Solid Foundation** ✅
  - [x] Manhattan Legalizer (Grid snapping & PCA Alignment).
  - [x] Physics Engine scaling (Star Model & Adaptive Damping).
  - [x] Atopile `ato-lock.yaml` and module hierarchy integration.
- **Milestone B (Q1-Q2 2026): Routing & Persistence** ✅
  - [x] **A* Geometric Planner** (Greedy Multiplier & Spatial Indexing).
  - [x] **MCP Server** with 40+ tools for LLM agent integration.
  - [x] **Live KiCad IPC** via kipy for real-time component manipulation (KiCad 9+).
  - [x] **`atoplace.lock` Sidecar Persistence** for atopile projects.
  - [x] **Unified Visualization** (SVG delta viewer with routing support).
  - [x] **BGA/QFN Fanout Generator** (dogbone & via-in-pad strategies).
  - [x] **Pin Swap Optimization** (bipartite matching for crossing reduction).
- **Milestone C (Q2 2026): Professional Agent** 🚧
  - [x] Differential Pair Detection (auto-detect from net names).
  - [ ] Differential Pair Coupled Routing.
  - [ ] Deep Signal Integrity Analysis (Crosstalk/Impedance).
  - [ ] Automated Manufacturing Outputs (Gerbers/BOM/PNP).
  - [ ] Multi-board design support.

## 📂 Architecture

```
atoplace/
├── board/          # Board abstraction & KiCad/Atopile adapters
│   ├── abstraction.py      # Board, Component, Net, Pad data models
│   ├── kicad_adapter.py    # KiCad pcbnew integration
│   ├── lock_file.py        # atoplace.lock sidecar persistence
│   └── atopile_adapter.py  # Atopile project loader
├── placement/      # Force-directed physics & Manhattan Legalizer
│   ├── force_directed.py   # Physics Engine (Star Model, adaptive damping)
│   ├── legalizer.py        # Manhattan Pipeline with PCA alignment
│   ├── module_detector.py  # Functional module hierarchy analysis
│   ├── constraints.py      # Placement constraints (Proximity, Edge, Zone, etc.)
│   └── visualizer.py       # SVG delta visualization with frame capture
├── routing/        # A* Geometric Router
│   ├── astar_router.py     # Core A* with Greedy Multiplier (w=2-3)
│   ├── manager.py          # Multi-phase routing pipeline
│   ├── spatial_index.py    # O(~1) collision detection via spatial hashing
│   ├── obstacle_map.py     # Obstacle generation from board
│   ├── diff_pairs.py       # Differential pair detection
│   ├── visualizer.py       # Routing visualization with A* debug
│   └── fanout/             # BGA/QFN escape routing
├── visualization/  # Unified visualization system
│   └── assets/             # External JS/CSS for HTML viewer
│       ├── svg-delta-viewer.js  # Interactive SVG playback engine
│       └── styles.css      # KiCad-inspired dark theme
├── nlp/            # Natural Language & Intent Engine
│   └── constraint_parser.py  # Parse "USB on left edge" to constraints
├── validation/     # Confidence Scorer & DFM/DRC Checker
│   ├── confidence.py       # Design quality assessment
│   └── drc.py              # Design rule checking
├── mcp/            # MCP Server for LLM Integration
│   ├── server.py           # FastMCP server with 40+ tools
│   ├── backends.py         # Backend mode detection & factory
│   ├── kipy_session.py     # Live KiCad IPC session (KiCad 9+)
│   └── context/            # Context generators (semantic grid, module map)
└── cli.py          # CLI entry point
```

## 🤖 MCP Server

atoplace includes a **Model Context Protocol (MCP)** server that exposes 40+ PCB design tools to LLM agents like Claude. This enables conversational PCB layout design with support for placement, routing, fanout generation, and design validation.

### Quick Start

Just run:
```bash
python -m atoplace.mcp.launcher
```

That's it! The launcher:
- **Auto-detects** KiCad Python on macOS, Linux, and Windows
- **Starts** the KiCad bridge (for pcbnew access)
- **Starts** the MCP server (exposes tools to LLM)
- **Manages** lifecycle and clean shutdown

### Claude Code / Claude Desktop Configuration

Add to your MCP config:

```json
{
  "mcpServers": {
    "atoplace": {
      "command": "python",
      "args": ["-m", "atoplace.mcp.launcher"]
    }
  }
}
```

> **Note**: Replace `python` with the path to your atoplace virtualenv Python if needed.

### 🔴 Live KiCad IPC Mode (KiCad 9+)

**New!** With KiCad 9+, atoplace can manipulate components in real-time without save/reload cycles. Changes appear instantly in your KiCad viewport!

**Requirements:**
- KiCad 9.0 or later (with IPC API enabled)
- `kicad-python` package: `pip install kicad-python`

**Setup:**
```bash
# Install the kipy optional dependency
pip install atoplace[kipy]

# Or install directly
pip install kicad-python
```

**Usage:**
1. Open your `.kicad_pcb` in KiCad 9+ PCB Editor
2. Start the MCP server with the kipy backend:
   ```bash
   ATOPLACE_BACKEND=kipy python -m atoplace.mcp.launcher
   ```
3. Components move instantly as you interact with the LLM!

**How it works:** The kipy backend connects directly to KiCad's IPC socket and uses the official API to update component positions in real-time. Native undo/redo (Cmd/Ctrl+Z) works seamlessly.

### Architecture

```mermaid
flowchart LR
    subgraph MCP["MCP Server (Python 3.10+)"]
        direction TB
        FastMCP[FastMCP]
        Tools[26 Layout Tools]
        Context[Context Generators]
    end

    subgraph KiPy["KiPy Backend (KiCad 9+)"]
        direction TB
        IPC[KiCad IPC API]
        LiveEdit[Real-time Updates]
        NativeUndo[Native Undo/Redo]
    end

    subgraph Bridge["KiCad Bridge (KiCad 8.x)"]
        direction TB
        pcbnew[pcbnew API]
        BoardIO[Board I/O]
        UndoRedo[Undo/Redo]
    end

    MCP <-->|"KiCad IPC\nSocket"| KiPy
    MCP <-->|"Unix Socket\nJSON-RPC"| Bridge
```

**Backend Modes:**
| Mode | KiCad Version | Real-time | How it works | Status |
|------|--------------|-----------|--------------|--------|
| `kipy` | 9.0+ | ✅ Yes | Direct IPC API connection | **Default** |
| `ipc` | 8.0+ | ❌ No | File-based bridge process | Fallback |
| `direct` | 8.0+ | ❌ No | Direct file manipulation | Fallback |

**Note:** KIPY mode is the primary backend for bleeding-edge development. It provides instant visual feedback in KiCad's viewport and native undo/redo integration. The server automatically falls back to IPC or direct mode if KIPY is unavailable.

### Available Tools (40+ total)

| Category | Tools |
|----------|-------|
| **Board Management** | `load_board`, `save_board`, `undo`, `redo` |
| **Placement Actions** | `move_absolute`, `move_relative`, `rotate`, `place_next_to`, `align_components`, `distribute_evenly`, `stack_components`, `arrange_pattern`, `cluster_around`, `lock_components` |
| **Discovery** | `find_components`, `get_board_summary`, `get_unplaced_components` |
| **Routing** | `route_board`, `route_net`, `detect_diff_pairs`, `get_routing_preview`, `analyze_pin_swaps`, `optimize_pin_swaps`, `export_pin_constraints` |
| **BGA/Fanout** | `detect_bga_components`, `fanout_component`, `fanout_all_bgas`, `get_fanout_preview` |
| **Context** | `inspect_region`, `get_semantic_grid`, `get_module_map` |
| **Validation** | `check_overlaps`, `validate_placement`, `run_drc`, `get_crossing_analysis` |
| **Optimization** | `optimize_placement`, `detect_modules`, `parse_constraint`, `get_atopile_context` |

### Environment Variables (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `ATOPLACE_BACKEND` | Backend mode: `kipy`, `ipc`, or `direct` | `kipy` (with fallback) |
| `ATOPLACE_USE_KIPY` | Enable kipy mode (`1` or `true`) | Deprecated - use `ATOPLACE_BACKEND` |
| `KICAD_PYTHON` | Override KiCad Python path | Auto-detected |
| `ATOPLACE_LOG` | Log file location | `/tmp/atoplace.log` |

**Default behavior:** The server attempts KIPY first, then falls back to IPC, then direct mode. For bleeding-edge KiCad 9+ users, no configuration is needed - KIPY mode activates automatically when KiCad is running with IPC enabled.

### Example LLM Conversation

```
User: Load the board at examples/dogtracker/layouts/default/default.kicad_pcb

Claude: [Calls load_board tool]
Board loaded successfully:
- 37 components
- 72 nets

User: Find all the capacitors and align them horizontally

Claude: [Calls find_components with query="C", filter_by="ref"]
Found 12 capacitors: C1, C2, C3...

[Calls align_components with refs=["C1","C2",...], axis="y"]
Aligned 12 capacitors along the Y axis.

User: Check if there are any overlaps

Claude: [Calls check_overlaps]
No overlapping components detected. The placement is valid.

User: Save the board

Claude: [Calls save_board]
Board saved to: examples/dogtracker/layouts/default/default.placed.kicad_pcb
```

## 🔒 Atopile Lock File Persistence

When working with **atopile** projects, placement positions can be lost when `ato build` regenerates the KiCad board. The `atoplace.lock` sidecar file solves this by persisting component positions between builds.

### How it works

After placement, atoplace automatically saves positions to a lock file next to the board:
```
my-project/
├── elec/
│   └── layout/
│       └── default/
│           ├── default.kicad_pcb
│           └── default.atoplace.lock  ← Saved positions
└── ato.yaml
```

### Lock File Format

```yaml
version: 1
created: 2026-01-14T10:30:00
build: default
components:
  U1:
    x: 125.5
    y: 80.0
    rotation: 0.0
    locked: true    # User-approved position
    module: mcu
  C1:
    x: 130.0
    y: 85.5
    locked: false   # Auto-placed, can be re-optimized
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--use-lock` | Apply saved positions before placement |
| `--only-locked` | Only restore `locked: true` positions |
| `--save-lock/--no-save-lock` | Auto-save after placement (default: on) |
| `--lock-all` | Mark all positions as user-approved |

### Workflow Example

```bash
# First placement - automatically saves lock file
atoplace place my-project/ --use-ato-modules

# After `ato build` regenerates board - restore positions
atoplace place my-project/ --use-lock

# Only restore approved positions, re-optimize others
atoplace place my-project/ --use-lock --only-locked

# Approve all current positions
atoplace place my-project/ --lock-all
```

### Merge Logic

The lock file uses **Lock > Physics** priority:
1. `locked: true` positions always preserved (user-approved)
2. `locked: false` positions used as initial hints, can be re-optimized
3. New components (not in lock file) placed by physics engine

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

*   **[atopile](https://github.com/atopile/atopile)**: The declarative language that makes code-driven hardware possible.
*   **[KiCad](https://kicad.org)**: The open-source EDA standard we build upon.
*   **[kicad-python](https://github.com/kicad/kicad-python)**: Official Python bindings for KiCad's IPC API.
*   **[Freerouting](https://github.com/freerouting/freerouting)**: The open-source autorouting engine.
