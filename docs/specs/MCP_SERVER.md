# MCP Server Integration Specification

## Overview

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) enables Claude and other LLMs to interact with AtoPlace through a standardized interface. This allows conversational PCB design workflows where users describe their intent in natural language and Claude orchestrates placement and routing operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLAUDE DESKTOP                          │
│                    or other MCP-compatible client               │
└─────────────────────────────────────────────────────────────────┘
                              │
                         MCP Protocol
                         (JSON-RPC over stdio)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     ATOPLACE MCP SERVER                         │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   TOOLS     │  │  RESOURCES  │  │       PROMPTS           │ │
│  │             │  │             │  │                         │ │
│  │ place_board │  │ board://    │  │ placement_workflow      │ │
│  │ add_const   │  │ confidence  │  │ constraint_guide        │ │
│  │ validate    │  │ dfm_rules   │  │ troubleshoot_drc        │ │
│  │ route       │  │ components  │  │                         │ │
│  │ export      │  │             │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                              │                                  │
│                    ┌─────────────────┐                         │
│                    │  ATOPLACE CORE  │                         │
│                    │  (Python API)   │                         │
│                    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## MCP Tools

Tools are functions Claude can invoke to perform actions:

```python
from mcp.server.fastmcp import FastMCP
from atoplace.board import Board
from atoplace.placement import ForceDirectedRefiner
from atoplace.validation import ConfidenceScorer

mcp = FastMCP("AtoPlace PCB Layout Server")

@mcp.tool()
def place_board(
    board_path: str,
    constraints: list[str] | None = None,
    dfm_profile: str = "jlcpcb"
) -> dict:
    """
    Run intelligent placement optimization on a KiCad board.

    Args:
        board_path: Path to .kicad_pcb file or atopile project directory
        constraints: Natural language constraints (e.g., "USB on left edge")
        dfm_profile: Manufacturing profile (jlcpcb, oshpark, pcbway)

    Returns:
        Placement result with confidence score and any flags
    """
    board = Board.from_kicad(board_path)
    # Parse and apply constraints
    # Run force-directed refinement
    # Return results with confidence report
    ...

@mcp.tool()
def add_constraint(
    board_path: str,
    constraint: str
) -> dict:
    """
    Add a placement constraint in natural language.

    Args:
        board_path: Path to the board file
        constraint: Natural language constraint (e.g., "keep C1 close to U1")

    Returns:
        Parsed constraint details and validation status
    """
    ...

@mcp.tool()
def validate_placement(board_path: str) -> dict:
    """
    Run DRC and confidence scoring on current placement.

    Returns:
        Detailed validation report with flags and suggestions
    """
    ...

@mcp.tool()
def route_board(
    board_path: str,
    max_passes: int = 20,
    ignore_nets: list[str] | None = None
) -> dict:
    """
    Run autorouting using Freerouting.

    Args:
        board_path: Path to the board file
        max_passes: Maximum optimization passes
        ignore_nets: Net classes to skip (e.g., ["GND", "VCC"])

    Returns:
        Routing result with completion percentage and any unrouted nets
    """
    ...

@mcp.tool()
def modify_placement(
    board_path: str,
    modification: str
) -> dict:
    """
    Apply a placement modification in natural language.

    Args:
        board_path: Path to the board file
        modification: Natural language command (e.g., "rotate J190 degrees")

    Returns:
        Modification result with before/after positions
    """
    ...

@mcp.tool()
def export_manufacturing(
    board_path: str,
    output_dir: str,
    fab: str = "jlcpcb"
) -> dict:
    """
    Generate manufacturing outputs (Gerbers, drill, BOM, PnP).

    Returns:
        List of generated files and any warnings
    """
    ...
```

## MCP Resources

Resources expose data that Claude can read:

```python
@mcp.resource("board://{board_path}/components")
def get_components(board_path: str) -> str:
    """List all components with positions and footprints."""
    board = Board.from_kicad(board_path)
    return format_component_list(board.components)

@mcp.resource("board://{board_path}/nets")
def get_nets(board_path: str) -> str:
    """List all nets with connected components."""
    board = Board.from_kicad(board_path)
    return format_net_list(board.nets)

@mcp.resource("board://{board_path}/confidence")
def get_confidence_report(board_path: str) -> str:
    """Get the current confidence assessment."""
    board = Board.from_kicad(board_path)
    scorer = ConfidenceScorer()
    report = scorer.assess(board)
    return format_confidence_report(report)

@mcp.resource("dfm://{profile}")
def get_dfm_rules(profile: str) -> str:
    """Get DFM rules for a specific fab profile."""
    return format_dfm_profile(get_profile(profile))

@mcp.resource("constraints://examples")
def get_constraint_examples() -> str:
    """Get examples of supported constraint patterns."""
    return CONSTRAINT_EXAMPLES
```

## MCP Prompts

Prompts provide reusable conversation templates:

```python
@mcp.prompt()
def placement_workflow() -> str:
    """Guide for complete placement workflow."""
    return """
    # PCB Placement Workflow

    1. **Load Board**: Use place_board tool with the board path
    2. **Review Components**: Check board://path/components resource
    3. **Add Constraints**: Use add_constraint for placement requirements
    4. **Run Placement**: Execute place_board with constraints
    5. **Validate**: Check confidence report and DRC results
    6. **Iterate**: Use modify_placement for adjustments
    7. **Route**: Run route_board when placement is finalized
    8. **Export**: Generate manufacturing files with export_manufacturing
    """

@mcp.prompt()
def constraint_guide() -> str:
    """Guide for writing placement constraints."""
    return """
    # Constraint Examples

    ## Proximity
    - "Keep C1 close to U1"
    - "Place decoupling caps within 3mm of MCU"

    ## Edge Placement
    - "USB connector on left edge"
    - "Power jack on bottom edge, centered"

    ## Grouping
    - "Group all bypass capacitors together"
    - "Keep analog section separate from digital"

    ## Rotation
    - "Rotate J1 90 degrees clockwise"
    - "Flip U2 to bottom layer"

    ## Fixed Position
    - "Fix LED1 at coordinates (25, 10)"
    - "Lock mounting holes in corners"
    """
```

## Server Implementation

```python
# atoplace/mcp/server.py

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json

from atoplace.board import Board
from atoplace.board.atopile_adapter import AtopileProjectLoader
from atoplace.placement import ForceDirectedRefiner, ConstraintSolver
from atoplace.validation import ConfidenceScorer, PreRouteValidator
from atoplace.routing import FreeroutingRunner
from atoplace.nlp import ConstraintParser, ModificationHandler
from atoplace.dfm import get_profile

mcp = FastMCP(
    "AtoPlace",
    description="AI-powered PCB placement and routing optimization"
)

# State management for loaded boards
_boards: dict[str, Board] = {}

def _load_board(path: str) -> Board:
    """Load board from path, detecting atopile projects."""
    p = Path(path)
    if p.is_dir() and (p / "ato.yaml").exists():
        loader = AtopileProjectLoader(p)
        return loader.load_board()
    elif p.suffix == ".kicad_pcb":
        return Board.from_kicad(p)
    else:
        raise ValueError(f"Unknown board format: {path}")

# Tools implementation
@mcp.tool()
def place_board(
    board_path: str,
    constraints: list[str] | None = None,
    dfm_profile: str = "jlcpcb"
) -> dict:
    """Run intelligent placement optimization on a PCB board."""
    board = _load_board(board_path)
    _boards[board_path] = board

    # Parse constraints
    parser = ConstraintParser()
    parsed_constraints = []
    if constraints:
        for c in constraints:
            parsed = parser.parse(c, board)
            if parsed:
                parsed_constraints.append(parsed)

    # Run placement
    refiner = ForceDirectedRefiner(
        dfm_profile=get_profile(dfm_profile)
    )
    refiner.refine(board, constraints=parsed_constraints)

    # Score result
    scorer = ConfidenceScorer()
    report = scorer.assess(board)

    # Save
    board.to_kicad(board_path)

    return {
        "status": "success",
        "confidence_score": report.overall_score,
        "flags": [f.to_dict() for f in report.flags],
        "components_placed": len(board.components)
    }

# ... additional tool implementations ...

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## Configuration

**Claude Desktop (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "atoplace": {
      "command": "python",
      "args": ["-m", "atoplace.mcp.server"],
      "env": {
        "KICAD_PATH": "/Applications/KiCad/KiCad.app"
      }
    }
  }
}
```

**Cursor (`.cursor/mcp.json`):**
```json
{
  "mcpServers": {
    "atoplace": {
      "command": "uv",
      "args": ["run", "atoplace-mcp"]
    }
  }
}
```

## Conversation Examples

**Example 1: Full Workflow**
```
User: I have an ESP32 sensor board at ~/projects/sensor-board.
      Place it with the USB-C connector on the left and sensors on top.

Claude: [Calls place_board with constraints]
        I've optimized the placement for your sensor board. Here's the result:
        - Confidence Score: 87%
        - 42 components placed
        - 2 flags to review:
          1. Crystal Y1 is 4.2mm from MCU (recommended <3mm)
          2. Decoupling cap C3 could be closer to U1.VDD

        Would you like me to adjust the crystal placement?

User: Yes, move the crystal closer to the ESP32.

Claude: [Calls modify_placement("move Y1 closer to U1")]
        Done. Crystal Y1 is now 2.1mm from the ESP32.
        Confidence score improved to 91%.

User: Great, now route it.

Claude: [Calls route_board]
        Routing complete:
        - 98% of nets routed (147/150)
        - 3 unrouted: GND (expected), VCC (expected), USB_D+ (manual review)
        - DRC: 0 errors, 2 warnings (clearance at J1)
```

**Example 2: Iterative Refinement**
```
User: Show me the current component positions on my board.

Claude: [Reads board://path/components resource]
        Here are the components on your board:

        | Ref  | Footprint  | Position     | Rotation |
        |------|------------|--------------|----------|
        | U1   | QFN-48     | (50.0, 30.0) | 0 deg    |
        | C1   | 0402       | (48.2, 32.1) | 90 deg   |
        | ...  | ...        | ...          | ...      |

User: Rotate U1 by 45 degrees.

Claude: [Calls modify_placement("rotate U1 45 degrees")]
        U1 rotated to 45 deg. Note: This is a non-standard angle that may
        complicate routing. Would you like to use 0 or 90 instead?
```

## Implementation Plan

### Phase 2C: Basic MCP Server
1. FastMCP server skeleton with stdio transport
2. `place_board` tool with constraint support
3. `validate_placement` tool with confidence reporting
4. Basic resources (components, nets, confidence)

### Phase 3: Full MCP Integration
1. `route_board` tool with Freerouting integration
2. `modify_placement` with all modification types
3. `export_manufacturing` for fab outputs
4. Prompts for guided workflows
5. Board state management for multi-turn conversations

### Phase 4: Advanced Features
1. Streaming progress updates during long operations
2. Image generation for visual placement preview
3. Undo/redo support
4. Concurrent board sessions

## Testing Strategy

```python
# tests/test_mcp_server.py

import pytest
from atoplace.mcp.server import mcp

@pytest.fixture
def test_board(tmp_path):
    """Create a minimal test board."""
    board_path = tmp_path / "test.kicad_pcb"
    # Create minimal KiCad board file
    return board_path

async def test_place_board_tool(test_board):
    """Test place_board tool execution."""
    result = await mcp.call_tool(
        "place_board",
        {"board_path": str(test_board)}
    )
    assert result["status"] == "success"
    assert "confidence_score" in result

async def test_constraint_parsing():
    """Test constraint parsing through MCP."""
    result = await mcp.call_tool(
        "add_constraint",
        {
            "board_path": "test.kicad_pcb",
            "constraint": "keep C1 close to U1"
        }
    )
    assert result["constraint_type"] == "proximity"
```
