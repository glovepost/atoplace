# Manufacturing Output Generation

**Status:** 🚧 Planned for future release

This module is reserved for manufacturing output generation features that are planned but not yet implemented.

## Planned Features

### 1. GerberGenerator
Generate Gerber files (RS-274X format) for PCB manufacturing from the board abstraction.

**Capabilities:**
- Export copper layers (front, back, inner layers)
- Generate solder mask layers
- Generate silkscreen layers
- Generate drill files (Excellon format)
- Support for modern Gerber X2 format with embedded attributes

**Use Cases:**
- Send designs to generic PCB manufacturers
- Archive manufacturing data
- Preview manufacturing output

### 2. JLCPCBExporter
Export board designs in JLCPCB's required format including BOM and CPL (component placement list).

**Capabilities:**
- Generate JLCPCB-compatible Gerber files
- Export BOM (Bill of Materials) in JLCPCB format
- Export CPL (Component Placement List) with rotation corrections
- Validate design against JLCPCB capabilities
- Auto-match components to JLCPCB part numbers

**Use Cases:**
- One-click export for JLCPCB assembly service
- Validate design before ordering
- Generate assembly files

### 3. ManufacturingOutputGenerator
Unified interface for generating manufacturing outputs for various fab houses.

**Capabilities:**
- Detect target fabricator from DFM profile
- Generate appropriate output format (Gerber, ODB++, etc.)
- Include design notes and special instructions
- Generate assembly drawings
- Create manufacturing package (ZIP with all required files)

**Supported Fabricators:**
- JLCPCB
- PCBWay
- OSH Park
- Generic Gerber RS-274X

## Current Workaround

Until these features are implemented, users can:

1. **Use KiCad's native export:**
   - Load the board in KiCad after running AtoPlace
   - Use KiCad's Plot dialog for Gerber export
   - Use KiCad's BOM and position file generators

2. **Use the KiCad adapter directly:**
   ```python
   from atoplace.board.abstraction import Board

   # After placement/routing
   board = Board.from_kicad("input.kicad_pcb")
   # ... perform placement operations ...
   board.to_kicad("output.kicad_pcb")

   # Then use KiCad GUI or CLI tools for manufacturing output
   ```

## Implementation Status

| Feature | Status | Priority | Estimated Effort |
|---------|--------|----------|------------------|
| Gerber Export | Planned | Medium | 2-3 weeks |
| JLCPCB BOM/CPL | Planned | High | 1 week |
| Assembly Drawings | Planned | Low | 1 week |
| ODB++ Export | Planned | Low | 2 weeks |

## Contributing

Contributions to implement these features are welcome! Key considerations:

- **Gerber generation:** Use pygerber or implement RS-274X writer
- **JLCPCB export:** Follow JLCPCB's documented format requirements
- **Testing:** Verify output against manufacturer specifications
- **Validation:** Check generated files with CAM viewers (gerbv, KiCad Gerber viewer)

## Why Not Implemented Yet?

The AtoPlace project has prioritized:
1. **Core placement algorithm** - force-directed refinement
2. **Routing engine** - A* pathfinding with diff pairs
3. **LLM integration** - MCP server for agent control

Manufacturing output generation can leverage existing tools (KiCad's plotters) until native support is added. The board abstraction (`Board` class) contains all necessary data for output generation, so adding these features is straightforward once the core algorithms are stable.

## References

- [Gerber Format Specification (RS-274X)](https://www.ucamco.com/en/gerber)
- [JLCPCB Assembly File Requirements](https://jlcpcb.com/help/article/How-to-generate-the-BOM-and-Centroid-file-from-KiCAD)
- [Excellon Drill Format](https://www.artwork.com/gerber/drill/)
- [ODB++ Specification](https://www.odb-sa.com/)
