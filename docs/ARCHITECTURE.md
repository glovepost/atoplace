# AtoPlace System Architecture

This document visualizes the high-level architecture of the AtoPlace codebase, reflecting the current state as of January 2026.

## System Components

```mermaid
graph TD
    %% Styling
    classDef interface fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef core fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef engine fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef logic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph "User Interface"
        CLI[CLI Entry Point<br>atoplace.cli]:::interface
        NLP[NLP Engine<br>atoplace.nlp]:::interface
    end

    subgraph "Data Layer"
        Board[Board Abstraction<br>atoplace.board]:::core
        KiCadIO[KiCad Adapter<br>pcbnew / file IO]:::core
        AtoIO[Atopile Adapter<br>ato.yaml / lockfile]:::core
        Constraints[Constraint Logic<br>atoplace.placement.constraints]:::core
    end

    subgraph "Placement Engine"
        Physics[Force-Directed Physics<br>atoplace.placement.force_directed]:::engine
        StarModel[Star Model Attraction<br>(O(N) for Power/GND)]:::logic
        
        Legalizer[Manhattan Legalizer<br>atoplace.placement.legalizer]:::engine
        Quantizer[Grid Quantizer<br>(Phase 1)]:::logic
        Aligner[Row/Col Aligner<br>(Phase 2 - PCA)]:::logic
        Shove[Overlap Solver<br>(Phase 3 - Abacus/MTV)]:::logic
    end

    subgraph "Routing Engine"
        Router[A* Geometric Router<br>atoplace.routing.astar_router]:::engine
        Spatial[Spatial Hash Index<br>atoplace.routing.spatial_index]:::logic
        ObsMap[Obstacle Builder<br>atoplace.routing.obstacle_map]:::logic
        Viz[Route Visualizer<br>atoplace.routing.visualizer]:::logic
    end

    subgraph "Validation"
        Validator[DFM & Confidence<br>atoplace.validation]:::engine
    end

    %% Data Flow Connections
    CLI -->|1. Parse Input| NLP
    CLI -->|2. Load Project| AtoIO
    CLI -->|3. Load Board| KiCadIO
    
    AtoIO -->|Enrich Metadata| Board
    KiCadIO -->|Load Geometry| Board
    NLP -->|Generate| Constraints
    Constraints -->|Guide| Physics

    %% Placement Flow
    CLI -->|4. Optimize| Physics
    Physics -->|Update Positions| Board
    Physics -->|Uses| StarModel

    CLI -->|5. Legalize| Legalizer
    Legalizer -->|Refine| Board
    Legalizer -->|1. Snap| Quantizer
    Legalizer -->|2. Align| Aligner
    Legalizer -->|3. Resolve| Shove

    %% Routing Flow
    CLI -->|6. Route| Router
    Router -->|Query| Spatial
    Router -->|Build Map| ObsMap
    ObsMap -->|Scan| Board
    Router -->|Add Traces| Board
    Router -->|Debug| Viz

    %% Validation Flow
    CLI -->|7. Verify| Validator
    Validator -->|Read| Board
    Validator -->|Report| CLI
```

## Module Descriptions

### Data Layer (`atoplace.board`)
- **Board Abstraction**: The central source of truth. Decouples algorithms from specific file formats.
- **Adapters**: Handles IO for KiCad (`.kicad_pcb`) and Atopile (`ato.yaml`, `ato-lock.yaml`).

### Placement Engine (`atoplace.placement`)
- **Force-Directed Physics**: Global optimization engine. Uses a **Star Model** for high-degree nets to prevent collapse and **Spatial Hashing** for performance.
- **Manhattan Legalizer**: Post-processing pipeline.
    1.  **Quantizer**: Snaps components to user-defined grids.
    2.  **Aligner**: Uses PCA to align passive clusters into strict rows/columns.
    3.  **Shove**: Removes overlaps using priority-based displacement.

### Routing Engine (`atoplace.routing`)
- **A* Router**: Deterministic geometric planner using a **Greedy Multiplier** for speed.
- **Spatial Hash**: O(~1) collision detection system replacing QuadTrees.
- **Obstacle Map**: Converts board geometry (pads, keepouts) into routing obstacles.

### Validation (`atoplace.validation`)
- **DFM Checker**: Verifies clearance, width, and ring rules.
- **Confidence Scorer**: Heuristic analysis of board quality/manufacturability.
