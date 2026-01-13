# MCP Server Specification

## 1. Goal
Expose AtoPlace capabilities to LLM agents via the Model Context Protocol.

## 2. Tools
*   `load_board(path)`
*   `place_next_to(ref, target, side, clearance)`
*   `inspect_region(refs)`
*   `save_board(path)`

## 3. Context
*   **Macro:** Board stats, module tree.
*   **Micro:** JSON geometry for selected components.