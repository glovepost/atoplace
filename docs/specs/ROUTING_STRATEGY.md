# Routing Strategy

## 1. Core Algorithm
A* with Greedy Multiplier ($w=2.0-3.0$) and Spatial Hash Indexing.

## 2. Implementation
See `atoplace/routing/astar_router.py`.

## 3. Fallback
Use Freerouting JAR for nets that fail the internal planner.