# 13 Things I Would Have Told Myself Before Building an Autorouter

**Source**: @seveibar on Twitter/X, March 28, 2025
**Blog**: https://blog.autorouting.com/p/13-things-i-would-have-told-myself
**Project**: [tscircuit](https://tscircuit.com) - open-source electronics CAD kernel in TypeScript

---

## Summary

Lessons learned from about a year of developing an autorouter. These insights are valuable for AtoPlace's Phase 3 (Routing Integration).

---

## The 13 Lessons

### 1. Know A* Like the Back of Your Hand, Use It Everywhere

> If I was king for a day, I would rename A* to "Fundamental Algorithm". It is truly one of the most adaptable and important algorithms for _any kind_ of search. It is simply the best foundation for any kind of informed search (not just for 2d grids!)

**Applicability to AtoPlace**: A* should be the foundation of the routing engine. Can also be applied to:
- Path finding between pads
- Via placement optimization
- Escape routing from BGAs/QFNs

---

### 2. Implementation Language Doesn't Matter

> The difference between a smart algorithm and the dumb algorithm is 1000x, whatever gets you to the smartest, most cacheable algorithm fastest is the best language.

**Applicability to AtoPlace**: Python is fine. Focus on algorithm quality over micro-optimizations.

---

### 3. Spatial Hash Indexing > Tree Data Structures

> You can't walk 5 feet into multi-dimensional space optimization without someone mentioning a QuadTree. Any time you're using a tree you're ignoring an O(~1) hash algorithm for a more complicated O(log(N)) algorithm.

**Applicability to AtoPlace**: Already implemented in `legalizer.py` with `_build_spatial_index()`. Use this pattern for routing collision detection instead of R-Trees or QuadTrees.

**Implementation**:
```python
def _build_spatial_index(obstacles, cell_size):
    """Grid-based spatial hash for O(~1) collision lookup."""
    index = {}
    for obs in obstacles:
        cells = get_cells_for_obstacle(obs, cell_size)
        for cell in cells:
            if cell not in index:
                index[cell] = []
            index[cell].append(obs)
    return index
```

---

### 4. Effective Spatial Partitioning + Caching is 1000x More Important Than Algorithm Performance

> Game developers "bake" navigation meshes into many gigabytes for their games. LLMs compress the entire internet into weights. The next generation of autorouters will use massive caches.

**Applicability to AtoPlace**: Consider pre-computing:
- Obstacle maps per layer
- Clearance grids (distance to nearest obstacle)
- Via-drop feasibility maps
- Escape paths from dense component areas

**Implementation Ideas**:
- Cache board state after placement for routing
- Pre-compute "routing channels" between component groups
- Store successful routing patterns for reuse

---

### 5. If You Do Not Have a Visualization for a Problem, You Will Never Solve It

> If I could have one thing printed on a poster, it would be VISUALIZE THE PROBLEM. You can't debug problems by staring at numbers.

**Applicability to AtoPlace**: Critical gap! Currently only text logging exists. Need:
- Real-time visualization of routing progress
- Overlay of obstacles, clearances, and paths
- Step-by-step algorithm visualization

**Implementation Plan**:
1. SVG export of routing state at each iteration
2. Optional matplotlib/plotly live visualization
3. HTML report with interactive layer views

---

### 6. Javascript Profiling Is Amazing - Use It!

> Javascript profiling tools are incredible, you can easily see the exact total time in ms spent on each line of code.

**Applicability to AtoPlace**: Use Python equivalents:
- `cProfile` for function-level profiling
- `line_profiler` for line-by-line timing
- `py-spy` for sampling profiler
- `snakeviz` for visualization

---

### 7. Never Use Recursive Functions

> Recursive functions are bad for multiple reasons:
> - They are almost always synchronous (can't be animated)
> - They are inherently a Depth-First Search (hard to use A*)
> - Mutability is unnatural in recursive functions but critical to performance.

**Applicability to AtoPlace**: Use iterative implementations with explicit stacks/queues:

```python
# BAD - recursive
def route_recursive(start, end, visited):
    if start == end:
        return [end]
    for neighbor in get_neighbors(start):
        if neighbor not in visited:
            path = route_recursive(neighbor, end, visited | {neighbor})
            if path:
                return [start] + path
    return None

# GOOD - iterative with priority queue (A*)
def route_astar(start, end):
    open_set = [(0, start, [start])]
    closed = set()
    while open_set:
        _, current, path = heapq.heappop(open_set)
        if current == end:
            return path
        if current in closed:
            continue
        closed.add(current)
        for neighbor in get_neighbors(current):
            if neighbor not in closed:
                g = len(path)
                h = heuristic(neighbor, end)
                heapq.heappush(open_set, (g + h, neighbor, path + [neighbor]))
    return None
```

---

### 8. Monte Carlo Algorithms Are Hacks. AVOID

> Monte Carlo algorithms use randomness to iterate towards a solution. They are bad because:
> - They lead to non-deterministic, hard-to-debug algorithms
> - They are basically never optimal relative to a heuristic.

**Applicability to AtoPlace**: Keep routing deterministic. Don't use:
- Random restarts
- Simulated annealing for routing
- Genetic algorithms for path optimization

Instead use:
- Deterministic tie-breaking (e.g., prefer left, then up)
- Consistent heuristics
- Reproducible ordering

---

### 9. Keep Intermediate Algorithms Grounded

> Being able to overlay different inputs/output visualizations of each stage of the algorithm helps you understand the context surrounding the problem you're solving.

**Applicability to AtoPlace**: For routing pipeline:
1. Show input: pads to connect, obstacles
2. Show net ordering decision
3. Show each net's routing attempt
4. Show via placement decisions
5. Show final result

---

### 10. Animate Your Iterations to Catch Stupid Behavior

> Animating the iterations of your algorithm will show you how "dumb" it's being by giving you an intuition for how many iterations are wasted exploring paths that don't matter.

**Applicability to AtoPlace**: Add optional animation mode:
```python
class AnimatedRouter:
    def __init__(self, board, animate=False, delay_ms=50):
        self.animate = animate
        self.delay_ms = delay_ms

    def route_net(self, net):
        for iteration in self._astar_iterations(net):
            if self.animate:
                self._render_frame(iteration)
                time.sleep(self.delay_ms / 1000)
```

---

### 11. Intersection Math Is Fast, Do You Really Need a Grid?

> Everyone defaults to using grid collision checking. Use fast vector math!! Checking a SINGLE grid square (memory access!) can literally be slower than doing a dot product to determine if two segments intersect!

**Applicability to AtoPlace**: Consider gridless routing for critical nets:

```python
def segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects p3-p4 using cross products."""
    d1 = cross_product(p4 - p3, p1 - p3)
    d2 = cross_product(p4 - p3, p2 - p3)
    d3 = cross_product(p2 - p1, p3 - p1)
    d4 = cross_product(p2 - p1, p4 - p1)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False
```

Hybrid approach:
- Use spatial hash for coarse filtering
- Use vector math for precise intersection tests

---

### 12. Measure Spatial Probability of Failure

Track where routing failures occur spatially. Build a "heat map" of failure probability:
- Dense areas with many failed routes
- Bottleneck regions
- Via-starved zones

Use this to:
- Prioritize difficult nets first
- Adjust placement before routing
- Identify design rule violations early

---

### 13. The "Greedy Multiplier" - Secret Hack to 100x A* Performance

Standard A* uses: `f(n) = g(n) + h(n)` where g=cost so far, h=heuristic to goal.

**Greedy Multiplier**: `f(n) = g(n) + w * h(n)` where w > 1

- w = 1: Standard A* (optimal but slow)
- w = 2-5: Much faster, slightly suboptimal
- w = 10+: Very fast, may miss good paths

For PCB routing, w = 2-3 is often ideal:
- Most paths are simple
- Suboptimality is acceptable
- Speed is critical for many nets

```python
def astar_greedy(start, end, w=2.0):
    """A* with greedy multiplier for faster routing."""
    open_set = [(0, start, [start])]
    g_scores = {start: 0}

    while open_set:
        _, current, path = heapq.heappop(open_set)
        if current == end:
            return path

        for neighbor in get_neighbors(current):
            tentative_g = g_scores[current] + cost(current, neighbor)
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f = tentative_g + w * heuristic(neighbor, end)  # Greedy multiplier!
                heapq.heappush(open_set, (f, neighbor, path + [neighbor]))

    return None
```

---

## Key Takeaways for AtoPlace

1. **A* is fundamental** - Use it as the core routing algorithm
2. **Spatial hashing over trees** - Already doing this, keep it
3. **Cache aggressively** - Pre-compute obstacle maps, clearance grids
4. **Visualize everything** - Critical gap to address
5. **Stay deterministic** - No randomness in routing
6. **Iterative over recursive** - Use explicit queues
7. **Greedy multiplier** - 2-3x heuristic weight for speed

---

## References

- Blog post: https://blog.autorouting.com/p/13-things-i-would-have-told-myself
- tscircuit: https://github.com/tscircuit/tscircuit
- tscircuit autorouter: https://github.com/tscircuit/autorouting
