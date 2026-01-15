# Unified Visualization Design

This document outlines the design for consolidating atoplace's fragmented visualization system into a single, unified HTML-based viewer capable of loading atopile projects and initiating placement.

## Baseline: SVG Delta Viewer

The **SVG delta viewer** (`svg_delta_viewer.py`) is chosen as the baseline for the unified approach because it provides:

1. **Vector Quality** - Crisp rendering at any zoom level (SVG vs Canvas pixels)
2. **Delta Compression** - Same 90% file size reduction as Canvas approach
3. **Efficient Updates** - DOM attribute manipulation (30-40 FPS vs 6 FPS for full SVG replacement)
4. **Semantic Elements** - Components are real DOM elements with `data-ref` attributes
5. **Rich Features** - Already has pan/zoom, layers, trails, ratsnest, forces, energy graph

The Canvas approach was designed for 1000+ components, but SVG delta is sufficient for typical boards (100-500 components) and provides better quality.

## Current State Analysis

### File Inventory

| File | Lines | Status | Keep/Migrate |
|------|-------|--------|--------------|
| `placement/visualizer.py` | ~1000 | Active | Keep (frame capture only) |
| `placement/canvas_renderer.py` | ~650 | Active | Migrate to external JS |
| `placement/delta_compression.py` | ~300 | Active | Keep (Python) |
| `placement/viewer_template.py` | ~700 | Active | Replace with minimal template |
| `placement/viewer_javascript.py` | ~1200 | Active | Migrate to external JS |
| `placement/streaming_visualizer.py` | ~350 | Active | Keep (with refactor) |
| `placement/stream_server.py` | ~400 | Active | Keep |
| `placement/stream_viewer.py` | ~200 | Active | Merge into unified template |
| `placement/svg_delta_viewer.py` | ~100 | Legacy? | Deprecate |
| `routing/visualizer.py` | ~500 | Active | Keep (frame capture only) |
| `visualization_color_manager.py` | ~340 | Active | Keep |
| `visualization_colors.yaml` | ~60 | Config | Keep |

### Pain Points

1. **JavaScript as Python Strings**
   - No syntax highlighting in IDE
   - No linting or type checking
   - No minification possible
   - Difficult to debug (line numbers don't match)

2. **Multiple Entry Points**
   - `PlacementVisualizer.export_canvas_html()`
   - `PlacementVisualizer.export_html_report()` (SVG)
   - `StreamingVisualizer.start_streaming()`
   - `RouteVisualizer.export_html_report()`
   - `generate_stream_viewer_html()`

3. **No Unified Data Model**
   - Placement uses `PlacementFrame`
   - Routing uses `VisualizationFrame`
   - Streaming uses raw dicts
   - No shared schema for interoperability

## Design Goals

1. **Single HTML Entry Point:** One viewer for placement, routing, and streaming
2. **External Assets:** JS and CSS as real files, not Python strings
3. **Project Loading:** Load atopile projects directly in browser
4. **Placement Control:** Trigger and monitor placement from the UI
5. **Backward Compatible:** Existing APIs continue to work

## Architecture

### Design Principle: SVG Delta as Single Source of Truth

The SVG delta viewer JavaScript (`svg_delta_viewer.py:generate_svg_delta_viewer_js()`) contains the authoritative implementation for:
- Coordinate transforms (`tx()`, `ty()`, `ts()`)
- Delta frame playback (`reconstructState()`, `applyDelta()`)
- Layer rendering (pads, ratsnest, forces, trails, module groups)
- UI interactions (pan, zoom, selection, playback)

This ~1500 lines of JavaScript will be:
1. **Extracted** to a real `.js` file for IDE support
2. **Extended** with project loading capabilities
3. **Used by** both static export and streaming modes

### Directory Structure

```
atoplace/
├── visualization/                    # New consolidated module
│   ├── __init__.py                   # Public API
│   │
│   ├── assets/                       # Static web assets (REAL files, not Python strings!)
│   │   ├── svg-delta-viewer.js       # Extracted from svg_delta_viewer.py
│   │   ├── stream-client.js          # WebSocket client for streaming
│   │   ├── project-loader.js         # Atopile project browser
│   │   └── styles.css                # Unified CSS (from viewer_template.py)
│   │
│   ├── capture/                      # Frame capture (Python)
│   │   ├── __init__.py
│   │   ├── placement.py              # PlacementFrameCapture
│   │   └── routing.py                # RoutingFrameCapture
│   │
│   ├── compression/                  # Data compression
│   │   ├── __init__.py
│   │   └── delta.py                  # DeltaCompressor (moved)
│   │
│   ├── colors/                       # Color management
│   │   ├── __init__.py
│   │   ├── manager.py                # ColorManager (moved)
│   │   └── colors.yaml               # Color config (moved)
│   │
│   ├── export/                       # HTML generation
│   │   ├── __init__.py
│   │   ├── template.py               # Minimal HTML shell
│   │   └── bundler.py                # Asset bundling
│   │
│   └── streaming/                    # WebSocket streaming
│       ├── __init__.py
│       └── server.py                 # StreamServer (moved)
│
├── placement/
│   └── visualizer.py                 # Simplified, delegates to visualization/
│
└── routing/
    └── visualizer.py                 # Simplified, delegates to visualization/
```

### Unified Data Model

```python
# visualization/models.py

@dataclass
class ViewerFrame:
    """Unified frame format for all visualization types."""

    # Identification
    index: int
    label: str
    frame_type: Literal["placement", "routing", "combined"]

    # Timing
    timestamp: float
    iteration: int
    phase: str

    # Component state (for placement)
    components: Dict[str, ComponentState] = field(default_factory=dict)

    # Routing state (for routing)
    traces: List[TraceSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)

    # Visual annotations
    forces: Dict[str, List[ForceVector]] = field(default_factory=dict)
    overlaps: List[Tuple[str, str]] = field(default_factory=list)
    modules: Dict[str, str] = field(default_factory=dict)

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComponentState:
    """Position and rotation of a component."""
    x: float
    y: float
    rotation: float
    layer: str = "F.Cu"


@dataclass
class TraceSegment:
    """A single trace segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    layer: str
    net: str
```

### JavaScript Module Structure

The SVG delta viewer already contains all core rendering logic. We extend it with ES6 modules:

```javascript
// assets/svg-delta-viewer.js - Extracted from svg_delta_viewer.py
// This is the existing ~1500 line implementation, now as a real .js file

// Key functions (already implemented):
// - tx(x), ty(y), ts(size) - Coordinate transforms
// - reconstructState(toFrame) - Delta decompression
// - applyDelta(delta) - Apply frame changes
// - updateSVGElements() - DOM manipulation
// - showFrame(frameIndex) - Main frame display
// - drawTrails(), updateRatsnest(), updateForces() - Layer rendering
// - pan/zoom handlers, playback controls, energy graph
```

```javascript
// assets/stream-client.js - NEW: WebSocket client for streaming mode

class StreamClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.onFrameCallback = null;
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            this.ws.onopen = () => resolve();
            this.ws.onerror = (e) => reject(e);
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'frame' && this.onFrameCallback) {
                    this.onFrameCallback(data.frame);
                }
            };
        });
    }

    onFrame(callback) {
        this.onFrameCallback = callback;
    }

    disconnect() {
        if (this.ws) this.ws.close();
    }
}

export { StreamClient };
```

```javascript
// assets/project-loader.js - NEW: Atopile project browser

class ProjectLoader {
    constructor(apiBaseUrl = '/api') {
        this.apiBase = apiBaseUrl;
        this.currentProject = null;
    }

    async loadProject(path) {
        const response = await fetch(`${this.apiBase}/project?path=${encodeURIComponent(path)}`);
        if (!response.ok) throw new Error(await response.text());
        this.currentProject = await response.json();
        return this.currentProject;
    }

    async loadBoard(buildName = 'default') {
        if (!this.currentProject) throw new Error('No project loaded');
        const response = await fetch(
            `${this.apiBase}/project/board?path=${encodeURIComponent(this.currentProject.root)}&build=${buildName}`
        );
        return response.json();
    }

    async startPlacement(options = {}) {
        const response = await fetch(`${this.apiBase}/place`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project: this.currentProject?.root,
                ...options
            })
        });
        return response.json();
    }
}

export { ProjectLoader };
```

### HTML Template

```html
<!-- assets/template.html -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{title}} - atoplace Viewer</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="app">
        <!-- Header with controls -->
        <header id="toolbar">
            <div class="playback-controls">
                <button id="btn-play" title="Play (Space)">▶</button>
                <button id="btn-reset" title="Reset">⏮</button>
                <input type="range" id="frame-slider" min="0" max="100" value="0">
                <span id="frame-counter">0 / 0</span>
            </div>

            <div class="layer-controls">
                <label><input type="checkbox" id="show-grid" checked> Grid</label>
                <label><input type="checkbox" id="show-pads" checked> Pads</label>
                <label><input type="checkbox" id="show-forces"> Forces</label>
                <label><input type="checkbox" id="show-ratsnest" checked> Ratsnest</label>
            </div>

            <div class="status">
                <span id="status-indicator" class="status-dot"></span>
                <span id="status-text">Ready</span>
            </div>
        </header>

        <!-- Main canvas area -->
        <main id="canvas-container">
            <canvas id="static-canvas"></canvas>
            <canvas id="dynamic-canvas"></canvas>
        </main>

        <!-- Sidebar with project/info -->
        <aside id="sidebar">
            <!-- Project browser (if enabled) -->
            <section id="project-panel" class="panel">
                <h3>Project</h3>
                <div id="project-tree"></div>
            </section>

            <!-- Module list -->
            <section id="modules-panel" class="panel">
                <h3>Modules</h3>
                <div id="module-list"></div>
            </section>

            <!-- Metrics -->
            <section id="metrics-panel" class="panel">
                <h3>Metrics</h3>
                <div id="metrics-display"></div>
            </section>
        </aside>
    </div>

    <!-- Data injection point -->
    <script id="viewer-data" type="application/json">
    {{data}}
    </script>

    <!-- Load modules -->
    <script type="module">
        import { UnifiedViewer } from './viewer.js';

        const dataElement = document.getElementById('viewer-data');
        const config = JSON.parse(dataElement.textContent);

        const viewer = new UnifiedViewer(
            document.getElementById('canvas-container'),
            {
                mode: config.mode || 'static',
                websocketUrl: config.websocketUrl,
                enableProjectLoader: config.enableProjectLoader
            }
        );

        if (config.staticProps && config.deltaFrames) {
            viewer.loadFrames(config.staticProps, config.deltaFrames);
        }

        if (config.mode === 'streaming' && config.websocketUrl) {
            viewer.connectStreaming(config.websocketUrl);
        }

        window.viewer = viewer;
    </script>
</body>
</html>
```

### Python Export API

```python
# visualization/export/__init__.py

from pathlib import Path
from typing import Optional, List
from ..models import ViewerFrame
from ..compression.delta import DeltaCompressor
from .template import render_template
from .bundler import bundle_assets


class ViewerExporter:
    """Export visualization data to HTML."""

    def __init__(
        self,
        mode: str = "static",
        enable_project_loader: bool = False,
        websocket_url: Optional[str] = None,
    ):
        self.mode = mode
        self.enable_project_loader = enable_project_loader
        self.websocket_url = websocket_url
        self.compressor = DeltaCompressor()
        self.frames: List[ViewerFrame] = []
        self.static_props: dict = {}

    def add_frame(self, frame: ViewerFrame):
        """Add a frame to the visualization."""
        self.frames.append(frame)

    def set_static_props(self, props: dict):
        """Set static component properties (dimensions, pads)."""
        self.static_props = props

    def export(
        self,
        output_path: Path,
        title: str = "atoplace Visualization",
        inline: bool = True,
    ) -> Path:
        """
        Export to HTML file.

        Args:
            output_path: Destination path
            title: Page title
            inline: If True, bundle all assets into single HTML file
                   If False, generate HTML + separate JS/CSS files

        Returns:
            Path to generated HTML file
        """
        # Compress frames
        delta_frames = self.compressor.compress_all(self.frames)

        # Build config object
        config = {
            "mode": self.mode,
            "enableProjectLoader": self.enable_project_loader,
            "websocketUrl": self.websocket_url,
            "staticProps": self.static_props,
            "deltaFrames": [f.to_dict() for f in delta_frames],
        }

        if inline:
            # Bundle everything into single HTML
            html = render_template(
                title=title,
                data=config,
                inline_js=bundle_assets("js"),
                inline_css=bundle_assets("css"),
            )
            output_path.write_text(html)
        else:
            # Generate HTML + asset directory
            assets_dir = output_path.parent / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Copy asset files
            bundle_assets("js", output=assets_dir / "viewer.bundle.js")
            bundle_assets("css", output=assets_dir / "styles.css")

            # Generate HTML with asset references
            html = render_template(
                title=title,
                data=config,
                js_path="assets/viewer.bundle.js",
                css_path="assets/styles.css",
            )
            output_path.write_text(html)

        return output_path


# Convenience function for common use case
def export_placement_html(
    frames: List[ViewerFrame],
    static_props: dict,
    output_path: Path,
    title: str = "Placement Visualization",
) -> Path:
    """Export placement frames to HTML."""
    exporter = ViewerExporter(mode="static")
    exporter.set_static_props(static_props)
    for frame in frames:
        exporter.add_frame(frame)
    return exporter.export(output_path, title=title)
```

## Project Loading Feature

### Backend API

For project loading to work from the browser, we need a lightweight HTTP API:

```python
# visualization/api.py

from fastapi import FastAPI, HTTPException
from pathlib import Path
from ..board.atopile_adapter import AtopileProjectLoader, detect_board_source

app = FastAPI()


@app.get("/api/project")
async def get_project_info(path: str):
    """Get atopile project information."""
    try:
        loader = AtopileProjectLoader(Path(path))
        return {
            "root": str(loader.root),
            "builds": loader.get_build_names(),
            "default_build": "default" if "default" in loader.get_build_names() else None,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/project/board")
async def get_board_info(path: str, build: str = "default"):
    """Get board information for a build."""
    try:
        loader = AtopileProjectLoader(Path(path))
        board = loader.load_board(build)

        return {
            "path": str(loader.get_board_path(build)),
            "components": len(board.components),
            "nets": len(board.nets),
            "bounds": board.get_bounds(),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/place")
async def start_placement(request: PlacementRequest):
    """Start placement optimization (async)."""
    # Returns task ID, placement runs in background
    # WebSocket streams progress
    pass
```

### JavaScript Project Loader

```javascript
// assets/project-loader.js

class ProjectLoader {
    constructor(viewer) {
        this.viewer = viewer;
        this.currentProject = null;
    }

    async load(path) {
        // Fetch project info
        const response = await fetch(`/api/project?path=${encodeURIComponent(path)}`);
        if (!response.ok) {
            throw new Error(await response.text());
        }

        this.currentProject = await response.json();
        this.renderProjectTree();
        return this.currentProject;
    }

    renderProjectTree() {
        const container = document.getElementById('project-tree');
        container.innerHTML = '';

        // Render builds
        const buildList = document.createElement('ul');
        for (const build of this.currentProject.builds) {
            const li = document.createElement('li');
            li.textContent = build;
            li.onclick = () => this.selectBuild(build);
            if (build === this.currentProject.default_build) {
                li.classList.add('default');
            }
            buildList.appendChild(li);
        }
        container.appendChild(buildList);
    }

    async selectBuild(buildName) {
        const boardInfo = await fetch(
            `/api/project/board?path=${encodeURIComponent(this.currentProject.root)}&build=${buildName}`
        ).then(r => r.json());

        // Update viewer with board info
        this.viewer.loadBoard(boardInfo);
    }

    async startPlacement(options = {}) {
        if (!this.currentProject) {
            throw new Error('No project loaded');
        }

        const response = await fetch('/api/place', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project: this.currentProject.root,
                build: options.build || this.currentProject.default_build,
                constraints: options.constraints || [],
            }),
        });

        return response.json();
    }
}

export { ProjectLoader };
```

## Migration Plan

### Phase 1: Extract SVG Delta JavaScript (1 day)

The SVG delta viewer is already a complete implementation. This phase extracts it to a real `.js` file.

1. Create `visualization/assets/` directory
2. Extract JavaScript from `svg_delta_viewer.py` → `svg-delta-viewer.js`
3. Extract CSS from `viewer_template.py` → `styles.css`
4. Update HTML template to load external files
5. Test with existing SVG delta export path

**Files created:**
- `visualization/assets/svg-delta-viewer.js` (~1500 lines, extracted)
- `visualization/assets/styles.css` (~300 lines, extracted)

**Files modified:**
- `placement/visualizer.py` - Update `export_svg_delta_html()` to load external JS

### Phase 2: Deprecate Canvas/Legacy Paths (0.5 day)

The SVG delta approach is superior. Deprecate alternatives:

1. Mark `export_canvas_html()` as deprecated (use `export_svg_delta_html()`)
2. Mark `export_html_report()` (full SVG) as deprecated
3. Add deprecation warnings pointing to SVG delta
4. Update CLI default to SVG delta

**Files modified:**
- `placement/visualizer.py` - Add deprecation warnings
- `cli.py` - Default to SVG delta export

### Phase 3: Add Routing Support to SVG Delta (1 day)

Extend SVG delta viewer to handle routing visualization:

1. Add trace/via rendering to `svg-delta-viewer.js`
2. Add layer toggles for routing layers (F.Cu, B.Cu, etc.)
3. Create adapter for routing frames → delta format
4. Update `routing/visualizer.py` to use SVG delta

**Files modified:**
- `visualization/assets/svg-delta-viewer.js` - Add routing layers
- `routing/visualizer.py` - Use SVG delta export

### Phase 4: Add Project Loading (2-3 days)

Enable loading atopile projects directly in the viewer:

1. Create `visualization/api.py` with FastAPI routes
2. Create `visualization/assets/project-loader.js`
3. Add project browser panel to HTML template
4. Integrate with streaming for live placement

**Files created:**
- `visualization/api.py` - REST API for project operations
- `visualization/assets/project-loader.js` - Browser-side project loading
- `visualization/assets/stream-client.js` - WebSocket client

**Files modified:**
- `visualization/export/template.py` - Add project panel to HTML

### Phase 5: Cleanup (1 day)

Remove deprecated/redundant files:

**Files to remove:**
- `placement/canvas_renderer.py` - Canvas approach deprecated
- `placement/viewer_javascript.py` - Replaced by extracted JS
- `placement/stream_viewer.py` - Merged into unified template

**Files to consolidate:**
- `placement/svg_delta_viewer.py` → `visualization/assets/svg-delta-viewer.js`
- `placement/viewer_template.py` → `visualization/export/template.py`
- `placement/delta_compression.py` → `visualization/compression/delta.py`

## Testing Strategy

### Unit Tests

```python
# tests/test_visualization_export.py

def test_viewer_exporter_static():
    """Test static HTML export."""
    exporter = ViewerExporter(mode="static")
    # Add test frames
    exporter.export(Path("test_output.html"))
    # Verify HTML structure

def test_viewer_exporter_streaming():
    """Test streaming mode configuration."""
    exporter = ViewerExporter(
        mode="streaming",
        websocket_url="ws://localhost:8765"
    )
    # Verify config includes WebSocket URL

def test_delta_compression():
    """Test frame compression."""
    compressor = DeltaCompressor()
    # Test compression ratio
```

### Integration Tests

```python
# tests/test_visualization_integration.py

def test_full_placement_visualization():
    """Test complete placement → visualization pipeline."""
    board = Board.from_kicad("test.kicad_pcb")
    viz = PlacementVisualizer(board)
    # Run placement
    viz.export_html("output.html")
    # Verify file exists and is valid HTML

def test_streaming_visualization():
    """Test WebSocket streaming."""
    # Start server
    # Connect client
    # Send frames
    # Verify receipt
```

### Browser Tests (E2E)

```javascript
// tests/e2e/viewer.spec.js (Playwright)

test('viewer loads frames', async ({ page }) => {
    await page.goto('file://output.html');
    await expect(page.locator('#frame-counter')).toContainText('0 / 100');
});

test('playback works', async ({ page }) => {
    await page.goto('file://output.html');
    await page.click('#btn-play');
    await page.waitForTimeout(1000);
    // Verify frame counter increased
});
```

## SVG Delta Viewer Feature Inventory

The existing implementation (`svg_delta_viewer.py`) provides these features that will be preserved:

### Frame Playback
- `showFrame(frameIndex)` - Display specific frame
- `reconstructState(toFrame)` - Delta decompression with sequential optimization
- `applyDelta(delta)` - Apply frame changes to current state
- Playback controls: play/pause, speed, first/prev/next/last

### Coordinate System
- `tx(x)`, `ty(y)`, `ts(size)` - Board-to-SVG coordinate transforms
- `boardBounds` - Padding, scale, min/max values
- Proper rotation handling with `transform` attribute

### Layer Visibility
- Top/Bottom copper layers
- Edge cuts (board outline)
- Grid
- Ratsnest (MST-based)
- Force vectors
- Pads
- Reference labels
- Module groups

### Module Support
- `moduleColors` - Per-module coloring
- `moduleVisibility` - Per-module toggle
- `updateModuleGroups()` - Dynamic bounding boxes
- Module class sanitization for CSS

### Interactions
- Pan: Drag to move view
- Zoom: Mouse wheel with zoom-toward-cursor
- Selection: Click component to highlight
- Seeking: Click energy graph to jump to frame

### Visualization Layers
- `updateRatsnest()` - MST edges between net pads
- `updateForces()` - Force vector arrows
- `drawTrails()` - Component movement history
- `updateOverlapHighlighting()` - Overlap detection

### Metrics Display
- Energy graph with click-to-seek
- Wire length overlay
- Frame counter
- Phase/iteration display
- Overlap count

## Success Metrics

1. **Code Reduction:** Target 30% fewer lines (consolidate 12 files → 5-6 files)
2. **Render Quality:** Vector-crisp at any zoom level (SVG advantage)
3. **FPS:** 30-40 FPS playback (SVG delta maintains this)
4. **File Size:** <5MB for typical 500-frame placement runs (delta compression)
5. **Developer Experience:** Real `.js` files with IDE support, linting, debugging

## Open Questions

1. **Asset Bundling:** Use esbuild for minification, or keep simple for debugging?
2. **Routing Layers:** How many layers to support? Dynamic palette vs fixed colors?
3. **API Backend:** FastAPI standalone server or integrate with existing MCP server?
4. **Streaming Protocol:** Extend existing `stream_server.py` or new implementation?

## Appendix: SVG Delta vs Canvas Comparison

| Feature | SVG Delta | Canvas |
|---------|-----------|--------|
| Render quality | Vector (infinite zoom) | Pixel (fixed resolution) |
| Performance | 30-40 FPS (DOM updates) | 60+ FPS (direct draw) |
| Component limit | ~500 components | 1000+ components |
| Selection/hover | Native DOM events | Manual hit testing |
| Accessibility | Semantic elements | Opaque bitmap |
| File export | SVG/PDF vector | PNG raster |
| Implementation | Complete (~1500 lines) | Partial (~650 lines) |

**Recommendation:** SVG delta for all current use cases. Canvas only needed for very large boards (1000+ components), which are rare in atopile projects.
