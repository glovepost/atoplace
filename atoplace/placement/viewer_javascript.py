"""JavaScript code generation for placement visualizers.

Provides all the interactive features: pan/zoom, layer visibility,
energy graph, playback controls, keyboard shortcuts, etc.
"""


def generate_viewer_javascript(
    has_energy_graph: bool = True,
    has_layer_controls: bool = True,
    is_streaming: bool = False
) -> str:
    """Generate complete JavaScript for viewer interactivity.

    Args:
        has_energy_graph: Include energy graph functionality
        has_layer_controls: Include layer visibility controls
        is_streaming: True for streaming viewer (disables some controls)

    Returns:
        Complete JavaScript code string
    """

    js_code = '''
// ============================================================================
// Global State
// ============================================================================
let currentFrame = 0;
let isPlaying = false;
let playbackInterval = null;
let playbackSpeed = 200; // ms per frame

// Pan/Zoom state
let viewTransform = { x: 0, y: 0, scale: 1.0 };
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let dragOffset = { x: 0, y: 0 };

// Layer visibility state
let layerVisibility = {
    top: true,
    bottom: true,
    edge: true,
    grid: false,
    ratsnest: true,
    trails: true,
    forces: false,
    pads: true,
    labels: true,
    groups: true
};

// Module visibility state
let moduleVisibility = {};

// Component selection state
let selectedComponent = null;

// Energy graph seeking
let isSeeking = false;

// ============================================================================
// Canvas/Container Setup
// ============================================================================
const frameContainer = document.getElementById('frame-container');
const staticCanvas = document.getElementById('staticCanvas');
const dynamicCanvas = document.getElementById('dynamicCanvas');

if (staticCanvas && dynamicCanvas) {
    const staticCtx = staticCanvas.getContext('2d');
    const dynamicCtx = dynamicCanvas.getContext('2d');
}

// ============================================================================
// Playback Controls
// ============================================================================
function firstFrame() {
    currentFrame = 0;
    showFrame(currentFrame);
}

function prevFrame() {
    if (currentFrame > 0) {
        currentFrame--;
        showFrame(currentFrame);
    }
}

function nextFrame() {
    if (currentFrame < totalFrames - 1) {
        currentFrame++;
        showFrame(currentFrame);
    }
}

function lastFrame() {
    currentFrame = totalFrames - 1;
    showFrame(currentFrame);
}

function togglePlay() {
    isPlaying = !isPlaying;
    const playBtn = document.getElementById('play-btn');

    if (isPlaying) {
        playBtn.innerHTML = '&#9616;&#9616;'; // Pause symbol
        startPlayback();
    } else {
        playBtn.innerHTML = '&#9658;'; // Play symbol
        stopPlayback();
    }
}

function startPlayback() {
    if (playbackInterval) return;

    playbackInterval = setInterval(() => {
        if (currentFrame < totalFrames - 1) {
            currentFrame++;
            showFrame(currentFrame);
        } else {
            // Loop back to start
            currentFrame = 0;
            showFrame(currentFrame);
        }
    }, playbackSpeed);
}

function stopPlayback() {
    if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
    }
}

function updateSpeed() {
    const speedSelect = document.getElementById('speed');
    playbackSpeed = parseInt(speedSelect.value);

    // Restart playback if playing
    if (isPlaying) {
        stopPlayback();
        startPlayback();
    }
}

// ============================================================================
// Pan/Zoom Controls
// ============================================================================
function zoomIn() {
    viewTransform.scale *= 1.2;
    updateView();
}

function zoomOut() {
    viewTransform.scale /= 1.2;
    updateView();
}

function resetView() {
    viewTransform = { x: 0, y: 0, scale: 1.0 };
    updateView();
}

function updateView() {
    // Update zoom level display
    const zoomLevel = document.getElementById('zoom-level');
    if (zoomLevel) {
        zoomLevel.textContent = Math.round(viewTransform.scale * 100) + '%';
    }

    // Resize canvases to maintain crisp rendering at current zoom level
    resizeCanvases();

    // Redraw at new resolution
    if (typeof renderFrame === 'function') {
        renderFrame(currentFrame);
    }
}

// Resize canvases to match current zoom and DPI for crisp rendering
function resizeCanvases() {
    if (!staticCanvas || !dynamicCanvas || !frameContainer) return;

    const dpr = window.devicePixelRatio || 1;
    const containerRect = frameContainer.getBoundingClientRect();

    // Base canvas size matches container
    const baseWidth = containerRect.width;
    const baseHeight = containerRect.height;

    // Internal canvas resolution accounts for zoom and device pixel ratio
    const resolution = dpr * viewTransform.scale;
    const canvasWidth = Math.round(baseWidth * resolution);
    const canvasHeight = Math.round(baseHeight * resolution);

    // Set internal resolution (actual pixels)
    staticCanvas.width = canvasWidth;
    staticCanvas.height = canvasHeight;
    dynamicCanvas.width = canvasWidth;
    dynamicCanvas.height = canvasHeight;

    // Set display size (CSS) - always full container, only pan with translate
    staticCanvas.style.width = baseWidth + 'px';
    staticCanvas.style.height = baseHeight + 'px';
    dynamicCanvas.style.width = baseWidth + 'px';
    dynamicCanvas.style.height = baseHeight + 'px';

    // Apply pan offset (but not scale - that's handled by canvas resolution)
    staticCanvas.style.transform = `translate(-50%, -50%) translate(${viewTransform.x}px, ${viewTransform.y}px)`;
    dynamicCanvas.style.transform = `translate(-50%, -50%) translate(${viewTransform.x}px, ${viewTransform.y}px)`;

    // Update global canvas scale for rendering calculations
    // This is now: pixels per mm, accounting for zoom and DPI
    canvasScale = (baseWidth / (boardMaxX - boardMinX)) * resolution;
}

// Mouse wheel zoom
if (frameContainer) {
    frameContainer.addEventListener('wheel', (e) => {
        e.preventDefault();

        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const oldScale = viewTransform.scale;
        viewTransform.scale *= delta;

        // Zoom toward mouse position
        const rect = frameContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left - rect.width / 2;
        const mouseY = e.clientY - rect.top - rect.height / 2;

        viewTransform.x = mouseX - (mouseX - viewTransform.x) * (viewTransform.scale / oldScale);
        viewTransform.y = mouseY - (mouseY - viewTransform.y) * (viewTransform.scale / oldScale);

        updateView();
    });

    // Drag to pan
    frameContainer.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // Left click
            isDragging = true;
            frameContainer.classList.add('dragging');
            dragStart = { x: e.clientX, y: e.clientY };
            dragOffset = { x: viewTransform.x, y: viewTransform.y };
        }
    });

    frameContainer.addEventListener('mousemove', (e) => {
        if (isDragging) {
            viewTransform.x = dragOffset.x + (e.clientX - dragStart.x);
            viewTransform.y = dragOffset.y + (e.clientY - dragStart.y);
            updateView();
        }
    });

    frameContainer.addEventListener('mouseup', () => {
        isDragging = false;
        frameContainer.classList.remove('dragging');
    });

    frameContainer.addEventListener('mouseleave', () => {
        isDragging = false;
        frameContainer.classList.remove('dragging');
    });

    // Component selection for trail highlighting
    frameContainer.addEventListener('click', (e) => {
        if (isDragging) return; // Don't select if we just finished dragging

        // Get click position relative to canvas CSS display
        const rect = dynamicCanvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Convert screen coordinates to board coordinates
        // Canvas internal resolution accounts for DPI and zoom, but click is in CSS pixels
        const dpr = window.devicePixelRatio || 1;
        const resolution = dpr * viewTransform.scale;

        // Get canvas center in CSS pixels
        const cssWidth = parseFloat(dynamicCanvas.style.width);
        const cssHeight = parseFloat(dynamicCanvas.style.height);
        const canvasCenterX = cssWidth / 2;
        const canvasCenterY = cssHeight / 2;

        // Convert click position to board coordinates
        // canvasScale already accounts for resolution, so divide by it
        const boardX = (clickX - canvasCenterX) * (boardMaxX - boardMinX) / cssWidth + boardCenterX;
        const boardY = (clickY - canvasCenterY) * (boardMaxY - boardMinY) / cssHeight + boardCenterY;

        // Hit-test: find which component was clicked
        let clickedComponent = null;
        const currentFrameData = frames[currentFrame];

        for (const [ref, compPos] of Object.entries(currentFrameData.components)) {
            const props = staticProps[ref];
            if (!props) continue;

            const [x, y, rotation] = compPos;
            const halfWidth = props.width / 2;
            const halfHeight = props.height / 2;

            // Simple bounding box hit test (ignoring rotation for now)
            if (boardX >= x - halfWidth && boardX <= x + halfWidth &&
                boardY >= y - halfHeight && boardY <= y + halfHeight) {
                clickedComponent = ref;
                break;
            }
        }

        if (clickedComponent) {
            // Toggle selection: clicking same component deselects
            if (selectedComponent === clickedComponent) {
                selectedComponent = null;
            } else {
                selectedComponent = clickedComponent;
            }
        } else {
            // Clicking empty space deselects
            selectedComponent = null;
        }

        // Redraw to show updated selection
        renderFrame(currentFrame);
    });
}

// ============================================================================
// Layer Visibility Controls
// ============================================================================
function updateLayers() {
    layerVisibility.top = document.getElementById('show-top')?.checked ?? true;
    layerVisibility.bottom = document.getElementById('show-bottom')?.checked ?? true;
    layerVisibility.edge = document.getElementById('show-edge')?.checked ?? true;
    layerVisibility.grid = document.getElementById('show-grid')?.checked || false;
    layerVisibility.ratsnest = document.getElementById('show-ratsnest')?.checked || false;
    layerVisibility.trails = document.getElementById('show-trails')?.checked || false;
    layerVisibility.forces = document.getElementById('show-forces')?.checked || false;
    layerVisibility.pads = document.getElementById('show-pads')?.checked || false;
    layerVisibility.labels = document.getElementById('show-labels')?.checked || false;
    layerVisibility.groups = document.getElementById('show-groups')?.checked || false;

    // Redraw current frame
    if (typeof renderFrame === 'function') {
        renderFrame(currentFrame);
    }
}

function updateModuleVisibility() {
    // Update module visibility from checkboxes
    const checkboxes = document.querySelectorAll('[id^="show-module-"]');
    checkboxes.forEach(checkbox => {
        const moduleName = checkbox.id.replace('show-module-', '').replace(/-/g, '.');
        moduleVisibility[moduleName] = checkbox.checked;
    });

    // Redraw current frame
    if (typeof renderFrame === 'function') {
        renderFrame(currentFrame);
    }
}

// Initialize module visibility
function initializeModuleVisibility() {
    const checkboxes = document.querySelectorAll('[id^="show-module-"]');
    checkboxes.forEach(checkbox => {
        const moduleName = checkbox.id.replace('show-module-', '').replace(/-/g, '.');
        moduleVisibility[moduleName] = checkbox.checked;
    });
}

// ============================================================================
// Energy Graph
// ============================================================================
function drawEnergyGraph() {
    const canvas = document.getElementById('energy-canvas');
    if (!canvas || typeof frames === 'undefined') return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 50;

    if (frames.length === 0) return;

    // Clear
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, 0, width, height);

    // Find max values
    let maxEnergy = 0;
    let maxWireLength = 0;
    frames.forEach(frame => {
        maxEnergy = Math.max(maxEnergy, frame.energy || 0);
        maxWireLength = Math.max(maxWireLength, frame.total_wire_length || 0);
    });

    // Draw energy line (blue)
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.beginPath();
    frames.forEach((frame, i) => {
        const x = (i / (frames.length - 1)) * width;
        const y = height - (frame.energy / maxEnergy) * (height - 4) - 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw wire length line (green)
    ctx.strokeStyle = '#2ecc71';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    frames.forEach((frame, i) => {
        const x = (i / (frames.length - 1)) * width;
        const y = height - (frame.total_wire_length / maxWireLength) * (height - 4) - 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw current frame indicator
    const currentX = (currentFrame / (frames.length - 1)) * width;
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, height);
    ctx.stroke();

    // Update range label
    const rangeLabel = document.getElementById('graph-range');
    if (rangeLabel) {
        rangeLabel.textContent = `0-${frames.length - 1}`;
    }
}

function seekToFrame(e) {
    if (typeof frames === 'undefined') return;

    const canvas = document.getElementById('energy-canvas');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const fraction = x / rect.width;
    const frame = Math.round(fraction * (frames.length - 1));

    currentFrame = Math.max(0, Math.min(frames.length - 1, frame));
    showFrame(currentFrame);
}

function startSeeking(e) {
    isSeeking = true;
    const graph = document.querySelector('.energy-graph');
    if (graph) graph.classList.add('seeking');
    seekToFrame(e);
}

function continueSeeking(e) {
    if (isSeeking) {
        seekToFrame(e);
    }
}

function stopSeeking() {
    isSeeking = false;
    const graph = document.querySelector('.energy-graph');
    if (graph) graph.classList.remove('seeking');
}

// ============================================================================
// Frame Display
// ============================================================================
function updateInfoBar(frame) {
    document.getElementById('frame-num').textContent = currentFrame;
    document.getElementById('phase').textContent = frame.phase || '-';
    document.getElementById('iteration').textContent = frame.iteration || 0;
    document.getElementById('energy').textContent = (frame.energy || 0).toFixed(2);
    document.getElementById('max-move').textContent = (frame.max_move || 0).toFixed(3);
    document.getElementById('overlaps').textContent = frame.overlap_count || 0;
    document.getElementById('wire-length').textContent = (frame.total_wire_length || 0).toFixed(1) + 'mm';
}

function showFrame(frameIndex) {
    if (typeof frames === 'undefined' || frameIndex < 0 || frameIndex >= frames.length) return;

    currentFrame = frameIndex;
    const frame = frames[frameIndex];

    // Update info bar
    updateInfoBar(frame);

    // Redraw frame
    if (typeof renderFrame === 'function') {
        renderFrame(frameIndex);
    }

    // Update energy graph
    if (typeof drawEnergyGraph === 'function') {
        drawEnergyGraph();
    }
}

// ============================================================================
// Initialization
// ============================================================================
function initialize() {
    // Initialize module visibility
    initializeModuleVisibility();

    // Initialize canvas resolution for crisp rendering
    resizeCanvases();

    // Draw energy graph if available
    if (typeof drawEnergyGraph === 'function' && typeof frames !== 'undefined') {
        drawEnergyGraph();

        // Redraw on window resize
        window.addEventListener('resize', () => {
            resizeCanvases();
            drawEnergyGraph();
            if (typeof renderFrame === 'function') {
                renderFrame(currentFrame);
            }
        });
    }

    // Show first frame
    if (typeof frames !== 'undefined' && frames.length > 0) {
        showFrame(0);
    }
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}
'''

    return js_code


def generate_canvas_render_javascript() -> str:
    """Generate JavaScript for Canvas-based rendering with all features.

    Returns:
        JavaScript code for Canvas rendering
    """

    js_code = '''
// ============================================================================
// Canvas Setup and Coordinate Transform
// ============================================================================

// Calculate canvas scale and offsets
const boardWidth = boardBounds.maxX - boardBounds.minX;
const boardHeight = boardBounds.maxY - boardBounds.minY;
const boardCenterX = (boardBounds.minX + boardBounds.maxX) / 2;
const boardCenterY = (boardBounds.minY + boardBounds.maxY) / 2;

// Scale to fit board in canvas with some padding
const canvasScale = Math.min(
    (dynamicCanvas.width - 100) / boardWidth,
    (dynamicCanvas.height - 100) / boardHeight
);

function setupCanvasTransform(ctx) {
    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Translate to center of canvas
    ctx.translate(dynamicCanvas.width / 2, dynamicCanvas.height / 2);

    // Scale from mm to pixels
    ctx.scale(canvasScale, canvasScale);

    // Translate to center board at origin
    ctx.translate(-boardCenterX, -boardCenterY);
}

// ============================================================================
// Canvas Rendering with Layer Visibility
// ============================================================================

function renderFrame(frameIndex) {
    if (!frames || frameIndex < 0 || frameIndex >= frames.length) return;

    const frame = frames[frameIndex];
    const ctx = dynamicCanvas.getContext('2d');

    // Reset transform to identity before clearing
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Clear dynamic canvas
    ctx.clearRect(0, 0, dynamicCanvas.width, dynamicCanvas.height);

    // Set up coordinate transform
    setupCanvasTransform(ctx);

    // Render layers in order based on visibility
    if (layerVisibility.edge) renderBoardOutline(ctx);
    if (layerVisibility.grid) renderGrid(ctx);
    if (layerVisibility.groups) renderModuleGroups(ctx, frame);
    if (layerVisibility.ratsnest) renderRatsnest(ctx, frame);
    if (layerVisibility.trails) renderTrails(ctx, frame, frameIndex);
    renderComponents(ctx, frame);
    if (layerVisibility.pads) renderPads(ctx, frame);
    if (layerVisibility.forces) renderForces(ctx, frame);
    if (layerVisibility.labels) renderLabels(ctx, frame);
}

function renderBoardOutline(ctx) {
    // Draw board outline - use polygon if available, otherwise rectangle (matches SVG)
    if (boardOutline && boardOutline.polygon && boardOutline.polygon.length >= 3) {
        // Draw actual polygon outline
        ctx.fillStyle = '#16213e'; // Board fill color
        ctx.strokeStyle = '#0f3460'; // Board outline color
        ctx.lineWidth = 0.2; // 2px equivalent in board units

        ctx.beginPath();
        ctx.moveTo(boardOutline.polygon[0][0], boardOutline.polygon[0][1]);
        for (let i = 1; i < boardOutline.polygon.length; i++) {
            ctx.lineTo(boardOutline.polygon[i][0], boardOutline.polygon[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Draw holes/cutouts with dashed line
        if (boardOutline.holes && boardOutline.holes.length > 0) {
            ctx.fillStyle = '#1a1a2e'; // Background color for holes
            ctx.strokeStyle = '#0f3460';
            ctx.lineWidth = 0.15; // 1.5px equivalent
            ctx.setLineDash([0.4, 0.2]); // 4px, 2px dashed

            boardOutline.holes.forEach(hole => {
                if (hole.length >= 3) {
                    ctx.beginPath();
                    ctx.moveTo(hole[0][0], hole[0][1]);
                    for (let i = 1; i < hole.length; i++) {
                        ctx.lineTo(hole[i][0], hole[i][1]);
                    }
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }
            });

            ctx.setLineDash([]);
        }
    } else {
        // Fall back to rectangle
        ctx.fillStyle = '#16213e';
        ctx.strokeStyle = '#0f3460';
        ctx.lineWidth = 0.2;

        ctx.fillRect(
            boardBounds.minX,
            boardBounds.minY,
            boardBounds.maxX - boardBounds.minX,
            boardBounds.maxY - boardBounds.minY
        );
        ctx.strokeRect(
            boardBounds.minX,
            boardBounds.minY,
            boardBounds.maxX - boardBounds.minX,
            boardBounds.maxY - boardBounds.minY
        );
    }
}

function renderGrid(ctx) {
    // Match SVG grid styling: #2a2a4e color, 0.5px stroke width
    ctx.strokeStyle = '#2a2a4e';
    ctx.lineWidth = 0.05; // 0.5px equivalent in board units

    const gridSpacing = 10; // mm
    const bounds = boardBounds;

    // Start from nearest grid point (matches SVG logic)
    const gridStartX = Math.ceil(bounds.minX / gridSpacing) * gridSpacing;
    const gridStartY = Math.ceil(bounds.minY / gridSpacing) * gridSpacing;

    // Vertical lines
    for (let x = gridStartX; x <= bounds.maxX; x += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, bounds.minY);
        ctx.lineTo(x, bounds.maxY);
        ctx.stroke();
    }

    // Horizontal lines
    for (let y = gridStartY; y <= bounds.maxY; y += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(bounds.minX, y);
        ctx.lineTo(bounds.maxX, y);
        ctx.stroke();
    }
}

function renderModuleGroups(ctx, frame) {
    if (!frame.modules) return;

    // Group components by module
    const moduleGroups = {};
    for (const [ref, moduleName] of Object.entries(frame.modules)) {
        // Skip if module is hidden or component is not visible due to layer filtering
        if (!moduleVisibility[moduleName]) continue;
        if (!isComponentVisible(ref)) continue;

        if (!moduleGroups[moduleName]) {
            moduleGroups[moduleName] = [];
        }
        moduleGroups[moduleName].push(ref);
    }

    // Draw bounding box for each module
    for (const [moduleName, refs] of Object.entries(moduleGroups)) {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        refs.forEach(ref => {
            const comp = frame.components[ref];
            const props = staticProps[ref];
            if (!comp || !props) return;

            const [x, y, rotation] = comp;
            minX = Math.min(minX, x - props.width/2);
            minY = Math.min(minY, y - props.height/2);
            maxX = Math.max(maxX, x + props.width/2);
            maxY = Math.max(maxY, y + props.height/2);
        });

        if (isFinite(minX)) {
            const padding = 1.5; // Match SVG padding
            const color = moduleColors[moduleName] || '#9b59b6';

            // Draw semi-transparent fill (matches SVG fill-opacity="0.15")
            ctx.fillStyle = color + '26'; // 15% opacity (0.15 * 255 = 38.25 ≈ 0x26)
            ctx.fillRect(minX - padding, minY - padding,
                        maxX - minX + 2*padding, maxY - minY + 2*padding);

            // Draw dashed border (matches SVG stroke)
            ctx.strokeStyle = color;
            ctx.lineWidth = 0.15; // 1.5px equivalent in board units
            ctx.setLineDash([0.4, 0.2]); // 4px, 2px equivalent in board units
            ctx.strokeRect(minX - padding, minY - padding,
                          maxX - minX + 2*padding, maxY - minY + 2*padding);
            ctx.setLineDash([]);

            // Draw module label at top of bounding box (matches SVG positioning)
            const labelX = (minX + maxX) / 2;
            const labelY = minY - padding;

            ctx.save();

            // Reset transform to render text in screen pixels (matches SVG font-size="10")
            ctx.setTransform(1, 0, 0, 1, 0, 0);

            // Convert board coordinates to screen coordinates
            const screenX = dynamicCanvas.width / 2 + (labelX - boardCenterX) * canvasScale;
            const screenY = dynamicCanvas.height / 2 + (labelY - boardCenterY) * canvasScale - 5; // 5px above like SVG

            ctx.font = 'bold 10px monospace'; // Match SVG font-size="10"
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillStyle = color;
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.strokeText(moduleName, screenX, screenY);
            ctx.fillText(moduleName, screenX, screenY);
            ctx.restore();
        }
    }
}

function renderRatsnest(ctx, frame) {
    if (!frame.connections) return;

    ctx.lineWidth = 0.1; // In board units (mm)

    frame.connections.forEach(([ref1, ref2, net]) => {
        const comp1 = frame.components[ref1];
        const comp2 = frame.components[ref2];
        if (!comp1 || !comp2) return;

        // Check layer visibility for both components
        if (!isComponentVisible(ref1) || !isComponentVisible(ref2)) return;

        // Color code based on net type (matches SVG version)
        let lineColor = 'rgba(74, 111, 165, 0.4)'; // Default dim blue for signals
        if (net) {
            const netLower = net.toLowerCase();
            if (netLower.includes('gnd') || netLower.includes('vss')) {
                lineColor = 'rgba(46, 204, 113, 0.4)'; // Green for ground
            } else if (netLower.includes('vcc') || netLower.includes('vdd') || netLower.includes('pwr')) {
                lineColor = 'rgba(231, 76, 60, 0.4)'; // Red for power
            }
        }

        ctx.strokeStyle = lineColor;
        ctx.beginPath();
        ctx.moveTo(comp1[0], comp1[1]);
        ctx.lineTo(comp2[0], comp2[1]);
        ctx.stroke();
    });
}

function renderTrails(ctx, frame, frameIndex) {
    if (frameIndex < 1) return;

    // Get all component refs from current frame
    const currentComponents = frame.components;

    for (const ref of Object.keys(currentComponents)) {
        // If a component is selected, only show its trail
        if (selectedComponent && ref !== selectedComponent) continue;

        // Collect position history for this component across all frames
        const history = [];
        for (let i = 0; i <= frameIndex; i++) {
            const frameData = frames[i];
            const pos = frameData.components[ref];
            if (pos) {
                history.push({ x: pos[0], y: pos[1], frameIndex: i });
            }
        }

        if (history.length < 2) continue;

        // Check if component moved significantly
        const start = history[0];
        const end = history[history.length - 1];
        const totalDist = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
        if (totalDist < 0.5) continue; // Skip components that barely moved

        const isSelected = (ref === selectedComponent);
        const trailColor = isSelected ? '#00d4ff' : '#ff6b6b'; // Cyan for selected, red for normal
        const trailOpacityMult = isSelected ? 1.2 : 1.0;

        // Draw connecting line segments (faded)
        if (history.length > 1) {
            ctx.strokeStyle = trailColor;
            ctx.lineWidth = isSelected ? 0.2 : 0.1; // 2px for selected, 1px for normal
            ctx.globalAlpha = isSelected ? 0.7 : 0.4;

            ctx.beginPath();
            ctx.moveTo(history[0].x, history[0].y);
            for (let i = 1; i < history.length; i++) {
                ctx.lineTo(history[i].x, history[i].y);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }

        // Draw breadcrumb dots along the path
        for (let i = 0; i < history.length; i++) {
            const pos = history[i];
            // Fade older positions
            const age = (frameIndex - pos.frameIndex) / Math.max(frameIndex, 1);
            const baseOpacity = 0.2 + (1 - age) * 0.6;
            const opacity = Math.min(1, baseOpacity * trailOpacityMult);

            // Selected trails have larger dots
            const baseRadius = isSelected ? 2 + (1 - age) * 2 : 1.5 + (1 - age) * 1.5;
            const radius = baseRadius / canvasScale; // Convert px to board units

            ctx.fillStyle = trailColor;
            ctx.globalAlpha = opacity;

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1.0;

        // Mark initial position (START) with a special circle
        const startColor = isSelected ? '#2ecc71' : trailColor; // Green for selected start
        ctx.strokeStyle = startColor;
        ctx.lineWidth = isSelected ? 0.2 : 0.15; // Thicker for selected
        ctx.globalAlpha = isSelected ? 1.0 : 0.7;

        // Filled circle for selected, stroked for normal
        if (isSelected) {
            ctx.fillStyle = startColor;
            ctx.beginPath();
            ctx.arc(start.x, start.y, 5 / canvasScale, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        } else {
            ctx.beginPath();
            ctx.arc(start.x, start.y, 3 / canvasScale, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;

        // Mark final position (END) with a filled circle and direction indicator when selected
        if (isSelected) {
            const endColor = '#00d4ff'; // Cyan for end
            ctx.fillStyle = endColor;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 0.2; // 2px equivalent

            ctx.beginPath();
            ctx.arc(end.x, end.y, 6 / canvasScale, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Add direction indicator from second-to-last to end
            if (history.length >= 2) {
                const prev = history[history.length - 2];
                const dx = end.x - prev.x;
                const dy = end.y - prev.y;
                const len = Math.sqrt(dx * dx + dy * dy);
                if (len > 0.1) {
                    // Draw a small direction line
                    ctx.strokeStyle = endColor;
                    ctx.lineWidth = 0.3; // 3px equivalent
                    ctx.globalAlpha = 0.8;

                    ctx.beginPath();
                    ctx.moveTo(end.x - dx/len * 3 / canvasScale, end.y - dy/len * 3 / canvasScale);
                    ctx.lineTo(end.x, end.y);
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;
                }
            }
        }
    }
}

// Helper function to check if a component is visible based on layer filtering
function isComponentVisible(ref) {
    // Check layer visibility - defaults to visible if layer info not available
    const layer = componentLayers[ref];
    if (layer === 'top' && !layerVisibility.top) return false;
    if (layer === 'bottom' && !layerVisibility.bottom) return false;
    return true;
}

function renderComponents(ctx, frame) {
    // Build set of overlapping components
    const overlappingRefs = new Set();
    if (frame.overlaps) {
        frame.overlaps.forEach(overlap => {
            overlappingRefs.add(overlap[0]);
            overlappingRefs.add(overlap[1]);
        });
    }

    for (const [ref, comp] of Object.entries(frame.components)) {
        const props = staticProps[ref];
        if (!props) continue;

        // Check layer visibility
        if (!isComponentVisible(ref)) continue;

        // Check module visibility
        const moduleName = frame.modules?.[ref];
        if (moduleName && !moduleVisibility[moduleName]) continue;

        const [x, y, rotation] = comp;

        ctx.save();
        ctx.translate(x, y);
        ctx.rotate((rotation * Math.PI) / 180);

        // Component body - match SVG colors and opacity
        const color = moduleName ? (moduleColors[moduleName] || '#3498db') : '#3498db';
        ctx.fillStyle = color + '99'; // 60% opacity (0.6 * 255 = 153 = 0x99)
        ctx.fillRect(-props.width/2, -props.height/2, props.width, props.height);

        // Stroke color: red for overlaps, white otherwise (matches SVG)
        ctx.strokeStyle = overlappingRefs.has(ref) ? '#ff0000' : '#ffffff';
        ctx.lineWidth = overlappingRefs.has(ref) ? 0.2 : 0.1; // 2px vs 1px equivalent in board units
        ctx.strokeRect(-props.width/2, -props.height/2, props.width, props.height);

        // Add selection highlight if this component is selected
        if (selectedComponent === ref) {
            ctx.strokeStyle = '#00d4ff'; // Cyan selection highlight
            ctx.lineWidth = 0.3; // 3px equivalent - thicker to stand out
            ctx.setLineDash([0.4, 0.2]); // Dashed pattern
            ctx.strokeRect(-props.width/2, -props.height/2, props.width, props.height);
            ctx.setLineDash([]); // Reset to solid
        }

        ctx.restore();
    }
}

function getPadColor(netName) {
    const name = (netName || '').trim().toLowerCase();
    if (!name) return '#95a5a6';

    if (name.includes('gnd') || name.includes('vss')) return '#2ecc71';
    if (name.includes('vcc') || name.includes('vdd') || name.includes('pwr')) return '#e74c3c';
    if (name.includes('usb')) return '#e67e22';

    const tokens = name.split(/[^a-z0-9]+/).filter(Boolean);
    if (tokens.some(token => token.startsWith('can'))) return '#c0392b';
    if (tokens.some(token => token.startsWith('swd') || token === 'swclk' || token === 'swo')) return '#2c3e50';
    if (name.includes('i2c') || tokens.includes('scl') || tokens.includes('sda')) return '#1abc9c';
    if (name.includes('spi') || tokens.some(token => ['sck', 'sclk', 'mosi', 'miso', 'cs', 'csn', 'ncs', 'ss'].includes(token))) {
        return '#f1c40f';
    }
    if (tokens.some(token => token.startsWith('uart') || token.startsWith('usart'))) return '#8e44ad';

    return '#3498db';
}

function renderPads(ctx, frame) {
    for (const [ref, comp] of Object.entries(frame.components)) {
        const props = staticProps[ref];
        if (!props || !props.pads) continue;

        // Check layer visibility
        if (!isComponentVisible(ref)) continue;

        // Check module visibility
        const moduleName = frame.modules?.[ref];
        if (moduleName && !moduleVisibility[moduleName]) continue;

        const [x, y, rotation] = comp;
        const rad = (rotation * Math.PI) / 180;
        const cos_r = Math.cos(rad);
        const sin_r = Math.sin(rad);

        props.pads.forEach(pad => {
            // Support both old (5-element) and new (6-element) format
            const [px_rel, py_rel, pw, ph] = pad;
            const padRot = pad.length >= 6 ? pad[4] : 0;
            const net = pad.length >= 6 ? pad[5] : pad[4];

            // Rotate pad position
            const px = x + px_rel * cos_r - py_rel * sin_r;
            const py = y + px_rel * sin_r + py_rel * cos_r;

            // Total rotation = component rotation + pad's own rotation
            const totalRotRad = (rotation + padRot) * Math.PI / 180;

            // Pad color based on net type (matches SVG version)
            const padColor = getPadColor(net);

            ctx.fillStyle = padColor;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 0.05; // 0.5px equivalent in board units

            // Apply rotation transform for the pad shape
            ctx.save();
            ctx.translate(px, py);
            ctx.rotate(totalRotRad);
            ctx.fillRect(-pw/2, -ph/2, pw, ph);
            ctx.strokeRect(-pw/2, -ph/2, pw, ph);
            ctx.restore();
        });
    }
}

function renderForces(ctx, frame) {
    if (!frame.forces) return;

    for (const [ref, forceVectors] of Object.entries(frame.forces)) {
        const comp = frame.components[ref];
        if (!comp) continue;

        // Check layer visibility
        if (!isComponentVisible(ref)) continue;

        // Check module visibility
        const moduleName = frame.modules?.[ref];
        if (moduleName && !moduleVisibility[moduleName]) continue;

        const [x, y] = comp;

        forceVectors.forEach(([fx, fy, ftype]) => {
            // Force type color mapping (matches SVG version and Force Types legend)
            const forceColors = {
                'repulsion': '#e74c3c',      // Red
                'attraction': '#2ecc71',     // Green
                'boundary': '#3498db',       // Blue
                'constraint': '#f39c12',     // Orange
                'alignment': '#9b59b6'       // Purple
            };
            const color = forceColors[ftype] || '#e74c3c'; // Default to repulsion color

            ctx.strokeStyle = color;
            ctx.lineWidth = 0.2; // In board units (mm)

            // Scale force for visibility
            const scale = 5;
            const ex = x + fx * scale;
            const ey = y + fy * scale;

            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(ex, ey);
            ctx.stroke();

            // Arrow head
            const angle = Math.atan2(fy, fx);
            const arrowSize = 1;
            ctx.beginPath();
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - arrowSize * Math.cos(angle - 0.5), ey - arrowSize * Math.sin(angle - 0.5));
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - arrowSize * Math.cos(angle + 0.5), ey - arrowSize * Math.sin(angle + 0.5));
            ctx.stroke();
        });
    }
}

function renderLabels(ctx, frame) {
    for (const [ref, comp] of Object.entries(frame.components)) {
        // Check layer visibility
        if (!isComponentVisible(ref)) continue;

        // Check module visibility
        const moduleName = frame.modules?.[ref];
        if (moduleName && !moduleVisibility[moduleName]) continue;

        const [x, y, rotation] = comp;
        const props = staticProps[ref];
        if (!props) continue;

        // Calculate font size in screen pixels (matches SVG: max(7, min(11, ts(min(w, h) * 0.35))))
        const componentSize = Math.min(props.width, props.height);
        const scaledSize = componentSize * canvasScale;
        const fontSize = Math.max(7, Math.min(11, scaledSize * 0.35));
        const labelPadding = 3; // Pixels from component edge

        // Estimate label dimensions
        const labelWidth = ref.length * fontSize * 0.6;
        const labelHeight = fontSize;

        // Component dimensions in screen pixels
        const compWidthPx = props.width * canvasScale;
        const compHeightPx = props.height * canvasScale;
        const compIsSmall = compWidthPx < 25 || compHeightPx < 25;

        // Choose position: prefer above/below for wide, left/right for tall (matches SVG logic)
        let labelX, labelY, textAlign, textBaseline;

        if (props.width >= props.height || compIsSmall) {
            // Place above component
            labelX = x;
            labelY = y - props.height/2 - labelPadding / canvasScale - labelHeight / (2 * canvasScale);
            textAlign = 'center';
            textBaseline = 'middle';
        } else {
            // Place to the right of component
            labelX = x + props.width/2 + labelPadding / canvasScale + labelWidth / (2 * canvasScale);
            labelY = y;
            textAlign = 'center';
            textBaseline = 'middle';
        }

        // Save context for font rendering in screen space
        ctx.save();

        // Reset transform to render text in screen pixels (not board coordinates)
        ctx.setTransform(1, 0, 0, 1, 0, 0);

        // Convert board coordinates to screen coordinates
        const screenX = dynamicCanvas.width / 2 + (labelX - boardCenterX) * canvasScale;
        const screenY = dynamicCanvas.height / 2 + (labelY - boardCenterY) * canvasScale;

        ctx.font = fontSize + 'px monospace';
        ctx.textAlign = textAlign;
        ctx.textBaseline = textBaseline;

        // Draw text outline (stroke) for better readability
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.strokeText(ref, screenX, screenY);

        // Draw text fill
        ctx.fillStyle = '#dddddd'; // Light gray to match SVG version
        ctx.fillText(ref, screenX, screenY);

        ctx.restore();
    }
}
'''

    return js_code
