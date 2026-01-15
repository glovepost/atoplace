/**
 * SVG Delta Viewer - JavaScript for SVG-based delta playback.
 *
 * Updates SVG DOM elements directly instead of replacing entire SVG strings.
 * This provides:
 * - Crisp vector quality at any zoom level
 * - Better performance than full SVG replacement (30-40 FPS vs 6 FPS)
 * - Smaller file sizes with delta compression
 *
 * Required globals (injected by Python template):
 * - boardBounds: {minX, minY, maxX, maxY, scale, padding}
 * - staticProps: {ref: {width, height, pads: [[x, y, w, h, net], ...]}}
 * - deltaFrames: [{changed_components: {ref: [x, y, rotation]}, ...}]
 * - totalFrames: number
 * - moduleColors: {module_name: color}
 * - componentLayers: {ref: 'top'|'bottom'}
 * - netlist: {net_name: [[ref, padIndex], ...]} (optional)
 */

// ============================================================================
// Coordinate Transform Functions
// ============================================================================
function tx(x) {
    return boardBounds.padding + (x - boardBounds.minX) * boardBounds.scale;
}

function ty(y) {
    // Direct Y mapping (no flip)
    return boardBounds.padding + (y - boardBounds.minY) * boardBounds.scale;
}

function ts(size) {
    return size * boardBounds.scale;
}

// ============================================================================
// Global State
// ============================================================================
let currentFrame = 0;
// totalFrames is declared inline with data
let isPlaying = false;
let playbackInterval = null;
let playbackSpeed = 100; // ms per frame
let lastReconstructedFrame = -1; // Track which frame we last reconstructed
let isSeeking = false;

// Pan/zoom state
let viewTransform = { x: 0, y: 0, scale: 1.0 };
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let dragOffset = { x: 0, y: 0 };

// Layer visibility
let layerVisibility = {
    top: true,
    bottom: true,
    edge: true,
    grid: false,
    ratsnest: false,
    trails: false,
    forces: false,
    pads: false,
    labels: true,
    groups: true
};

// Module visibility state
let moduleVisibility = {};

// Component selection state
let selectedComponent = null;

// Current frame state (accumulated from deltas)
let currentState = {
    components: {},  // ref -> [x, y, rotation]
    modules: {},     // ref -> module_name
    overlaps: [],
    forces: [],
    energy: 0,
    max_move: 0,
    overlap_count: 0,
    wire_length: 0
};

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

function sanitizeModuleName(name) {
    return (name || 'default').replace(/[._\s]/g, '-');
}

function applyModuleClass(el, moduleName) {
    if (!el) return;
    const safeModuleName = sanitizeModuleName(moduleName);
    Array.from(el.classList).forEach(cls => {
        if (cls.startsWith('module-')) {
            el.classList.remove(cls);
        }
    });
    el.classList.add(`module-${safeModuleName}`);
}

// ============================================================================
// Frame Container Setup
// ============================================================================
const frameContainer = document.getElementById('frame-container');
const svgContainer = frameContainer;

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
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Apply transform to SVG
    svg.style.transform = `translate(-50%, -50%) translate(${viewTransform.x}px, ${viewTransform.y}px) scale(${viewTransform.scale})`;

    // Update zoom level display
    const zoomLevel = document.getElementById('zoom-level');
    if (zoomLevel) {
        zoomLevel.textContent = Math.round(viewTransform.scale * 100) + '%';
    }
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
        if (isDragging) return;

        // Check if we clicked on a component
        const component = e.target.closest('.component');
        if (component) {
            const ref = component.getAttribute('data-ref');
            if (selectedComponent === ref) {
                // Deselect
                selectedComponent = null;
            } else {
                selectedComponent = ref;
            }
            updateComponentSelection();
            drawTrails();
        } else {
            // Clicked empty space
            if (selectedComponent) {
                selectedComponent = null;
                updateComponentSelection();
                drawTrails();
            }
        }
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
    layerVisibility.groups = document.getElementById('show-groups')?.checked ?? true;

    updateVisibility();
}

function updateVisibility() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Update layer visibility classes
    svg.querySelectorAll('.board-outline').forEach(el => {
        el.style.display = layerVisibility.edge ? '' : 'none';
    });
    svg.querySelectorAll('.comp-top').forEach(el => {
        el.style.display = layerVisibility.top ? '' : 'none';
    });
    svg.querySelectorAll('.comp-bottom').forEach(el => {
        el.style.display = layerVisibility.bottom ? '' : 'none';
    });
    svg.querySelectorAll('.grid-layer, .grid-line').forEach(el => {
        el.style.display = layerVisibility.grid ? '' : 'none';
    });
    svg.querySelectorAll('.ratsnest-layer, .ratsnest').forEach(el => {
        el.style.display = layerVisibility.ratsnest ? '' : 'none';
    });
    svg.querySelectorAll('.force-layer, .force-vector').forEach(el => {
        el.style.display = layerVisibility.forces ? '' : 'none';
    });
    svg.querySelectorAll('.pad-element').forEach(el => {
        const isTop = el.classList.contains('comp-top');
        const isBottom = el.classList.contains('comp-bottom');
        const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);
        el.style.display = (layerVisibility.pads && layerVisible) ? '' : 'none';
    });
    svg.querySelectorAll('.ref-label').forEach(el => {
        const isTop = el.classList.contains('comp-top');
        const isBottom = el.classList.contains('comp-bottom');
        const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);
        el.style.display = (layerVisibility.labels && layerVisible) ? '' : 'none';
    });
    svg.querySelectorAll('.module-group').forEach(el => {
        el.style.display = layerVisibility.groups ? '' : 'none';
    });

    // Update trails
    if (layerVisibility.trails) {
        drawTrails();
    } else {
        const trailGroup = svg.querySelector('#trail-layer');
        if (trailGroup) trailGroup.innerHTML = '';
    }

    if (layerVisibility.ratsnest) {
        updateRatsnest();
    }

    if (layerVisibility.forces) {
        updateForces();
    }

    updateModuleGroups();

    // Apply module visibility
    updateModuleVisibility();
}

function updateModuleVisibility() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Update module visibility from checkboxes
    // Template uses . and space -> - normalization
    const checkboxes = document.querySelectorAll('[id^="show-module-"]');
    checkboxes.forEach(checkbox => {
        // Extract module name, reversing the . and space -> - normalization
        const moduleName = checkbox.id.replace('show-module-', '');
        moduleVisibility[moduleName] = checkbox.checked;
    });

    // Apply visibility to elements
    for (const [moduleName, visible] of Object.entries(moduleVisibility)) {
        // Module classes in SVG use a safe normalization
        const safeModuleName = sanitizeModuleName(moduleName);
        svg.querySelectorAll(`.module-${safeModuleName}`).forEach(el => {
            if (!visible) {
                el.style.display = 'none';
                return;
            }

            if (el.classList.contains('pad-element')) {
                const isTop = el.classList.contains('comp-top');
                const isBottom = el.classList.contains('comp-bottom');
                const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);
                el.style.display = (layerVisibility.pads && layerVisible) ? '' : 'none';
                return;
            }
            if (el.classList.contains('ref-label')) {
                const isTop = el.classList.contains('comp-top');
                const isBottom = el.classList.contains('comp-bottom');
                const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);
                el.style.display = (layerVisibility.labels && layerVisible) ? '' : 'none';
                return;
            }
            if (el.classList.contains('module-group')) {
                el.style.display = layerVisibility.groups ? '' : 'none';
                return;
            }

            // Check if parent layer is visible
            const isTop = el.classList.contains('comp-top');
            const isBottom = el.classList.contains('comp-bottom');
            const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);

            el.style.display = layerVisible ? '' : 'none';
        });
    }
}

function initializeModuleVisibility() {
    const checkboxes = document.querySelectorAll('[id^="show-module-"]');
    checkboxes.forEach(checkbox => {
        const moduleName = checkbox.id.replace('show-module-', '');
        moduleVisibility[moduleName] = checkbox.checked;
    });
}

// ============================================================================
// Component Selection Highlight
// ============================================================================
function updateComponentSelection() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Remove previous selection
    svg.querySelectorAll('.component.selected').forEach(el => {
        el.classList.remove('selected');
    });

    // Add selection to current component
    if (selectedComponent) {
        svg.querySelectorAll(`.component[data-ref="${selectedComponent}"]`).forEach(el => {
            el.classList.add('selected');
        });
    }
}

// ============================================================================
// Delta Frame Playback
// ============================================================================
function showFrame(frameIndex) {
    if (frameIndex < 0 || frameIndex >= totalFrames) return;

    currentFrame = frameIndex;

    // Debug logging
    if (frameIndex < 3 || frameIndex % 20 === 0) {
        console.log(`showFrame(${frameIndex}) called`);
    }

    // Reconstruct full state by replaying all deltas from 0 to frameIndex
    reconstructState(frameIndex);

    // Update SVG DOM elements
    updateSVGElements();

    // Update info bar
    updateInfoBar();

    // Update trails if visible
    if (layerVisibility.trails) {
        drawTrails();
    }

    // Update energy graph
    if (typeof drawEnergyGraph === 'function') {
        drawEnergyGraph();
    }
}

function reconstructState(toFrame) {
    // Optimization: if moving forward sequentially, only apply new deltas
    if (toFrame === lastReconstructedFrame + 1) {
        applyDelta(deltaFrames[toFrame]);
        lastReconstructedFrame = toFrame;
        if (toFrame < 3) {
            console.log(`  Sequential update: applied delta ${toFrame}, state has ${Object.keys(currentState.components).length} components`);
        }
        return;
    }

    // If seeking or moving backward, replay from start
    // Reset to empty state
    currentState.components = {};
    currentState.modules = {};
    currentState.overlaps = [];
    currentState.forces = [];
    currentState.energy = 0;
    currentState.max_move = 0;
    currentState.overlap_count = 0;
    currentState.wire_length = 0;

    // Replay all deltas from 0 to toFrame
    for (let i = 0; i <= toFrame; i++) {
        applyDelta(deltaFrames[i]);
    }

    lastReconstructedFrame = toFrame;

    if (toFrame < 3) {
        console.log(`  Full reconstruction to frame ${toFrame}, state has ${Object.keys(currentState.components).length} components`);
        // Log first component's position
        const firstRef = Object.keys(currentState.components)[0];
        if (firstRef) {
            console.log(`    ${firstRef}: ${currentState.components[firstRef]}`);
        }
    }
}

function applyDelta(delta) {
    // Update component positions/rotations
    // Merge deltas: null values mean "no change", keep previous value
    if (delta.changed_components) {
        for (const [ref, comp] of Object.entries(delta.changed_components)) {
            const prev = currentState.components[ref] || [0, 0, 0];
            const [x, y, rotation] = comp;
            currentState.components[ref] = [
                x !== null && x !== undefined ? x : prev[0],
                y !== null && y !== undefined ? y : prev[1],
                rotation !== null && rotation !== undefined ? rotation : prev[2]
            ];
        }
    }

    // Update modules
    if (delta.changed_modules) {
        for (const [ref, moduleName] of Object.entries(delta.changed_modules)) {
            currentState.modules[ref] = moduleName;
        }
    }

    // Update overlaps
    if (delta.overlaps !== undefined) {
        currentState.overlaps = delta.overlaps;
    }

    // Update forces
    if (delta.forces !== undefined) {
        currentState.forces = delta.forces;
    }

    // Update metrics
    if (delta.energy !== undefined) currentState.energy = delta.energy;
    if (delta.max_move !== undefined) currentState.max_move = delta.max_move;
    if (delta.overlap_count !== undefined) currentState.overlap_count = delta.overlap_count;
    if (delta.wire_length !== undefined) currentState.wire_length = delta.wire_length;
}

function updateSVGElements() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) {
        console.error('SVG not found in frame container');
        return;
    }

    // Update component transforms using proper coordinate transformation
    let updatedCount = 0;
    for (const [ref, comp] of Object.entries(currentState.components)) {
        const [x, y, rotation] = comp;
        const group = svg.querySelector(`.component[data-ref="${ref}"]`);
        if (group) {
            // Transform board coordinates to SVG coordinates
            const cx = tx(x);
            const cy = ty(y);

            // Get component dimensions from static props
            const props = staticProps[ref];
            if (props) {
                const hw = ts(props.width / 2);
                const hh = ts(props.height / 2);

                // Update the rect position inside the group
                const rect = group.querySelector('rect');
                if (rect) {
                    rect.setAttribute('x', cx - hw);
                    rect.setAttribute('y', cy - hh);
                    rect.setAttribute('width', ts(props.width));
                    rect.setAttribute('height', ts(props.height));
                }
            }

            // Apply rotation transform around the center point
            // SVG uses negative rotation for CCW (to match board coordinates)
            group.setAttribute('transform', `rotate(${-rotation} ${cx} ${cy})`);
            updatedCount++;
        } else {
            if (currentFrame < 3) {
                console.warn(`Component not found: ${ref}`);
            }
        }
    }

    // Debug logging (only on first few frames)
    if (currentFrame < 3) {
        console.log(`Frame ${currentFrame}: Updated ${updatedCount} components out of ${Object.keys(currentState.components).length}`);
    }

    // Update module classes and colors
    for (const [ref, moduleName] of Object.entries(currentState.modules)) {
        const group = svg.querySelector(`.component[data-ref="${ref}"]`);
        const label = svg.querySelector(`.ref-label[data-ref="${ref}"]`);
        const pads = svg.querySelectorAll(`.pad-element[data-ref="${ref}"]`);

        applyModuleClass(group, moduleName);
        applyModuleClass(label, moduleName);
        pads.forEach(pad => applyModuleClass(pad, moduleName));

        if (group && moduleName) {
            const color = moduleColors[moduleName];
            if (color) {
                const rect = group.querySelector('rect');
                if (rect) {
                    rect.setAttribute('fill', color);
                }
            }
        }
    }

    // Update overlap highlighting
    updateOverlapHighlighting();

    // Update module groups
    updateModuleGroups();

    // Update ratsnest if visible
    if (layerVisibility.ratsnest) {
        updateRatsnest();
    }

    // Update forces if visible
    if (layerVisibility.forces) {
        updateForces();
    }

    // Update pads and labels
    updatePads();
    updateLabels();
}

function updateOverlapHighlighting() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Clear all overlap classes
    svg.querySelectorAll('.component').forEach(el => {
        el.classList.remove('overlapping');
    });

    // Add overlap class to overlapping components
    for (const [ref1, ref2] of currentState.overlaps) {
        const comp1 = svg.querySelector(`.component[data-ref="${ref1}"]`);
        const comp2 = svg.querySelector(`.component[data-ref="${ref2}"]`);
        if (comp1) comp1.classList.add('overlapping');
        if (comp2) comp2.classList.add('overlapping');
    }
}

function isModuleVisible(moduleName) {
    const safeName = sanitizeModuleName(moduleName);
    const legacyName = moduleName.replace(/[.\s]/g, '-');
    if (Object.prototype.hasOwnProperty.call(moduleVisibility, safeName)) {
        return moduleVisibility[safeName];
    }
    if (Object.prototype.hasOwnProperty.call(moduleVisibility, legacyName)) {
        return moduleVisibility[legacyName];
    }
    if (Object.prototype.hasOwnProperty.call(moduleVisibility, moduleName)) {
        return moduleVisibility[moduleName];
    }
    return true;
}

function isComponentLayerVisible(ref) {
    const layer = componentLayers[ref] || 'top';
    if (layer === 'top') return layerVisibility.top;
    if (layer === 'bottom') return layerVisibility.bottom;
    return true;
}

function updateBoundsWithRect(bounds, cx, cy, width, height) {
    const hw = width / 2;
    const hh = height / 2;
    bounds.minX = Math.min(bounds.minX, cx - hw);
    bounds.minY = Math.min(bounds.minY, cy - hh);
    bounds.maxX = Math.max(bounds.maxX, cx + hw);
    bounds.maxY = Math.max(bounds.maxY, cy + hh);
}

function updateBoundsWithRotatedRect(bounds, cx, cy, width, height, rotation) {
    const hw = width / 2;
    const hh = height / 2;
    const rad = (rotation || 0) * Math.PI / 180;
    const cosR = Math.cos(rad);
    const sinR = Math.sin(rad);
    const corners = [
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ];

    for (const [dx, dy] of corners) {
        const rx = cx + (dx * cosR - dy * sinR);
        const ry = cy + (dx * sinR + dy * cosR);
        bounds.minX = Math.min(bounds.minX, rx);
        bounds.minY = Math.min(bounds.minY, ry);
        bounds.maxX = Math.max(bounds.maxX, rx);
        bounds.maxY = Math.max(bounds.maxY, ry);
    }
}

function getNetStyle(netName) {
    const lower = (netName || '').toLowerCase();
    if (lower.includes('gnd') || lower.includes('vss')) {
        return { color: '#2ecc71', opacity: 0.45, width: 0.6 };
    }
    if (lower.includes('vcc') || lower.includes('vdd') || lower.includes('pwr')) {
        return { color: '#e74c3c', opacity: 0.45, width: 0.6 };
    }
    return { color: '#4a6fa5', opacity: 0.35, width: 0.5 };
}

function computeMSTEdges(points) {
    const count = points.length;
    if (count < 2) return [];

    const visited = new Array(count).fill(false);
    visited[0] = true;
    const edges = [];

    for (let step = 1; step < count; step++) {
        let bestDist = Infinity;
        let bestFrom = -1;
        let bestTo = -1;

        for (let i = 0; i < count; i++) {
            if (!visited[i]) continue;
            const xi = points[i].x;
            const yi = points[i].y;
            for (let j = 0; j < count; j++) {
                if (visited[j]) continue;
                const dx = xi - points[j].x;
                const dy = yi - points[j].y;
                const dist = dx * dx + dy * dy;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestFrom = i;
                    bestTo = j;
                }
            }
        }

        if (bestTo === -1) break;
        visited[bestTo] = true;
        edges.push([bestFrom, bestTo]);
    }

    return edges;
}

function updateModuleGroups() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Remove any static module groups from the initial SVG
    svg.querySelectorAll('.module-group').forEach(el => el.remove());

    let moduleGroupsLayer = svg.querySelector('#module-groups-layer');
    if (!moduleGroupsLayer) {
        // Create the layer if it doesn't exist
        moduleGroupsLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        moduleGroupsLayer.id = 'module-groups-layer';
        // Insert before components so it sits above background but below parts
        const firstComponent = svg.querySelector('.component');
        if (firstComponent) {
            svg.insertBefore(moduleGroupsLayer, firstComponent);
        } else {
            svg.appendChild(moduleGroupsLayer);
        }
    }

    // Clear existing groups
    moduleGroupsLayer.innerHTML = '';

    if (!layerVisibility.groups) return;

    // Group components by module
    const moduleGroups = {};
    for (const [ref, moduleName] of Object.entries(currentState.modules)) {
        if (!moduleName || moduleName === 'unassigned') continue;
        if (!isModuleVisible(moduleName)) continue;
        if (!isComponentLayerVisible(ref)) continue;
        if (!moduleGroups[moduleName]) moduleGroups[moduleName] = [];
        moduleGroups[moduleName].push(ref);
    }

    // Draw bounding box for each module
    for (const [moduleName, refs] of Object.entries(moduleGroups)) {
        if (refs.length < 2) continue;

        // Calculate bounding box in board coordinates, then transform
        const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
        let includedCount = 0;
        for (const ref of refs) {
            const comp = currentState.components[ref];
            if (!comp) continue;
            const [x, y] = comp;
            const props = staticProps[ref];
            if (!props) continue;

            const rotation = comp[2] || 0;
            includedCount += 1;
            updateBoundsWithRotatedRect(bounds, x, y, props.width, props.height, rotation);

            if (props.pads && props.pads.length) {
                const radians = rotation * Math.PI / 180;
                const cosR = Math.cos(radians);
                const sinR = Math.sin(radians);
                for (const [padX, padY, padW, padH] of props.pads) {
                    const px = x + (padX * cosR - padY * sinR);
                    const py = y + (padX * sinR + padY * cosR);
                    updateBoundsWithRect(bounds, px, py, padW, padH);
                }
            }
        }

        if (includedCount < 2 || !isFinite(bounds.minX)) continue;

        const padding = 1.5;
        const color = moduleColors[moduleName] || '#3498db';
        const safeModuleName = sanitizeModuleName(moduleName);

        const minX = bounds.minX - padding;
        const minY = bounds.minY - padding;
        const maxX = bounds.maxX + padding;
        const maxY = bounds.maxY + padding;

        // Transform to SVG coordinates
        const svgMinX = tx(minX);
        const svgMinY = ty(minY);
        const svgMaxX = tx(maxX);
        const svgMaxY = ty(maxY);

        // Create group rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', svgMinX);
        rect.setAttribute('y', svgMinY);
        rect.setAttribute('width', svgMaxX - svgMinX);
        rect.setAttribute('height', svgMaxY - svgMinY);
        rect.setAttribute('fill', color);
        rect.setAttribute('fill-opacity', '0.15');
        rect.setAttribute('stroke', color);
        rect.setAttribute('stroke-width', '1.5');
        rect.setAttribute('stroke-dasharray', '4 2');
        rect.classList.add('module-group', `module-${safeModuleName}`);
        moduleGroupsLayer.appendChild(rect);

        // Add label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        const labelX = (svgMinX + svgMaxX) / 2;
        const labelY = svgMinY - 5;
        text.setAttribute('x', labelX);
        text.setAttribute('y', labelY);
        text.setAttribute('fill', color);
        text.setAttribute('font-size', '10');
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('text-anchor', 'middle');
        text.textContent = moduleName;
        text.classList.add('module-group', `module-${safeModuleName}`);
        moduleGroupsLayer.appendChild(text);
    }
}

function updatePads() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Update pad positions based on component positions
    for (const [ref, comp] of Object.entries(currentState.components)) {
        const [x, y, rotation] = comp;
        const props = staticProps[ref];
        if (!props || !props.pads) continue;

        const radians = rotation * Math.PI / 180;
        const cosR = Math.cos(radians);
        const sinR = Math.sin(radians);

        props.pads.forEach((pad, index) => {
            const [pxRel, pyRel, pw, ph, net] = pad;

            // Apply rotation and translation (in board coordinates)
            const px = x + (pxRel * cosR - pyRel * sinR);
            const py = y + (pxRel * sinR + pyRel * cosR);

            // Transform to SVG coordinates
            const pcx = tx(px);
            const pcy = ty(py);
            const phw = ts(pw / 2);
            const phh = ts(ph / 2);

            let padEl = svg.querySelector(`.pad-element[data-ref="${ref}"][data-pad-index="${index}"]`);
            if (!padEl) {
                padEl = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                padEl.classList.add('pad-element');
                padEl.setAttribute('data-ref', ref);
                padEl.setAttribute('data-pad-index', index);
                padEl.setAttribute('fill', getPadColor(net || ''));
                padEl.setAttribute('stroke', '#ffffff');
                padEl.setAttribute('stroke-width', '0.5');
                const layerClass = componentLayers[ref] === 'bottom' ? 'comp-bottom' : 'comp-top';
                padEl.classList.add(layerClass);
                svg.appendChild(padEl);
            }

            const isTop = padEl.classList.contains('comp-top');
            const isBottom = padEl.classList.contains('comp-bottom');
            const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);
            padEl.style.display = (layerVisibility.pads && layerVisible) ? '' : 'none';
            padEl.setAttribute('x', pcx - phw);
            padEl.setAttribute('y', pcy - phh);
            padEl.setAttribute('width', ts(pw));
            padEl.setAttribute('height', ts(ph));

            const moduleName = currentState.modules[ref] || 'default';
            applyModuleClass(padEl, moduleName);
        });
    }
}

function updateLabels() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Update label positions based on component positions
    for (const [ref, comp] of Object.entries(currentState.components)) {
        const [x, y, rotation] = comp;
        const props = staticProps[ref];
        if (!props) continue;

        const cx = tx(x);
        const cy = ty(y);
        const hw = ts(props.width / 2);
        const hh = ts(props.height / 2);
        const fontSize = Math.max(7, Math.min(11, ts(Math.min(props.width, props.height) * 0.35)));
        const labelPadding = 3;
        const labelWidth = ref.length * fontSize * 0.6;
        const labelHeight = fontSize;
        const compIsSmall = ts(props.width) < 25 || ts(props.height) < 25;

        // Find label element for this component
        const label = svg.querySelector(`.ref-label[data-ref="${ref}"]`);
        if (label) {
            let labelX = cx;
            let labelY = cy;
            if (props.width >= props.height || compIsSmall) {
                labelX = cx;
                labelY = cy - hh - labelPadding - labelHeight / 2;
            } else {
                labelX = cx + hw + labelPadding + labelWidth / 2;
                labelY = cy;
            }

            label.setAttribute('x', labelX);
            label.setAttribute('y', labelY);
            label.setAttribute('font-size', fontSize);
            label.setAttribute('text-anchor', 'middle');
            const isTop = label.classList.contains('comp-top');
            const isBottom = label.classList.contains('comp-bottom');
            const layerVisible = (isTop && layerVisibility.top) || (isBottom && layerVisibility.bottom) || (!isTop && !isBottom);
            label.style.display = (layerVisibility.labels && layerVisible) ? '' : 'none';
        }
    }
}

function updateRatsnest() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    // Remove existing ratsnest lines (including those from initial SVG)
    svg.querySelectorAll('.ratsnest').forEach(el => el.remove());

    let ratsnestLayer = svg.querySelector('#ratsnest-layer');
    if (!ratsnestLayer) {
        ratsnestLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        ratsnestLayer.id = 'ratsnest-layer';
        ratsnestLayer.classList.add('ratsnest-layer');
        const firstComponent = svg.querySelector('.component');
        if (firstComponent) {
            svg.insertBefore(ratsnestLayer, firstComponent);
        } else {
            svg.appendChild(ratsnestLayer);
        }
    } else {
        ratsnestLayer.innerHTML = '';
    }

    if (!layerVisibility.ratsnest) return;

    const netPads = {};
    const hasNetlist = typeof netlist !== 'undefined' && netlist && Object.keys(netlist).length > 0;

    if (hasNetlist) {
        for (const [netName, padRefs] of Object.entries(netlist)) {
            const pads = [];
            for (const [ref, padIndex] of padRefs) {
                if (!isComponentLayerVisible(ref)) continue;
                const comp = currentState.components[ref];
                const props = staticProps[ref];
                if (!comp || !props || !props.pads || !props.pads[padIndex]) continue;

                const [x, y, rotation] = comp;
                const [padX, padY] = props.pads[padIndex];
                const radians = (rotation || 0) * Math.PI / 180;
                const cosR = Math.cos(radians);
                const sinR = Math.sin(radians);
                const px = padX * cosR - padY * sinR + x;
                const py = padX * sinR + padY * cosR + y;
                pads.push({ x: px, y: py });
            }
            if (pads.length) netPads[netName] = pads;
        }
    } else {
        for (const [ref, comp] of Object.entries(currentState.components)) {
            if (!isComponentLayerVisible(ref)) continue;
            const [x, y, rotation] = comp;
            const props = staticProps[ref];
            if (!props || !props.pads) continue;

            const radians = (rotation || 0) * Math.PI / 180;
            const cosR = Math.cos(radians);
            const sinR = Math.sin(radians);

            for (const [padX, padY, padW, padH, netName] of props.pads) {
                if (!netName || !netName.trim()) continue;
                const px = padX * cosR - padY * sinR + x;
                const py = padX * sinR + padY * cosR + y;
                if (!netPads[netName]) netPads[netName] = [];
                netPads[netName].push({ x: px, y: py });
            }
        }
    }

    for (const [netName, pads] of Object.entries(netPads)) {
        if (pads.length < 2) continue;

        const style = getNetStyle(netName);
        const edges = computeMSTEdges(pads);
        for (const [i, j] of edges) {
            const p1 = pads[i];
            const p2 = pads[j];
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', tx(p1.x));
            line.setAttribute('y1', ty(p1.y));
            line.setAttribute('x2', tx(p2.x));
            line.setAttribute('y2', ty(p2.y));
            line.setAttribute('stroke', style.color);
            line.setAttribute('stroke-width', style.width);
            line.setAttribute('opacity', style.opacity);
            line.classList.add('ratsnest');
            ratsnestLayer.appendChild(line);
        }
    }
}

function updateForces() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    svg.querySelectorAll('.force-vector').forEach(el => el.remove());

    let forceLayer = svg.querySelector('#force-layer');
    if (!forceLayer) {
        forceLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        forceLayer.id = 'force-layer';
        forceLayer.classList.add('force-layer');
        const firstComponent = svg.querySelector('.component');
        if (firstComponent) {
            svg.insertBefore(forceLayer, firstComponent);
        } else {
            svg.appendChild(forceLayer);
        }
    } else {
        forceLayer.innerHTML = '';
    }

    if (!layerVisibility.forces) return;

    const forceColors = {
        repulsion: '#e74c3c',
        attraction: '#2ecc71',
        boundary: '#3498db',
        constraint: '#f39c12',
        alignment: '#9b59b6'
    };

    const maxArrowLength = 25;
    for (const [ref, forceList] of Object.entries(currentState.forces || {})) {
        const comp = currentState.components[ref];
        if (!comp) continue;
        const [x, y] = comp;
        const cx = tx(x);
        const cy = ty(y);

        for (const [fx, fy, forceType] of forceList) {
            if (Math.abs(fx) < 0.1 && Math.abs(fy) < 0.1) continue;

            const rawLength = Math.sqrt(fx * fx + fy * fy);
            if (rawLength < 0.1) continue;

            const arrowLength = Math.min(5 + Math.log1p(rawLength) * 5, maxArrowLength);
            const nx = fx / rawLength;
            const ny = fy / rawLength;
            const endX = cx + nx * arrowLength;
            const endY = cy + ny * arrowLength;

            const color = forceColors[forceType] || '#e74c3c';

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', cx);
            line.setAttribute('y1', cy);
            line.setAttribute('x2', endX);
            line.setAttribute('y2', endY);
            line.setAttribute('stroke', color);
            line.setAttribute('stroke-width', '2');
            line.classList.add('force-vector');
            forceLayer.appendChild(line);

            const headSize = 4;
            const px = -ny;
            const py = nx;
            const base1X = endX - nx * headSize + px * headSize * 0.5;
            const base1Y = endY - ny * headSize + py * headSize * 0.5;
            const base2X = endX - nx * headSize - px * headSize * 0.5;
            const base2Y = endY - ny * headSize - py * headSize * 0.5;

            const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            arrow.setAttribute('points', `${endX},${endY} ${base1X},${base1Y} ${base2X},${base2Y}`);
            arrow.setAttribute('fill', color);
            arrow.classList.add('force-vector');
            forceLayer.appendChild(arrow);
        }
    }
}

// ============================================================================
// Trail Rendering
// ============================================================================
function drawTrails() {
    const svg = frameContainer.querySelector('svg');
    if (!svg) return;

    let trailGroup = svg.querySelector('#trail-layer');
    if (!trailGroup) {
        trailGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        trailGroup.id = 'trail-layer';
        // Insert before components layer
        const firstG = svg.querySelector('g');
        if (firstG) {
            svg.insertBefore(trailGroup, firstG);
        } else {
            svg.appendChild(trailGroup);
        }
    }

    trailGroup.innerHTML = '';

    if (currentFrame < 1) return;

    // Collect position history for all components
    for (const ref of Object.keys(currentState.components)) {
        // If component is selected, only show its trail
        if (selectedComponent && ref !== selectedComponent) continue;

        const history = [];
        for (let i = 0; i <= currentFrame; i++) {
            // Reconstruct state at frame i
            const comp = getComponentAtFrame(ref, i);
            if (comp && comp[0] !== null && comp[1] !== null) {
                history.push({ x: comp[0], y: comp[1], frame: i });
            }
        }

        if (history.length < 2) continue;

        // Check if moved significantly (in board coords)
        const start = history[0];
        const end = history[history.length - 1];
        const dist = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
        if (dist < 0.5) continue;  // 0.5mm threshold

        const isSelected = (ref === selectedComponent);
        const trailColor = isSelected ? '#00d4ff' : '#ff6b6b';
        const trailOpacityMult = isSelected ? 1.2 : 1.0;

        // Draw connecting lines
        if (history.length > 1) {
            let pathD = `M ${tx(history[0].x)} ${ty(history[0].y)}`;
            for (let i = 1; i < history.length; i++) {
                pathD += ` L ${tx(history[i].x)} ${ty(history[i].y)}`;
            }
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', pathD);
            path.setAttribute('stroke', trailColor);
            path.setAttribute('stroke-width', isSelected ? '2' : '1');
            path.setAttribute('stroke-opacity', isSelected ? '0.7' : '0.4');
            path.setAttribute('fill', 'none');
            trailGroup.appendChild(path);
        }

        // Draw breadcrumb dots
        for (let i = 0; i < history.length; i++) {
            const pos = history[i];
            const age = (currentFrame - pos.frame) / Math.max(currentFrame, 1);
            const baseOpacity = 0.2 + (1 - age) * 0.6;
            const opacity = Math.min(1, baseOpacity * trailOpacityMult);
            const radius = isSelected ? 2 + (1 - age) * 2 : 1.5 + (1 - age) * 1.5;

            const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot.setAttribute('cx', tx(pos.x));
            dot.setAttribute('cy', ty(pos.y));
            dot.setAttribute('r', radius);
            dot.setAttribute('fill', trailColor);
            dot.setAttribute('opacity', opacity);
            trailGroup.appendChild(dot);
        }

        // Start marker
        const startColor = isSelected ? '#2ecc71' : trailColor;
        const startDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        startDot.setAttribute('cx', tx(start.x));
        startDot.setAttribute('cy', ty(start.y));
        startDot.setAttribute('r', isSelected ? '5' : '3');
        startDot.setAttribute('fill', isSelected ? startColor : 'none');
        startDot.setAttribute('stroke', startColor);
        startDot.setAttribute('stroke-width', isSelected ? '2' : '1.5');
        startDot.setAttribute('opacity', isSelected ? '1' : '0.7');
        trailGroup.appendChild(startDot);

        // End marker for selected
        if (isSelected) {
            const endColor = '#00d4ff';
            const endDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            endDot.setAttribute('cx', tx(end.x));
            endDot.setAttribute('cy', ty(end.y));
            endDot.setAttribute('r', '6');
            endDot.setAttribute('fill', endColor);
            endDot.setAttribute('stroke', '#ffffff');
            endDot.setAttribute('stroke-width', '2');
            trailGroup.appendChild(endDot);

            // Direction indicator
            if (history.length >= 2) {
                const prev = history[history.length - 2];
                const dx = tx(end.x) - tx(prev.x);
                const dy = ty(end.y) - ty(prev.y);
                const len = Math.sqrt(dx * dx + dy * dy);
                if (len > 0.1) {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    const endSvgX = tx(end.x);
                    const endSvgY = ty(end.y);
                    line.setAttribute('x1', endSvgX - dx/len * 3);
                    line.setAttribute('y1', endSvgY - dy/len * 3);
                    line.setAttribute('x2', endSvgX);
                    line.setAttribute('y2', endSvgY);
                    line.setAttribute('stroke', endColor);
                    line.setAttribute('stroke-width', '3');
                    line.setAttribute('opacity', '0.8');
                    trailGroup.appendChild(line);
                }
            }
        }
    }
}

function getComponentAtFrame(ref, frameIndex) {
    // Reconstruct component position at specific frame by replaying deltas
    // Properly merge delta fields to handle partial updates (e.g., rotation-only)
    let x = null, y = null, rotation = null;

    for (let i = 0; i <= frameIndex; i++) {
        const delta = deltaFrames[i];
        if (delta.changed_components && delta.changed_components[ref]) {
            const comp = delta.changed_components[ref];
            // Merge: only update non-null values
            if (comp[0] !== null && comp[0] !== undefined) x = comp[0];
            if (comp[1] !== null && comp[1] !== undefined) y = comp[1];
            if (comp[2] !== null && comp[2] !== undefined) rotation = comp[2];
        }
    }

    if (x === null || y === null) return null;
    return [x, y, rotation || 0];
}

// ============================================================================
// Info Bar - Matches template IDs
// ============================================================================
function updateInfoBar() {
    const frameNum = document.getElementById('frame-num');
    const phaseEl = document.getElementById('phase');
    const iterationEl = document.getElementById('iteration');
    const energyEl = document.getElementById('energy');
    const maxMoveEl = document.getElementById('max-move');
    const overlapsEl = document.getElementById('overlaps');
    const wireLengthEl = document.getElementById('wire-length');

    const delta = deltaFrames[currentFrame] || {};

    if (frameNum) frameNum.textContent = currentFrame;
    if (phaseEl) phaseEl.textContent = delta.phase || '-';
    if (iterationEl) iterationEl.textContent = delta.iteration || 0;
    if (energyEl) energyEl.textContent = (currentState.energy || delta.energy || 0).toFixed(2);
    if (maxMoveEl) maxMoveEl.textContent = (currentState.max_move || 0).toFixed(4);
    if (overlapsEl) overlapsEl.textContent = currentState.overlap_count || 0;
    if (wireLengthEl) wireLengthEl.textContent = (currentState.wire_length || 0).toFixed(1) + 'mm';
}

// ============================================================================
// Energy Graph
// ============================================================================
function drawEnergyGraph() {
    const canvas = document.getElementById('energy-canvas');
    if (!canvas || typeof deltaFrames === 'undefined') return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 50;

    if (deltaFrames.length === 0) return;

    // Extract energies and wire lengths from deltas
    const energies = [];
    const wireLengths = [];
    let energy = 0;
    let wireLength = 0;
    for (const delta of deltaFrames) {
        if (delta.energy !== undefined) energy = delta.energy;
        if (delta.wire_length !== undefined) wireLength = delta.wire_length;
        energies.push(energy);
        wireLengths.push(wireLength);
    }

    const maxEnergy = Math.max(...energies) || 1;
    const maxWireLength = Math.max(...wireLengths) || 1;

    // Draw graph
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < energies.length; i++) {
        const x = (i / (energies.length - 1)) * width;
        const y = height - (energies[i] / maxEnergy) * (height - 4) - 2;
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Draw wire length line (green)
    ctx.strokeStyle = '#2ecc71';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < wireLengths.length; i++) {
        const x = (i / (wireLengths.length - 1)) * width;
        const y = height - (wireLengths[i] / maxWireLength) * (height - 4) - 2;
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Draw current frame indicator
    const currentX = (currentFrame / (energies.length - 1)) * width;
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, height);
    ctx.stroke();

    // Update range label
    const rangeLabel = document.getElementById('graph-range');
    if (rangeLabel) {
        rangeLabel.textContent = `0-${energies.length - 1}`;
    }
}

// ============================================================================
// Energy Graph Interaction Handlers (called by template)
// ============================================================================
function seekToFrame(event) {
    const canvas = document.getElementById('energy-canvas');
    if (!canvas || totalFrames <= 1) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const frameIndex = Math.round((x / rect.width) * (totalFrames - 1));

    currentFrame = Math.max(0, Math.min(frameIndex, totalFrames - 1));
    showFrame(currentFrame);
}

function startSeeking(event) {
    isSeeking = true;
    const graph = document.querySelector('.energy-graph');
    if (graph) graph.classList.add('seeking');
    seekToFrame(event);
}

function continueSeeking(event) {
    if (isSeeking) {
        seekToFrame(event);
    }
}

function stopSeeking() {
    isSeeking = false;
    const graph = document.querySelector('.energy-graph');
    if (graph) graph.classList.remove('seeking');
}

// ============================================================================
// Initialization
// ============================================================================
function initialize() {
    // Initialize module visibility
    initializeModuleVisibility();

    // Draw energy graph
    if (typeof drawEnergyGraph === 'function' && typeof deltaFrames !== 'undefined') {
        drawEnergyGraph();

        // Redraw on window resize
        window.addEventListener('resize', () => {
            drawEnergyGraph();
        });
    }

    // Show first frame (delay slightly to ensure DOM is fully rendered)
    if (typeof deltaFrames !== 'undefined' && deltaFrames.length > 0) {
        // totalFrames already set inline
        setTimeout(() => {
            const svg = frameContainer.querySelector('svg');
            if (!svg) {
                console.error('SVG not found in frame container during initialization');
                console.log('Frame container:', frameContainer);
                console.log('SVG container:', document.getElementById('svg-container'));
                return;
            }
            console.log('SVG found, showing frame 0');
            showFrame(0);
            updateLayers();
        }, 100);
    }
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}
