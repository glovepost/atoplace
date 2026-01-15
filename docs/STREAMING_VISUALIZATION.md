# Real-Time Streaming Visualization

WebSocket-based real-time visualization system for watching placement optimization as it runs.

## Overview

The streaming visualization system enables you to watch placement optimization in real-time through a web browser. Unlike the post-process Canvas visualization, streaming provides:

- **Live Updates**: See components move as the algorithm runs
- **Interactive Controls**: Pause, resume, or stop optimization from the browser
- **Real-Time Metrics**: Monitor energy, wire length, and convergence live
- **Multiple Viewers**: Connect multiple browsers to the same optimization session
- **Low Latency**: Configurable frame rate (default 10 FPS) with < 100ms latency

## Quick Start

### Basic Usage

```python
import asyncio
from atoplace.board.kicad_adapter import Board
from atoplace.placement.streaming_visualizer import StreamingVisualizer

async def optimize_with_streaming():
    # Load board
    board = Board.from_kicad("my_board.kicad_pcb")

    # Create streaming visualizer
    viz = StreamingVisualizer(
        board,
        host='localhost',
        port=8765,
        max_fps=10.0
    )

    try:
        # Start streaming server and generate viewer HTML
        await viz.start_streaming(generate_viewer=True)
        print("Open placement_debug/stream_viewer.html in your browser!")

        # Your placement algorithm here
        for iteration in range(100):
            # ... placement step ...

            # Capture and stream frame
            await viz.capture_and_stream(
                label=f"Iteration {iteration}",
                iteration=iteration,
                phase="refinement",
                modules=modules,
                forces=forces,
                energy=energy,
                max_move=max_move
            )

            # Check for user interaction
            if await viz.is_paused():
                await viz.wait_resume()

            if await viz.is_stop_requested():
                break

        # Export final visualization
        viz.export_canvas_html("final_placement.html")

    finally:
        await viz.stop_streaming()

# Run
asyncio.run(optimize_with_streaming())
```

### Testing the System

Run the provided test scripts to verify streaming works:

```bash
# Test with mock data (no KiCad required)
python test_streaming_mock.py

# Test with real board (requires KiCad Python)
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 test_streaming.py examples/tutorial/tutorial_with_components.kicad_pcb
```

Then open `placement_debug/stream_viewer.html` in your browser to watch the optimization in real-time.

## Architecture

### Components

#### 1. StreamServer (`stream_server.py`)

WebSocket server managing client connections and broadcasting frames.

**Key Features:**
- WebSocket server on configurable host/port
- Frame rate limiting (default 10 FPS)
- Interactive control handling (pause/resume/stop)
- Automatic reconnection support
- Multiple concurrent viewers
- Statistics tracking (clients, frames sent, data usage)

**API:**
```python
server = StreamServer(host='localhost', port=8765, max_fps=10.0)
await server.start()
await server.broadcast_frame(frame_data)
await server.broadcast_status("info", "Starting optimization")
server.is_paused()  # Check if user paused
server.is_stop_requested()  # Check if user requested stop
await server.stop()
```

#### 2. Stream Viewer (`stream_viewer.py`)

Generates HTML viewer with WebSocket client and Canvas rendering.

**Key Features:**
- WebSocket client with auto-reconnect
- Canvas-based real-time rendering
- Interactive controls (pause/resume/stop buttons)
- Performance monitoring (FPS, latency, data usage)
- Event log showing server messages
- Responsive layout with sidebar

**Generated HTML includes:**
- `StreamClient` class - WebSocket communication
- `PlacementCanvasRenderer` class - Canvas rendering
- Interactive UI with controls
- Real-time metrics display

#### 3. StreamingVisualizer (`streaming_visualizer.py`)

Async wrapper combining PlacementVisualizer with WebSocket streaming.

**Key Features:**
- Wraps existing PlacementVisualizer (backward compatible)
- Async API for streaming frames
- Interactive control checking
- Server lifecycle management
- Statistics and viewer URL access
- Final HTML export support

**API:**
```python
viz = StreamingVisualizer(board, host='localhost', port=8765, max_fps=10.0)
await viz.start_streaming(generate_viewer=True)
await viz.capture_and_stream(label, iteration, phase, modules, forces, energy, max_move)
await viz.is_paused()
await viz.is_stop_requested()
await viz.wait_resume()
await viz.send_status(status, message)
viz.export_canvas_html("final.html")
await viz.stop_streaming()
```

## Configuration

### Server Settings

```python
viz = StreamingVisualizer(
    board,
    host='localhost',        # Server host (default: localhost)
    port=8765,               # WebSocket port (default: 8765)
    max_fps=10.0,            # Maximum frame rate (default: 10 FPS)
    grid_spacing=1.27        # Grid spacing for visualization
)
```

### Frame Rate Tuning

Higher FPS provides smoother visualization but increases network usage:

| FPS | Use Case | Network Usage |
|-----|----------|---------------|
| 5 | Slow algorithms, dial-up | ~50 KB/s |
| 10 | Default, good balance | ~100 KB/s |
| 30 | Fast algorithms, smooth | ~300 KB/s |
| 60 | Maximum smoothness | ~600 KB/s |

The server automatically skips frames to maintain the target FPS.

### Viewer Generation

Control viewer HTML generation:

```python
# Generate viewer automatically
await viz.start_streaming(generate_viewer=True)

# Skip viewer generation (use existing HTML)
await viz.start_streaming(generate_viewer=False)
```

## Interactive Controls

### Browser Controls

The viewer HTML provides interactive controls:

- **Pause Button** - Pause the optimization, keeping the server running
- **Resume Button** - Resume a paused optimization
- **Stop Button** - Stop the optimization permanently
- **Status Messages** - See server status and error messages

### Algorithm Integration

Check control state in your optimization loop:

```python
for iteration in range(1000):
    # ... placement step ...

    await viz.capture_and_stream(...)

    # Check if user paused
    if await viz.is_paused():
        print("Paused by user")
        await viz.send_status("info", "Optimization paused")
        await viz.wait_resume()  # Block until resume
        print("Resumed")
        await viz.send_status("info", "Optimization resumed")

    # Check if user stopped
    if await viz.is_stop_requested():
        print("Stopped by user")
        await viz.send_status("warning", f"Stopped at iteration {iteration}")
        break
```

## Message Protocol

### Client → Server Messages

```json
{"type": "pause"}      // Pause optimization
{"type": "resume"}     // Resume optimization
{"type": "stop"}       // Stop optimization
{"type": "ping"}       // Keep-alive ping
```

### Server → Client Messages

#### Frame Update
```json
{
  "type": "frame",
  "timestamp": 1234567890.123,
  "data": {
    "index": 42,
    "label": "Iteration 42",
    "iteration": 42,
    "phase": "refinement",
    "components": {
      "C1": [10.5, 20.3, 0.0],
      "R1": [15.2, 25.1, 90.0]
    },
    "modules": {
      "C1": "power_supply",
      "R1": "analog"
    },
    "forces": {
      "C1": [[1.2, 0.5, "attraction"]]
    },
    "energy": 123.45,
    "max_move": 0.5,
    "overlap_count": 2,
    "total_wire_length": 150.3
  }
}
```

#### Status Message
```json
{
  "type": "status",
  "status": "info",      // "info", "warning", "error", "complete"
  "message": "Starting optimization"
}
```

#### Control State
```json
{
  "type": "control_state",
  "paused": false,
  "stop_requested": false
}
```

#### Welcome Message
```json
{
  "type": "welcome",
  "fps": 10.0,
  "paused": false
}
```

## Performance

### Network Usage

Frame size depends on:
- Number of components (50-100 bytes per component)
- Force vectors (if included)
- Module assignments

**Example: 100-component board at 10 FPS**
- Frame size: ~8 KB (80 bytes/component average)
- Network usage: ~80 KB/s
- Total for 500 iterations: ~4 MB

### Optimization Tips

1. **Reduce Frame Rate** - Lower FPS for slow networks
2. **Skip Force Vectors** - Only send forces every N iterations
3. **Batch Updates** - Process multiple placement steps per frame
4. **Selective Streaming** - Only stream when clients connected

```python
# Only compute forces occasionally
forces = compute_forces() if iteration % 5 == 0 else {}

# Skip streaming if no clients
if viz.server.clients:
    await viz.capture_and_stream(...)
```

## Comparison: Streaming vs Canvas

| Feature | Streaming | Canvas |
|---------|-----------|--------|
| **Timing** | During optimization | After optimization |
| **Latency** | Real-time (< 100ms) | N/A |
| **File Size** | Network only | 3-10 MB |
| **Interactivity** | Full (pause/stop) | Playback only |
| **Frame Rate** | 5-60 FPS (live) | 60 FPS (playback) |
| **History** | Current view only | All frames saved |
| **Setup** | WebSocket server | Single HTML file |

**Use streaming when:**
- Monitoring long-running optimizations
- Need to stop early if converged
- Debugging algorithm behavior live
- Multiple people watching progress

**Use Canvas export when:**
- Analyzing completed optimizations
- Sharing results with others
- Stepping through frames precisely
- No need for live monitoring

## Troubleshooting

### "Address already in use"

**Problem:** Port 8765 is already in use by another process.

**Solution:** Use a different port:
```python
viz = StreamingVisualizer(board, port=8766)
```

Or kill the existing process:
```bash
lsof -i :8765
kill <PID>
```

### No clients connecting

**Problem:** Browser can't connect to WebSocket server.

**Causes:**
1. Firewall blocking port 8765
2. Wrong host/port in viewer HTML
3. Server not started yet

**Solutions:**
```python
# Verify server started
print(f"Server URL: {viz.get_viewer_url()}")

# Check firewall settings
# macOS: System Preferences → Security & Privacy → Firewall
```

### Frames not updating

**Problem:** Browser shows "Connected" but frames don't update.

**Causes:**
1. No calls to `capture_and_stream()`
2. Frame rate too high (frames skipped)
3. Algorithm not running (blocking call)

**Solutions:**
```python
# Verify capture is called
await viz.capture_and_stream(...)  # Add this in your loop

# Lower frame rate
viz = StreamingVisualizer(board, max_fps=5.0)

# Check algorithm is async
await asyncio.sleep(0)  # Yield to event loop
```

### High memory usage

**Problem:** Memory grows during streaming.

**Cause:** Frames are also stored locally for final export.

**Solutions:**
```python
# Don't store frames if only streaming
viz.visualizer.frames = []  # Clear frames periodically

# Or disable frame storage (requires modification)
```

### Browser lag/stuttering

**Problem:** Browser rendering is choppy.

**Causes:**
1. Too many components (>1000)
2. Force vectors overwhelming renderer
3. Browser hardware acceleration disabled

**Solutions:**
```python
# Reduce frame rate
viz = StreamingVisualizer(board, max_fps=5.0)

# Skip force rendering occasionally
forces = {} if iteration % 10 != 0 else compute_forces()

# Enable browser GPU acceleration (browser settings)
```

## Advanced Usage

### Custom Status Messages

Send custom status updates to viewers:

```python
await viz.send_status("info", "Phase 1: Initial placement")
await viz.send_status("warning", "High overlap detected")
await viz.send_status("error", "Component out of bounds")
await viz.send_status("complete", "Optimization converged")
```

### Server Statistics

Monitor server performance:

```python
stats = viz.get_server_stats()
print(f"Clients: {stats['clients_connected']}")
print(f"Frames sent: {stats['frames_sent']}")
print(f"Data sent: {stats['bytes_sent'] / 1024:.1f} KB")
print(f"Average FPS: {stats['avg_fps']:.1f}")
print(f"Uptime: {stats['uptime_seconds']:.1f}s")
```

### Context Manager API

Use `StreamManager` for automatic cleanup:

```python
from atoplace.placement.stream_server import StreamManager

async def optimize():
    async with StreamManager(host='localhost', port=8765) as stream:
        for iteration in range(100):
            # ... placement step ...

            await stream.send_frame(frame_data)

            if stream.is_paused():
                await stream.wait_resume()

            if stream.is_stop_requested():
                break
```

### Multiple Algorithms

Stream from multiple algorithms to same viewer (requires custom HTML):

```python
# Algorithm 1 - port 8765
viz1 = StreamingVisualizer(board1, port=8765)

# Algorithm 2 - port 8766
viz2 = StreamingVisualizer(board2, port=8766)

# Run concurrently
await asyncio.gather(
    optimize_board1(viz1),
    optimize_board2(viz2)
)
```

## Security Considerations

### Network Exposure

The WebSocket server listens on `localhost` by default, restricting access to the local machine.

**To allow remote access:**
```python
# WARNING: This exposes the server to the network
viz = StreamingVisualizer(board, host='0.0.0.0', port=8765)
```

**Security implications:**
- Anyone on the network can connect
- No authentication required
- Control commands (pause/stop) are unauthenticated

**Recommendations:**
- Use `localhost` for single-machine access
- Use firewall rules for network access
- Use SSH tunneling for remote access
- Don't expose to public internet

### Resource Limits

The server has no built-in rate limiting or DoS protection.

**Recommendations:**
- Limit client connections if needed
- Monitor memory/CPU usage
- Use reverse proxy for production

## Dependencies

The streaming system requires:

```bash
pip install websockets>=12.0
```

Or install with optional dependency:

```bash
pip install atoplace[streaming]
```

## Related Documentation

- [Canvas Visualization](CANVAS_VISUALIZATION.md) - Post-process visualization with delta compression
- [Visualization Colors](VISUALIZATION_COLORS.md) - Color configuration
- [Product Plan](PRODUCT_PLAN.md) - Phase 2: Visualization systems

## Future Enhancements

Potential improvements for future releases:

1. **Authentication** - Basic auth or token-based access control
2. **Recording** - Save stream to video file (MP4/WebM)
3. **Replay** - Stream pre-recorded optimization sessions
4. **Multiple Boards** - Compare multiple boards side-by-side
5. **3D View** - Real-time 3D visualization for multi-layer boards
6. **Collaborative** - Multiple users can control the same optimization
7. **Cloud Streaming** - Stream to cloud-hosted viewer
8. **Performance Graphs** - Real-time charts of energy, wire length, etc.
