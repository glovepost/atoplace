"""Streaming-enabled placement visualizer.

Wraps PlacementVisualizer to add real-time WebSocket streaming capabilities
without requiring async/await in the placement algorithm code.

Usage:
    # Synchronous usage (backwards compatible)
    viz = PlacementVisualizer(board)
    viz.capture_from_board(...)

    # Streaming usage (async)
    async def optimize_with_streaming():
        viz = StreamingVisualizer(board, host='localhost', port=8765)
        await viz.start_streaming()

        for iteration in range(100):
            # ... placement step ...

            # Capture and stream frame
            await viz.capture_and_stream(
                label=f"Iteration {iteration}",
                iteration=iteration,
                modules=modules,
                forces=forces
            )

            # Check for user interaction
            if await viz.is_paused():
                await viz.wait_resume()

            if await viz.is_stop_requested():
                break

        await viz.stop_streaming()
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .visualizer import PlacementVisualizer
from .stream_server import StreamServer
from .stream_viewer import generate_stream_viewer_html

logger = logging.getLogger(__name__)


class StreamingVisualizer:
    """Placement visualizer with real-time WebSocket streaming.

    Combines PlacementVisualizer for frame capture with StreamServer
    for real-time broadcasting to web clients.
    """

    def __init__(
        self,
        board,
        host: str = "localhost",
        port: int = 8765,
        max_fps: float = 10.0,
        grid_spacing: float = 1.27,
    ):
        """Initialize streaming visualizer.

        Args:
            board: Board instance
            host: WebSocket server host
            port: WebSocket server port
            max_fps: Maximum streaming frame rate
            grid_spacing: Grid spacing for visualization
        """
        # Underlying visualizer for frame capture
        self.visualizer = PlacementVisualizer(board, grid_spacing)

        # WebSocket server for streaming
        self.server = StreamServer(host, port, max_fps)
        self.host = host
        self.port = port

        # Streaming state
        self.streaming = False
        self._stream_task = None

    async def start_streaming(self, generate_viewer: bool = True):
        """Start the WebSocket server and optionally generate viewer HTML.

        Args:
            generate_viewer: If True, generate HTML viewer file
        """
        await self.server.start()
        self.streaming = True

        viewer_url = f"ws://{self.host}:{self.port}"
        logger.info(f"Streaming server started at {viewer_url}")

        if generate_viewer:
            # Generate viewer HTML
            board_bounds = (
                self.visualizer.min_x,
                self.visualizer.min_y,
                self.visualizer.max_x,
                self.visualizer.max_y,
            )

            html_path = Path("placement_debug/stream_viewer.html")
            generate_stream_viewer_html(
                websocket_url=viewer_url,
                board_bounds=board_bounds,
                static_props=self.visualizer.static_props,
                output_path=html_path,
            )

            logger.info(f"Generated viewer at {html_path}")
            logger.info(f"Open {html_path} in your browser to watch in real-time")

    async def stop_streaming(self):
        """Stop the WebSocket server."""
        self.streaming = False
        await self.server.stop()
        logger.info("Streaming server stopped")

    async def capture_and_stream(
        self,
        label: str,
        iteration: int = 0,
        phase: str = "refinement",
        modules: Dict[str, str] = None,
        forces: Dict[str, List[Tuple[float, float, str]]] = None,
        energy: float = 0.0,
        max_move: float = 0.0,
    ):
        """Capture frame and stream to connected clients.

        This is the main method to use in placement loops. It captures
        the frame using the underlying visualizer and broadcasts it via
        WebSocket to all connected clients.

        Args:
            label: Frame description
            iteration: Current iteration
            phase: Algorithm phase
            modules: Module assignments
            forces: Force vectors
            energy: System energy
            max_move: Maximum component movement
        """
        # Capture frame using underlying visualizer
        self.visualizer.capture_from_board(
            label=label,
            iteration=iteration,
            phase=phase,
            modules=modules,
            forces=forces,
            energy=energy,
            max_move=max_move,
        )

        # Stream frame if server is running
        if self.streaming and self.server.clients:
            # Get the last captured frame
            if self.visualizer.frames:
                frame = self.visualizer.frames[-1]

                # Convert to streamable format (simplified, no pads in stream)
                frame_data = {
                    "index": frame.index,
                    "label": frame.label,
                    "iteration": frame.iteration,
                    "phase": frame.phase,
                    "components": {
                        ref: list(comp) for ref, comp in frame.components.items()
                    },
                    "modules": frame.modules,
                    "forces": frame.forces,
                    "overlaps": frame.overlaps,
                    "movement": {
                        ref: list(mov) for ref, mov in frame.movement.items()
                    },
                    "connections": frame.connections,
                    "energy": frame.energy,
                    "max_move": frame.max_move,
                    "overlap_count": frame.overlap_count,
                    "total_wire_length": frame.total_wire_length,
                }

                # Broadcast to clients
                await self.server.broadcast_frame(frame_data)

    async def is_paused(self) -> bool:
        """Check if user paused the optimization.

        Returns:
            True if paused, False otherwise
        """
        return self.server.is_paused()

    async def is_stop_requested(self) -> bool:
        """Check if user requested stopping the optimization.

        Returns:
            True if stop requested, False otherwise
        """
        return self.server.is_stop_requested()

    async def wait_resume(self):
        """Wait until user resumes optimization.

        Blocks until pause state is cleared. Use this in your optimization
        loop when is_paused() returns True.
        """
        while self.server.is_paused():
            await asyncio.sleep(0.1)

    async def send_status(self, status: str, message: str = ""):
        """Send status message to connected clients.

        Args:
            status: Status type ("info", "warning", "error", "complete")
            message: Status message text
        """
        if self.streaming:
            await self.server.broadcast_status(status, message)

    def export_canvas_html(self, filename: str = "placement_canvas.html", output_dir: str = "placement_debug"):
        """Export Canvas-based HTML with all captured frames.

        This exports the complete visualization after optimization finishes,
        using the Canvas renderer with delta compression.

        Args:
            filename: Output filename
            output_dir: Output directory

        Returns:
            Path to generated HTML file
        """
        return self.visualizer.export_canvas_html(filename, output_dir)

    def export_html_report(self, filename: str = "placement_debug.html", output_dir: str = "placement_debug"):
        """Export legacy SVG-based HTML report.

        Args:
            filename: Output filename
            output_dir: Output directory

        Returns:
            Path to generated HTML file
        """
        return self.visualizer.export_html_report(filename, output_dir)

    def get_server_stats(self) -> Dict:
        """Get streaming server statistics.

        Returns:
            Dictionary with server stats
        """
        return self.server.get_stats()

    def get_viewer_url(self) -> str:
        """Get URL for the web viewer.

        Returns:
            WebSocket URL string
        """
        return f"ws://{self.host}:{self.port}"

    @property
    def frames(self):
        """Access captured frames from underlying visualizer."""
        return self.visualizer.frames

    @property
    def static_props(self):
        """Access static component properties from underlying visualizer."""
        return self.visualizer.static_props


def create_streaming_visualizer(
    board,
    host: str = "localhost",
    port: int = 8765,
    max_fps: float = 10.0
) -> StreamingVisualizer:
    """Create a streaming visualizer for real-time visualization.

    Args:
        board: Board instance
        host: WebSocket server host
        port: WebSocket server port
        max_fps: Maximum streaming frame rate

    Returns:
        StreamingVisualizer instance
    """
    return StreamingVisualizer(board, host, port, max_fps)


# Example usage
async def example_streaming_optimization():
    """Example of using StreamingVisualizer in an optimization loop."""
    from atoplace.board.kicad_adapter import Board

    # Load board
    board = Board.from_kicad("example.kicad_pcb")

    # Create streaming visualizer
    viz = StreamingVisualizer(board, host='localhost', port=8765, max_fps=10.0)

    try:
        # Start streaming server and generate viewer
        await viz.start_streaming(generate_viewer=True)
        await viz.send_status("info", "Starting optimization")

        # Optimization loop
        for iteration in range(100):
            # Simulate placement step
            await asyncio.sleep(0.1)  # Replace with actual placement algorithm

            # Capture and stream frame
            await viz.capture_and_stream(
                label=f"Iteration {iteration}",
                iteration=iteration,
                phase="refinement",
                modules={},  # Module assignments
                forces={},   # Force vectors
                energy=100.0 - iteration,  # Decreasing energy
                max_move=1.0 / (iteration + 1),  # Decreasing movement
            )

            # Check for user interaction
            if await viz.is_paused():
                await viz.send_status("info", "Optimization paused by user")
                await viz.wait_resume()
                await viz.send_status("info", "Optimization resumed")

            if await viz.is_stop_requested():
                await viz.send_status("warning", "Optimization stopped by user")
                break

        # Optimization complete
        await viz.send_status("complete", "Optimization finished successfully")

        # Export final visualization
        viz.export_canvas_html("final_placement.html")

    finally:
        # Always stop the server
        await viz.stop_streaming()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_streaming_optimization())
