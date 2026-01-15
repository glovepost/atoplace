#!/usr/bin/env python3
"""Demo script for real-time streaming visualization.

This runs a longer optimization (300 iterations) so users have time to
open the browser and watch the streaming in action.
"""

import asyncio
import logging
import random
import math
from pathlib import Path
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MockComponent:
    """Mock component for testing."""
    reference: str
    x: float
    y: float
    rotation: float
    width: float
    height: float


class MockBoard:
    """Mock board for testing streaming visualizer."""

    def __init__(self, num_components=30):
        """Create mock board with random components."""
        self.components = {}
        self.outline = type('obj', (object,), {
            'min_x': 0.0, 'min_y': 0.0,
            'max_x': 120.0, 'max_y': 100.0
        })()

        # Create mock components scattered randomly
        component_types = ['C', 'R', 'U', 'L', 'D']
        for i in range(num_components):
            comp_type = random.choice(component_types)
            ref = f"{comp_type}{i+1}"

            self.components[ref] = MockComponent(
                reference=ref,
                x=random.uniform(10, 110),
                y=random.uniform(10, 90),
                rotation=random.uniform(0, 360),
                width=random.uniform(2, 10),
                height=random.uniform(2, 10),
            )


class MockPlacementVisualizer:
    """Mock visualizer that mimics PlacementVisualizer interface."""

    def __init__(self, board):
        """Initialize mock visualizer."""
        self.board = board
        self.frames = []
        self.static_props = {}

        # Mock static properties
        for ref, comp in board.components.items():
            self.static_props[ref] = {
                'width': comp.width,
                'height': comp.height,
                'pads': []
            }

        # Board bounds
        self.min_x = board.outline.min_x
        self.min_y = board.outline.min_y
        self.max_x = board.outline.max_x
        self.max_y = board.outline.max_y

    def capture_from_board(self, label, iteration, phase, modules, forces, energy, max_move):
        """Capture current board state."""
        frame = type('obj', (object,), {
            'index': len(self.frames),
            'label': label,
            'iteration': iteration,
            'phase': phase,
            'components': {
                ref: (comp.x, comp.y, comp.rotation)
                for ref, comp in self.board.components.items()
            },
            'modules': modules or {},
            'forces': forces or {},
            'overlaps': [],
            'movement': {},
            'connections': [],
            'energy': energy,
            'max_move': max_move,
            'overlap_count': 0,
            'total_wire_length': 0.0,
        })()
        self.frames.append(frame)


async def demo_streaming():
    """Demo streaming visualizer with longer optimization."""

    from atoplace.placement.stream_server import StreamServer
    from atoplace.placement.stream_viewer import generate_stream_viewer_html

    logger.info("=" * 70)
    logger.info("ATOPLACE STREAMING VISUALIZATION DEMO")
    logger.info("=" * 70)
    logger.info("Creating mock board with 30 components...")
    board = MockBoard(num_components=30)

    logger.info(f"Mock board created: {len(board.components)} components")

    # Create mock visualizer
    viz = MockPlacementVisualizer(board)

    # Create streaming server
    host = 'localhost'
    port = 8765
    server = StreamServer(host, port, max_fps=10.0)

    try:
        # Start streaming server
        logger.info("Starting WebSocket streaming server...")
        await server.start()

        # Generate viewer HTML
        logger.info("Generating stream viewer HTML...")
        board_bounds = (viz.min_x, viz.min_y, viz.max_x, viz.max_y)
        html_path = Path("placement_debug")
        html_path.mkdir(exist_ok=True)
        html_path = html_path / "stream_viewer.html"

        generate_stream_viewer_html(
            websocket_url=f"ws://{host}:{port}",
            board_bounds=board_bounds,
            static_props=viz.static_props,
            output_path=html_path,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("STREAMING SERVER READY")
        logger.info("=" * 70)
        logger.info(f"WebSocket URL: ws://{host}:{port}")
        logger.info(f"Viewer HTML:   {html_path}")
        logger.info("")
        logger.info("⚠️  OPEN THE VIEWER NOW:")
        logger.info("   open placement_debug/stream_viewer.html")
        logger.info("")
        logger.info("Waiting 5 seconds for you to open the browser...")
        logger.info("=" * 70)

        # Wait for user to open browser
        for i in range(5, 0, -1):
            logger.info(f"Starting in {i}...")
            await asyncio.sleep(1)

        await server.broadcast_status("info", "Starting placement optimization")

        # Longer optimization - 300 iterations
        num_iterations = 300
        components = list(board.components.values())

        # Board center
        board_center_x = (board.outline.min_x + board.outline.max_x) / 2
        board_center_y = (board.outline.min_y + board.outline.max_y) / 2

        logger.info("")
        logger.info(f"🚀 Starting optimization: {num_iterations} iterations")
        logger.info("Components will gradually converge toward center...")
        logger.info("Watch your browser for live updates!")
        logger.info("")

        for iteration in range(num_iterations):
            # Simulate placement movements
            for comp in components:
                dx = board_center_x - comp.x
                dy = board_center_y - comp.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist > 0.1:
                    # Slower convergence for better visualization
                    step_size = 0.3 * (1.0 - iteration / num_iterations)
                    comp.x += (dx/dist) * step_size + random.gauss(0, 0.08)
                    comp.y += (dy/dist) * step_size + random.gauss(0, 0.08)
                    comp.rotation += random.gauss(0, 0.5)

            # Module assignments
            modules = {}
            for comp in components:
                if 'C' in comp.reference:
                    modules[comp.reference] = 'power_supply'
                elif 'R' in comp.reference:
                    modules[comp.reference] = 'analog'
                elif 'U' in comp.reference:
                    modules[comp.reference] = 'microcontroller'
                elif 'L' in comp.reference:
                    modules[comp.reference] = 'power_supply'
                else:
                    modules[comp.reference] = 'digital'

            # Force vectors (every 3 iterations)
            forces = {}
            if iteration % 3 == 0:
                for comp in components:
                    dx = board_center_x - comp.x
                    dy = board_center_y - comp.y
                    magnitude = math.sqrt(dx*dx + dy*dy)
                    if magnitude > 0.1:
                        forces[comp.reference] = [
                            (dx/10, dy/10, "attraction")
                        ]

            # Energy and movement decay
            energy = 1000.0 * (1.0 - iteration / num_iterations)
            max_move = 2.0 * (1.0 - iteration / num_iterations)

            # Capture frame
            viz.capture_from_board(
                label=f"Iteration {iteration}",
                iteration=iteration,
                phase="convergence",
                modules=modules,
                forces=forces,
                energy=energy,
                max_move=max_move,
            )

            # Stream frame
            if server.clients and viz.frames:
                frame = viz.frames[-1]

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
                    "movement": {},
                    "connections": frame.connections,
                    "energy": frame.energy,
                    "max_move": frame.max_move,
                    "overlap_count": frame.overlap_count,
                    "total_wire_length": frame.total_wire_length,
                }

                await server.broadcast_frame(frame_data)

            # Log progress
            if iteration % 20 == 0:
                client_count = len(server.clients)
                status = "📺 STREAMING" if client_count > 0 else "⏸️  Waiting for clients"
                logger.info(f"{status} | Iteration {iteration}/{num_iterations} | Energy: {energy:.1f} | Clients: {client_count}")

            # Check for user interaction
            if server.is_paused():
                logger.info("⏸️  Optimization PAUSED by user")
                await server.broadcast_status("info", "Optimization paused")
                while server.is_paused():
                    await asyncio.sleep(0.1)
                logger.info("▶️  Optimization RESUMED")
                await server.broadcast_status("info", "Optimization resumed")

            if server.is_stop_requested():
                logger.warning("⏹️  Optimization STOPPED by user")
                await server.broadcast_status("warning", f"Stopped at iteration {iteration}")
                break

            # Slower updates for better visualization
            await asyncio.sleep(0.1)

        # Complete
        logger.info("")
        logger.info("✅ Optimization complete!")
        await server.broadcast_status("complete", f"Optimization finished after {iteration+1} iterations")

        # Statistics
        stats = server.get_stats()
        logger.info("")
        logger.info("=" * 70)
        logger.info("STREAMING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Clients connected:  {stats['clients_connected']}")
        logger.info(f"Frames sent:        {stats['frames_sent']}")
        logger.info(f"Data transmitted:   {stats['bytes_sent']/1024:.1f} KB")
        logger.info(f"Average FPS:        {stats['avg_fps']:.1f}")
        logger.info(f"Uptime:             {stats['uptime_seconds']:.1f} seconds")
        logger.info("=" * 70)

        # Keep server alive longer
        logger.info("")
        logger.info("Keeping server alive for 30 seconds...")
        logger.info("Try the pause/resume/stop controls in your browser!")
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await server.broadcast_status("error", f"Error: {str(e)}")
    finally:
        logger.info("Stopping streaming server...")
        await server.stop()
        logger.info("Done!")


def main():
    """Main entry point."""
    try:
        asyncio.run(demo_streaming())
    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    main()
