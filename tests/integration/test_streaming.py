#!/usr/bin/env python3
"""Test script for real-time streaming visualization.

This script demonstrates the StreamingVisualizer by running a simple
placement optimization with live WebSocket streaming to a web browser.

Usage:
    python test_streaming.py [board.kicad_pcb]

Then open placement_debug/stream_viewer.html in your browser to watch
the optimization in real-time.
"""

import asyncio
import sys
import logging
import random
import math
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_streaming_placement():
    """Test streaming visualizer with a simple placement simulation."""

    # Import atoplace modules
    from atoplace.board.kicad_adapter import Board
    from atoplace.placement.streaming_visualizer import StreamingVisualizer

    # Load board
    if len(sys.argv) > 1:
        board_path = sys.argv[1]
    else:
        # Default to tutorial board
        board_path = "examples/tutorial/tutorial_with_components.kicad_pcb"

    logger.info(f"Loading board from {board_path}")

    if not Path(board_path).exists():
        logger.error(f"Board file not found: {board_path}")
        logger.info("Available test boards:")
        logger.info("  - examples/tutorial/tutorial_with_components.kicad_pcb")
        logger.info("  - examples/dogtracker/layouts/default/default.kicad_pcb")
        return

    try:
        board = Board.from_kicad(board_path)
    except Exception as e:
        logger.error(f"Failed to load board: {e}")
        return

    logger.info(f"Board loaded: {len(board.components)} components")

    if len(board.components) == 0:
        logger.warning("Board has no components! Streaming test may not be very interesting.")
        logger.info("Try using examples/tutorial/tutorial_with_components.kicad_pcb")
        return

    # Create streaming visualizer
    viz = StreamingVisualizer(
        board,
        host='localhost',
        port=8765,
        max_fps=10.0
    )

    try:
        # Start streaming server and generate viewer
        logger.info("Starting WebSocket streaming server...")
        await viz.start_streaming(generate_viewer=True)

        logger.info("=" * 70)
        logger.info("STREAMING SERVER READY")
        logger.info("=" * 70)
        logger.info(f"WebSocket URL: {viz.get_viewer_url()}")
        logger.info("Viewer HTML:   placement_debug/stream_viewer.html")
        logger.info("")
        logger.info("Open stream_viewer.html in your browser to watch in real-time!")
        logger.info("=" * 70)

        await viz.send_status("info", "Starting placement optimization")

        # Simulate placement optimization
        num_iterations = 100
        components = list(board.components.values())

        # Initialize random positions within board bounds
        board_center_x = (board.outline.min_x + board.outline.max_x) / 2
        board_center_y = (board.outline.min_y + board.outline.max_y) / 2
        board_width = board.outline.max_x - board.outline.min_x
        board_height = board.outline.max_y - board.outline.min_y

        logger.info(f"Starting optimization: {num_iterations} iterations")

        for iteration in range(num_iterations):
            # Simulate placement movements
            # Components gradually converge toward center with some randomness
            for comp in components:
                # Get current position
                cx, cy = comp.position

                # Calculate direction toward center
                dx = board_center_x - cx
                dy = board_center_y - cy
                dist = math.sqrt(dx*dx + dy*dy)

                if dist > 0.1:  # If not at center
                    # Move toward center with some randomness
                    step_size = 0.5 * (1.0 - iteration / num_iterations)  # Decreasing step size
                    nx = cx + (dx/dist) * step_size + random.gauss(0, 0.1)
                    ny = cy + (dy/dist) * step_size + random.gauss(0, 0.1)

                    # Slight rotation
                    rot = comp.rotation + random.gauss(0, 1.0)

                    # Update position
                    comp.position = (nx, ny)
                    comp.rotation = rot

            # Simulate module assignments (group components by type)
            modules = {}
            for i, comp in enumerate(components):
                if 'C' in comp.reference:
                    modules[comp.reference] = 'power_supply'
                elif 'R' in comp.reference:
                    modules[comp.reference] = 'analog'
                elif 'U' in comp.reference:
                    modules[comp.reference] = 'microcontroller'
                else:
                    modules[comp.reference] = 'digital'

            # Simulate force vectors (smaller as we converge)
            forces = {}
            if iteration % 5 == 0:  # Only show forces every 5 iterations to reduce data
                for comp in components:
                    cx, cy = comp.position
                    dx = board_center_x - cx
                    dy = board_center_y - cy
                    magnitude = math.sqrt(dx*dx + dy*dy)
                    if magnitude > 0.1:
                        forces[comp.reference] = [
                            (dx/10, dy/10, "attraction")
                        ]

            # Simulate energy decay
            energy = 1000.0 * (1.0 - iteration / num_iterations)
            max_move = 2.0 * (1.0 - iteration / num_iterations)

            # Capture and stream frame
            await viz.capture_and_stream(
                label=f"Iteration {iteration}",
                iteration=iteration,
                phase="convergence",
                modules=modules,
                forces=forces,
                energy=energy,
                max_move=max_move,
            )

            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}/{num_iterations} - Energy: {energy:.1f}")

            # Check for user interaction
            if await viz.is_paused():
                logger.info("⏸ Optimization PAUSED by user")
                await viz.send_status("info", "Optimization paused by user")
                await viz.wait_resume()
                logger.info("▶ Optimization RESUMED")
                await viz.send_status("info", "Optimization resumed")

            if await viz.is_stop_requested():
                logger.warning("⏹ Optimization STOPPED by user")
                await viz.send_status("warning", f"Optimization stopped by user at iteration {iteration}")
                break

            # Small delay to simulate real computation
            await asyncio.sleep(0.05)

        # Optimization complete
        logger.info("✓ Optimization complete!")
        await viz.send_status("complete", f"Optimization finished after {iteration+1} iterations")

        # Export final visualization
        logger.info("Exporting final visualization...")
        html_path = viz.export_canvas_html("test_streaming_final.html")
        logger.info(f"Final visualization: {html_path}")

        # Show statistics
        stats = viz.get_server_stats()
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

        # Keep server alive for a bit so clients can see final frame
        logger.info("Keeping server alive for 10 seconds...")
        await asyncio.sleep(10)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        await viz.send_status("error", f"Error: {str(e)}")
    finally:
        # Always stop the server
        logger.info("Stopping streaming server...")
        await viz.stop_streaming()
        logger.info("Done!")


def main():
    """Main entry point."""
    try:
        asyncio.run(test_streaming_placement())
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
