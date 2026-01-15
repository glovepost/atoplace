"""WebSocket streaming server for real-time placement visualization.

Streams placement frames to connected web clients as optimization runs,
enabling users to see placement evolving in real-time and interact with
the optimization process.

Key features:
- Real-time frame streaming via WebSocket
- Interactive controls (pause/resume/stop)
- Multiple concurrent viewers
- Automatic reconnection handling
- Low latency (<100ms)

Usage:
    server = StreamServer(host='localhost', port=8765)
    await server.start()

    # In visualization loop:
    await server.broadcast_frame(frame_data)
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import asdict
import time

logger = logging.getLogger(__name__)

# Try to import websockets, provide helpful error if missing
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError(
        "websockets library required for streaming. Install with:\n"
        "  pip install websockets\n"
        "  or: pip install atoplace[streaming]"
    )


class StreamServer:
    """WebSocket server for streaming placement frames.

    Broadcasts frames to all connected clients and handles interactive
    control commands (pause, resume, stop).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        max_fps: float = 10.0,
    ):
        """Initialize streaming server.

        Args:
            host: Server host address
            port: Server port
            max_fps: Maximum streaming frame rate (frames per second)
        """
        self.host = host
        self.port = port
        self.max_fps = max_fps
        self.min_frame_interval = 1.0 / max_fps

        # Connected clients
        self.clients: Set[WebSocketServerProtocol] = set()

        # Control state
        self.paused = False
        self.stop_requested = False

        # Server instance
        self.server = None
        self.server_task = None

        # Statistics
        self.frames_sent = 0
        self.bytes_sent = 0
        self.last_frame_time = 0.0
        self.start_time = 0.0

    async def start(self):
        """Start the WebSocket server."""
        self.start_time = time.time()
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info(f"Streaming server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server and disconnect all clients."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Streaming server stopped")

        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
            self.clients.clear()

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new client connection.

        Args:
            websocket: Client WebSocket connection
        """
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        try:
            # Send welcome message with server info
            await websocket.send(json.dumps({
                "type": "welcome",
                "fps": self.max_fps,
                "paused": self.paused,
            }))

            # Handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_addr}: {message}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_addr}: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Error in client handler for {client_addr}: {e}")
        finally:
            self.clients.discard(websocket)

    async def _handle_client_message(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle a message from a client.

        Supported commands:
        - {"type": "pause"} - Pause optimization
        - {"type": "resume"} - Resume optimization
        - {"type": "stop"} - Stop optimization
        - {"type": "ping"} - Keep-alive ping

        Args:
            websocket: Client WebSocket connection
            data: Parsed message data
        """
        msg_type = data.get("type")

        if msg_type == "pause":
            self.paused = True
            logger.info("Optimization paused by client")
            await self._broadcast_control_state()

        elif msg_type == "resume":
            self.paused = False
            logger.info("Optimization resumed by client")
            await self._broadcast_control_state()

        elif msg_type == "stop":
            self.stop_requested = True
            logger.info("Optimization stop requested by client")
            await self._broadcast_control_state()

        elif msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _broadcast_control_state(self):
        """Broadcast current control state to all clients."""
        message = json.dumps({
            "type": "control_state",
            "paused": self.paused,
            "stop_requested": self.stop_requested,
        })
        await self._broadcast(message)

    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """Broadcast a visualization frame to all connected clients.

        Implements frame rate limiting to avoid overwhelming clients.

        Args:
            frame_data: Frame data dictionary (will be JSON serialized)
        """
        if not self.clients:
            return  # No clients connected, skip

        # Frame rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.min_frame_interval:
            return  # Too soon, skip this frame

        self.last_frame_time = current_time

        # Prepare frame message
        message = json.dumps({
            "type": "frame",
            "data": frame_data,
            "timestamp": current_time,
        })

        # Broadcast to all clients
        await self._broadcast(message)

        # Update statistics
        self.frames_sent += 1
        self.bytes_sent += len(message)

    async def broadcast_stats(self, stats: Dict[str, Any]):
        """Broadcast optimization statistics to clients.

        Args:
            stats: Statistics dictionary (energy, wire length, etc.)
        """
        message = json.dumps({
            "type": "stats",
            "data": stats,
        })
        await self._broadcast(message)

    async def broadcast_status(self, status: str, message: str = ""):
        """Broadcast status message to clients.

        Args:
            status: Status type ("info", "warning", "error", "complete")
            message: Status message text
        """
        msg = json.dumps({
            "type": "status",
            "status": status,
            "message": message,
        })
        await self._broadcast(msg)

    async def _broadcast(self, message: str):
        """Broadcast a message to all connected clients.

        Args:
            message: JSON string to broadcast
        """
        if not self.clients:
            return

        # Send to all clients, removing disconnected ones
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    def is_paused(self) -> bool:
        """Check if optimization is paused by user."""
        return self.paused

    def is_stop_requested(self) -> bool:
        """Check if user requested stopping optimization."""
        return self.stop_requested

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with server stats
        """
        uptime = time.time() - self.start_time if self.start_time else 0
        return {
            "clients_connected": len(self.clients),
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "uptime_seconds": uptime,
            "avg_fps": self.frames_sent / uptime if uptime > 0 else 0,
        }


class StreamManager:
    """Context manager for streaming server lifecycle.

    Handles server startup, shutdown, and provides convenient API for
    streaming frames during placement optimization.

    Usage:
        async with StreamManager(host='localhost', port=8765) as stream:
            for iteration in range(100):
                # ... placement step ...

                # Stream frame
                await stream.send_frame({
                    'iteration': iteration,
                    'components': components,
                    'energy': energy
                })

                # Check for user interaction
                if stream.is_paused():
                    await stream.wait_resume()

                if stream.is_stop_requested():
                    break
    """

    def __init__(self, host: str = "localhost", port: int = 8765, max_fps: float = 10.0):
        """Initialize stream manager.

        Args:
            host: Server host address
            port: Server port
            max_fps: Maximum streaming frame rate
        """
        self.server = StreamServer(host, port, max_fps)
        self.url = f"ws://{host}:{port}"

    async def __aenter__(self):
        """Start server when entering context."""
        await self.server.start()
        logger.info(f"Streaming available at {self.url}")
        logger.info("Open stream_viewer.html in your browser to watch")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop server when exiting context."""
        await self.server.stop()

    async def send_frame(self, frame_data: Dict[str, Any]):
        """Send a frame to connected viewers.

        Args:
            frame_data: Frame data dictionary
        """
        await self.server.broadcast_frame(frame_data)

    async def send_stats(self, stats: Dict[str, Any]):
        """Send optimization statistics to viewers.

        Args:
            stats: Statistics dictionary
        """
        await self.server.broadcast_stats(stats)

    async def send_status(self, status: str, message: str = ""):
        """Send status message to viewers.

        Args:
            status: Status type ("info", "warning", "error", "complete")
            message: Status message
        """
        await self.server.broadcast_status(status, message)

    def is_paused(self) -> bool:
        """Check if user paused optimization."""
        return self.server.is_paused()

    def is_stop_requested(self) -> bool:
        """Check if user requested stop."""
        return self.server.is_stop_requested()

    async def wait_resume(self):
        """Wait until user resumes optimization.

        Blocks until pause state is cleared.
        """
        while self.server.is_paused():
            await asyncio.sleep(0.1)

    def get_viewer_url(self) -> str:
        """Get URL for web viewer.

        Returns:
            WebSocket URL string
        """
        return self.url

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with server stats
        """
        return self.server.get_stats()
