"""
IPC Protocol for MCP Server <-> KiCad Bridge Communication

Uses Unix domain sockets with a JSON-RPC-like protocol.
Designed to work with Python 3.9 (KiCad) and Python 3.10+ (MCP).
"""

import json
import socket
import os
import uuid
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

# Default socket path
DEFAULT_SOCKET_PATH = "/tmp/atoplace-bridge.sock"


@dataclass
class IPCRequest:
    """Request message format."""

    id: str
    method: str
    params: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "IPCRequest":
        d = json.loads(data)
        return cls(id=d["id"], method=d["method"], params=d.get("params", {}))


@dataclass
class IPCResponse:
    """Response message format."""

    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = {"id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "IPCResponse":
        d = json.loads(data)
        return cls(id=d["id"], result=d.get("result"), error=d.get("error"))

    @property
    def success(self) -> bool:
        return self.error is None


class IPCError(Exception):
    """IPC communication error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# =============================================================================
# IPC Client (used by MCP Server to talk to KiCad Bridge)
# =============================================================================


class IPCClient:
    """
    Client for communicating with the KiCad Bridge.

    Features:
    - Auto-reconnect on connection loss
    - Configurable timeout
    - Clean error handling
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        timeout: float = 30.0,
        auto_reconnect: bool = True,
        max_retries: int = 3,
    ):
        self.socket_path = socket_path
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self._socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Connect to the bridge server."""
        if self._socket:
            return True

        if not os.path.exists(self.socket_path):
            logger.debug("Bridge socket not found: %s", self.socket_path)
            return False

        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect(self.socket_path)
            logger.info("Connected to bridge at %s", self.socket_path)
            return True
        except Exception as e:
            logger.debug("Failed to connect to bridge: %s", e)
            self._socket = None
            return False

    def disconnect(self):
        """Disconnect from the bridge server."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def is_connected(self) -> bool:
        """Check if connected to bridge."""
        return self._socket is not None

    def ensure_connected(self) -> bool:
        """Ensure connection is established, reconnecting if needed."""
        if self._socket:
            return True
        return self.connect()

    def call(self, method: str, **params) -> Dict[str, Any]:
        """
        Call a method on the bridge.

        Args:
            method: Method name to call
            **params: Method parameters

        Returns:
            Result dictionary

        Raises:
            IPCError: On communication or method error
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._do_call(method, params)
            except (ConnectionError, BrokenPipeError, socket.timeout) as e:
                last_error = e
                self.disconnect()

                if not self.auto_reconnect or attempt >= self.max_retries - 1:
                    break

                logger.debug("Connection lost, reconnecting (attempt %d)", attempt + 1)
                time.sleep(0.1 * (attempt + 1))  # Backoff

                if not self.connect():
                    break

        raise IPCError("connection_failed", f"Bridge communication failed: {last_error}")

    def _do_call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single RPC call."""
        if not self.ensure_connected():
            raise IPCError("not_connected", "Not connected to KiCad bridge")

        request = IPCRequest(id=str(uuid.uuid4()), method=method, params=params)

        # Send request
        data = request.to_json() + "\n"
        self._socket.sendall(data.encode("utf-8"))

        # Receive response
        response_data = self._recv_line()
        response = IPCResponse.from_json(response_data)

        if response.id != request.id:
            raise IPCError("id_mismatch", "Response ID doesn't match request")

        if response.error:
            raise IPCError(
                response.error.get("code", "unknown"),
                response.error.get("message", "Unknown error"),
            )

        return response.result or {}

    def _recv_line(self) -> str:
        """Receive a newline-terminated line from socket."""
        chunks = []
        while True:
            chunk = self._socket.recv(8192)
            if not chunk:
                raise ConnectionError("Connection closed by bridge")
            chunks.append(chunk)
            if b"\n" in chunk:
                break

        data = b"".join(chunks)
        return data.decode("utf-8").strip()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# =============================================================================
# IPC Server (used by KiCad Bridge to receive commands)
# =============================================================================


class IPCServer:
    """
    Server for receiving commands from the MCP server.

    Used by the KiCad bridge process (Python 3.9) to handle
    commands from the MCP server (Python 3.10+).
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self._socket: Optional[socket.socket] = None
        self._handlers: Dict[str, Callable] = {}
        self._running = False

    def register_handler(self, method: str, handler: Callable):
        """Register a handler for a method."""
        self._handlers[method] = handler

    def start(self):
        """Start the server."""
        # Remove existing socket if present
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(self.socket_path)
        self._socket.listen(1)

        # Make socket accessible
        os.chmod(self.socket_path, 0o666)

        logger.info("Bridge server listening on %s", self.socket_path)
        self._running = True

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._socket:
            self._socket.close()
            self._socket = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def serve_forever(self):
        """Accept and handle connections until stopped."""
        if not self._socket:
            self.start()

        logger.info("Bridge server ready")

        while self._running:
            try:
                self._socket.settimeout(1.0)
                try:
                    conn, addr = self._socket.accept()
                except socket.timeout:
                    continue

                logger.info("Client connected")
                self._handle_connection(conn)

            except Exception as e:
                if self._running:
                    logger.error("Server error: %s", e)

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        conn.settimeout(300.0)  # 5 minute timeout per request
        buffer = ""

        try:
            while self._running:
                try:
                    data = conn.recv(8192)
                    if not data:
                        break

                    buffer += data.decode("utf-8")

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            response = self._handle_request(line)
                            conn.sendall((response.to_json() + "\n").encode("utf-8"))

                except socket.timeout:
                    continue

        except Exception as e:
            logger.error("Connection error: %s", e)
        finally:
            conn.close()
            logger.info("Client disconnected")

    def _handle_request(self, data: str) -> IPCResponse:
        """Handle a single request and return response."""
        try:
            request = IPCRequest.from_json(data)
        except json.JSONDecodeError as e:
            return IPCResponse(id="unknown", error={"code": "parse_error", "message": str(e)})

        handler = self._handlers.get(request.method)
        if not handler:
            return IPCResponse(
                id=request.id,
                error={"code": "method_not_found", "message": f"Unknown method: {request.method}"},
            )

        try:
            result = handler(**request.params)
            return IPCResponse(id=request.id, result=result)
        except Exception as e:
            logger.exception("Handler error for %s", request.method)
            return IPCResponse(id=request.id, error={"code": "handler_error", "message": str(e)})

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# =============================================================================
# Board Serialization (works with Python 3.9)
# =============================================================================


def _to_serializable(value):
    """Convert a value to a JSON-serializable type."""
    if value is None:
        return None
    # Handle enums by getting their value or name
    if hasattr(value, "value"):
        return value.value if isinstance(value.value, (str, int, float, bool)) else str(value.value)
    if hasattr(value, "name"):
        return value.name
    return value


def serialize_board(board) -> Dict[str, Any]:
    """
    Serialize a Board object to a dictionary.

    This function is designed to work with Python 3.9 and doesn't
    use any 3.10+ features.
    """
    components = {}
    for ref, comp in board.components.items():
        pads = []
        for pad in comp.pads:
            pads.append(
                {
                    "number": pad.number,
                    "x": pad.x,
                    "y": pad.y,
                    "width": pad.width,
                    "height": pad.height,
                    "net": pad.net,
                    "shape": _to_serializable(getattr(pad, "shape", None)),
                }
            )

        components[ref] = {
            "reference": comp.reference,
            "footprint": comp.footprint,
            "value": comp.value,
            "x": comp.x,
            "y": comp.y,
            "width": comp.width,
            "height": comp.height,
            "rotation": comp.rotation,
            "layer": _to_serializable(comp.layer),
            "locked": comp.locked,
            "pads": pads,
            "fields": getattr(comp, "fields", {}),
        }

    nets = {}
    for name, net in board.nets.items():
        nets[name] = {
            "name": net.name,
            "code": net.code,
            "is_power": net.is_power,
            "is_ground": net.is_ground,
            "connections": list(net.connections),
        }

    outline = None
    if board.outline:
        outline = {
            "width": board.outline.width,
            "height": board.outline.height,
            "origin_x": board.outline.origin_x,
            "origin_y": board.outline.origin_y,
            "has_outline": board.outline.has_outline,
        }

    return {
        "name": board.name,
        "components": components,
        "nets": nets,
        "outline": outline,
    }


def deserialize_board(data: Dict[str, Any]):
    """
    Deserialize a dictionary to a Board object.

    This function is designed to work with Python 3.9.
    """
    from ..board.abstraction import Board, BoardOutline, Component, Net, Pad

    board = Board(name=data.get("name", "board"))

    # Deserialize outline
    if data.get("outline"):
        o = data["outline"]
        board.outline = BoardOutline(
            width=o["width"],
            height=o["height"],
            origin_x=o.get("origin_x", 0),
            origin_y=o.get("origin_y", 0),
            has_outline=o.get("has_outline", True),
        )

    # Deserialize components
    for ref, c in data.get("components", {}).items():
        comp = Component(
            reference=c["reference"],
            footprint=c.get("footprint", ""),
            value=c.get("value", ""),
            x=c["x"],
            y=c["y"],
            width=c.get("width", 1.0),
            height=c.get("height", 1.0),
            rotation=c.get("rotation", 0.0),
            layer=c.get("layer", "F.Cu"),
            locked=c.get("locked", False),
        )

        # Deserialize pads
        comp.pads = []
        for p in c.get("pads", []):
            pad = Pad(
                number=p["number"],
                x=p["x"],
                y=p["y"],
                width=p.get("width", 0.5),
                height=p.get("height", 0.5),
                net=p.get("net"),
                shape=p.get("shape"),
            )
            pad.component_ref = ref
            comp.pads.append(pad)

        if c.get("fields"):
            comp.fields = c["fields"]

        board.components[ref] = comp

    # Deserialize nets
    for name, n in data.get("nets", {}).items():
        net = Net(
            name=n["name"],
            code=n.get("code", 0),
            is_power=n.get("is_power", False),
            is_ground=n.get("is_ground", False),
        )
        for comp_ref, pad_num in n.get("connections", []):
            net.add_connection(comp_ref, pad_num)
        board.nets[name] = net

    return board
