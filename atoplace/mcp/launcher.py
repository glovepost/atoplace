#!/usr/bin/env python3
"""
AtoPlace MCP Launcher

Single entry point that handles all the complexity of running the MCP server
with KiCad bridge. Just run: python -m atoplace.mcp.launcher

The launcher:
1. Auto-detects KiCad Python (no env vars needed in most cases)
2. Starts the KiCad bridge process
3. Starts the MCP server
4. Manages lifecycle and clean shutdown
"""

import os
import sys
import subprocess
import signal
import time
import atexit
import logging
from pathlib import Path
from typing import Optional

# Socket path - consistent location
SOCKET_PATH = "/tmp/atoplace-bridge.sock"

# Configure logging to file (keeps STDIO clean for MCP protocol)
LOG_FILE = os.environ.get("ATOPLACE_LOG", "/tmp/atoplace.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a")],
)
logger = logging.getLogger("atoplace.launcher")


class MCPLauncher:
    """Manages the MCP server and KiCad bridge processes."""

    def __init__(self):
        self.bridge_proc: Optional[subprocess.Popen] = None
        self.mcp_proc: Optional[subprocess.Popen] = None
        self.kicad_python: Optional[str] = None
        self.project_root = str(Path(__file__).parent.parent.parent)
        self._shutdown_called = False

    def start(self) -> int:
        """
        Start the MCP server with KiCad bridge.

        Returns:
            Exit code (0 for success)
        """
        try:
            # Step 1: Find and validate KiCad Python
            self._setup_kicad_python()

            # Step 2: Clean up any stale socket
            self._cleanup_socket()

            # Step 3: Start bridge process
            self._start_bridge()

            # Step 4: Wait for bridge to be ready
            if not self._wait_for_bridge():
                logger.error("Bridge failed to start")
                self.shutdown()
                return 1

            # Step 5: Start MCP server (takes over STDIO)
            return self._run_mcp_server()

        except Exception as e:
            logger.exception("Launcher error: %s", e)
            self.shutdown()
            return 1

    def _setup_kicad_python(self):
        """Find and validate KiCad Python."""
        from .kicad import get_kicad_python

        self.kicad_python = get_kicad_python()
        logger.info("KiCad Python: %s", self.kicad_python)

    def _cleanup_socket(self):
        """Remove stale socket file."""
        if os.path.exists(SOCKET_PATH):
            try:
                os.unlink(SOCKET_PATH)
                logger.debug("Removed stale socket: %s", SOCKET_PATH)
            except OSError:
                pass

    def _start_bridge(self):
        """Start the KiCad bridge process."""
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.project_root}:{pythonpath}" if pythonpath else self.project_root
        )

        cmd = [
            self.kicad_python,
            "-m",
            "atoplace.mcp.bridge",
            "--socket",
            SOCKET_PATH,
        ]

        logger.info("Starting bridge: %s", " ".join(cmd))

        self.bridge_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

        logger.info("Bridge started (PID: %d)", self.bridge_proc.pid)

    def _wait_for_bridge(self, timeout: float = 10.0) -> bool:
        """Wait for bridge socket to be ready."""
        start = time.time()

        while time.time() - start < timeout:
            # Check if bridge died
            if self.bridge_proc.poll() is not None:
                output = self.bridge_proc.stdout.read()
                logger.error("Bridge exited unexpectedly:\n%s", output)
                return False

            # Check if socket exists
            if os.path.exists(SOCKET_PATH):
                logger.info("Bridge ready (%.1fs)", time.time() - start)
                return True

            time.sleep(0.1)

        logger.error("Bridge socket not ready after %.1fs", timeout)
        return False

    def _run_mcp_server(self) -> int:
        """Run the MCP server (blocks until completion)."""
        env = os.environ.copy()
        env["ATOPLACE_USE_IPC"] = "1"
        env["ATOPLACE_IPC_SOCKET"] = SOCKET_PATH

        cmd = [
            sys.executable,
            "-m",
            "atoplace.mcp.server",
        ]

        logger.info("Starting MCP server")

        # MCP server takes over STDIO
        self.mcp_proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Wait for MCP server to exit
        return self.mcp_proc.wait()

    def shutdown(self):
        """Clean shutdown of all processes."""
        if self._shutdown_called:
            return
        self._shutdown_called = True

        logger.info("Shutting down...")

        # Stop MCP server
        if self.mcp_proc and self.mcp_proc.poll() is None:
            self.mcp_proc.terminate()
            try:
                self.mcp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mcp_proc.kill()

        # Stop bridge
        if self.bridge_proc and self.bridge_proc.poll() is None:
            self.bridge_proc.terminate()
            try:
                self.bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bridge_proc.kill()

        # Clean up socket
        self._cleanup_socket()

        logger.info("Shutdown complete")


def main():
    """Entry point for the MCP launcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AtoPlace MCP Server - AI-powered PCB layout tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m atoplace.mcp.launcher        # Start MCP server (default)
  atoplace-mcp                           # Same, using entry point

The server auto-detects KiCad Python. Override with KICAD_PYTHON env var.
Logs are written to /tmp/atoplace.log
        """,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        print("atoplace-mcp 0.1.0")
        return 0

    # Create launcher and register cleanup
    launcher = MCPLauncher()
    atexit.register(launcher.shutdown)

    # Handle signals
    def signal_handler(signum, frame):
        launcher.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    return launcher.start()


if __name__ == "__main__":
    sys.exit(main())
