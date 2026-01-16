"""
RPC Client

Connects to the KiCad worker process.
"""

import subprocess
import json
import uuid
import threading
from pathlib import Path
from typing import Any, Dict
import os

from .protocol import RpcRequest, RpcResponse

class RpcClient:
    def __init__(self, kicad_python_path: str = "python3"):
        self.worker_script = Path(__file__).parent / "worker.py"
        self.process = subprocess.Popen(
            [kicad_python_path, str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        # Thread lock to prevent concurrent calls from interleaving
        self._lock = threading.Lock()

    def call(self, method: str, **params) -> Any:
        # Acquire lock to ensure request/response pairs aren't interleaved
        with self._lock:
            req = RpcRequest(id=str(uuid.uuid4()), method=method, params=params)

            if self.process.poll() is not None:
                raise RuntimeError("Worker process died")

            self.process.stdin.write(req.to_json() + "\n")
            self.process.stdin.flush()

            response_line = self.process.stdout.readline()
            if not response_line:
                stderr = self.process.stderr.read()
                raise RuntimeError(f"Worker returned empty response. Stderr: {stderr}")

            resp = RpcResponse.from_json(response_line)
            if resp.error:
                raise RuntimeError(f"RPC Error: {resp.error}")

            return resp.result

    def close(self):
        """Close the RPC client and cleanup resources."""
        # Explicitly close pipes to prevent file descriptor leaks
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()

        # Terminate the process
        self.process.terminate()

        # Wait for process to exit (with timeout to prevent hanging)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate gracefully
            self.process.kill()
            self.process.wait()
