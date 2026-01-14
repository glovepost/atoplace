"""
KiCad Python Detection Utility

Auto-detects KiCad's Python interpreter across platforms.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Standard KiCad installation paths by platform
KICAD_PYTHON_PATHS = {
    "darwin": [
        "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3",
        "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3",
    ],
    "linux": [
        "/usr/lib/kicad/bin/python3",
        "/usr/bin/python3",  # System Python with kicad-python package
    ],
    "win32": [
        r"C:\Program Files\KiCad\8.0\bin\python.exe",
        r"C:\Program Files\KiCad\7.0\bin\python.exe",
    ],
}


def find_kicad_python() -> Optional[str]:
    """
    Find KiCad's Python interpreter.

    Checks in order:
    1. KICAD_PYTHON environment variable
    2. Standard installation paths for current platform

    Returns:
        Path to KiCad Python interpreter, or None if not found.
    """
    # Check environment variable first
    env_path = os.environ.get("KICAD_PYTHON")
    if env_path and os.path.isfile(env_path):
        logger.debug("Using KICAD_PYTHON from environment: %s", env_path)
        return env_path

    # Try platform-specific paths
    platform = sys.platform
    paths = KICAD_PYTHON_PATHS.get(platform, [])

    for path in paths:
        if os.path.isfile(path):
            logger.debug("Found KiCad Python at: %s", path)
            return path

    return None


def validate_kicad_python(python_path: str, timeout: float = 10.0) -> Tuple[bool, str]:
    """
    Validate that a Python interpreter has pcbnew available.

    Args:
        python_path: Path to Python interpreter
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    if not os.path.isfile(python_path):
        return False, f"Python not found: {python_path}"

    try:
        result = subprocess.run(
            [python_path, "-c", "import pcbnew; print(pcbnew.Version())"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"KiCad {version}"
        else:
            error = result.stderr.strip() or "pcbnew import failed"
            return False, error

    except subprocess.TimeoutExpired:
        return False, "Timeout validating KiCad Python"
    except Exception as e:
        return False, str(e)


def get_kicad_python() -> str:
    """
    Get validated KiCad Python path.

    Returns:
        Path to KiCad Python interpreter.

    Raises:
        RuntimeError: If KiCad Python not found or invalid.
    """
    python_path = find_kicad_python()

    if not python_path:
        raise RuntimeError(
            "KiCad Python not found. Please either:\n"
            "  1. Install KiCad 8.0+ from https://kicad.org\n"
            "  2. Set KICAD_PYTHON environment variable to your KiCad Python path\n"
            "\n"
            "Expected locations:\n"
            f"  macOS: /Applications/KiCad/KiCad.app/.../python3\n"
            f"  Linux: /usr/lib/kicad/bin/python3\n"
            f"  Windows: C:\\Program Files\\KiCad\\8.0\\bin\\python.exe"
        )

    valid, message = validate_kicad_python(python_path)
    if not valid:
        raise RuntimeError(f"KiCad Python validation failed: {message}")

    logger.info("Using KiCad Python: %s (%s)", python_path, message)
    return python_path
