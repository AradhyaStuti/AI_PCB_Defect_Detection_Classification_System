"""Config for paths, API, and logging. Everything is overridable via env vars.

If python-dotenv is installed, values are also loaded from a local .env file.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH: Path = Path(
    os.environ.get(
        "PCB_MODEL_PATH",
        str(_BASE_DIR / "model" / "best_resnet50_pcb_defects_50epochs.pth"),
    )
)

GOLDEN_DIR: Path = Path(
    os.environ.get("PCB_GOLDEN_DIR", str(_BASE_DIR / "PCB_USED"))
)

LOG_DIR: Path = Path(os.environ.get("PCB_LOG_DIR", str(_BASE_DIR / "logs")))

API_HOST: str = os.environ.get("PCB_API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("PCB_API_PORT", "8000"))
MAX_UPLOAD_MB: int = int(os.environ.get("PCB_MAX_UPLOAD_MB", "10"))

LOG_LEVEL: str = os.environ.get("PCB_LOG_LEVEL", "INFO")


def configure_logging() -> None:
    root = logging.getLogger()
    # No-op if logging was already set up (e.g. another module called us first).
    if root.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "pcb_detection.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(fmt)

    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
