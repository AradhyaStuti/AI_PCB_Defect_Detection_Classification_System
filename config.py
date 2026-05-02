"""Paths, API, and logging config. Everything overridable via env vars.

Env vars are prefixed with PCB_ so they don't collide with anything else
the host might already be using. python-dotenv is optional: handy in
dev with a local .env, not needed in prod where the host sets them directly.
"""

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

MODEL_PATH = Path(
    os.environ.get(
        "PCB_MODEL_PATH",
        str(_BASE_DIR / "model" / "best_resnet50_pcb_defects_50epochs.pth"),
    )
)
GOLDEN_DIR = Path(os.environ.get("PCB_GOLDEN_DIR", str(_BASE_DIR / "PCB_USED")))
LOG_DIR = Path(os.environ.get("PCB_LOG_DIR", str(_BASE_DIR / "logs")))

API_HOST = os.environ.get("PCB_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PCB_API_PORT", "8000"))
MAX_UPLOAD_MB = int(os.environ.get("PCB_MAX_UPLOAD_MB", "10"))

LOG_LEVEL = os.environ.get("PCB_LOG_LEVEL", "INFO")


def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        # Already configured by another module; leave it alone.
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    rotating = logging.handlers.RotatingFileHandler(
        LOG_DIR / "pcb_detection.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    rotating.setFormatter(fmt)

    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(rotating)
