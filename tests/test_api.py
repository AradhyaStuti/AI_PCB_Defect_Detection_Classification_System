"""Tests for the FastAPI endpoints."""

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app


@pytest.fixture
def client():
    return TestClient(app)


def make_png(w=256, h=256):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_detect_smoke(client, mock_pipeline):
    with patch("api._get_pipeline", return_value=mock_pipeline):
        r = client.post(
            "/detect",
            files={"file": ("pcb.png", make_png(), "image/png")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["defect_count"] == len(body["detections"])
    assert body["inference_time_ms"] >= 0
    assert "timestamp" in body


def test_detect_without_file(client):
    assert client.post("/detect").status_code == 422


def test_detect_garbage_payload(client):
    # TODO: also cover JPEGs and PNGs with an alpha channel. Right now
    # we only exercise fully invalid bytes.
    r = client.post(
        "/detect",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 400


def test_detect_rejects_oversized(client):
    # MAX_UPLOAD_MB=0 means anything is too big.
    with patch("api.MAX_UPLOAD_MB", 0):
        r = client.post(
            "/detect",
            files={"file": ("img.png", make_png(), "image/png")},
        )
    assert r.status_code == 413


def test_detect_rejects_tiny_image(client, mock_pipeline):
    with patch("api._get_pipeline", return_value=mock_pipeline):
        r = client.post(
            "/detect",
            files={"file": ("tiny.png", make_png(32, 32), "image/png")},
        )
    assert r.status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
