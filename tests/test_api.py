"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app
from inference_new import PCBDefectPipeline


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _make_png(width: int = 256, height: int = 256) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_body_has_status_ok(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_body_has_timestamp(self, client: TestClient) -> None:
        assert "timestamp" in client.get("/health").json()


class TestDetectEndpoint:
    def test_missing_file_returns_422(self, client: TestClient) -> None:
        assert client.post("/detect").status_code == 422

    def test_oversized_file_returns_413(self, client: TestClient) -> None:
        with patch("api.MAX_UPLOAD_MB", 0):
            response = client.post(
                "/detect",
                files={"file": ("img.png", _make_png(), "image/png")},
            )
        assert response.status_code == 413

    def test_invalid_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/detect",
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
        assert response.status_code == 400

    def test_small_image_returns_422(
        self, client: TestClient, mock_pipeline: PCBDefectPipeline
    ) -> None:
        with patch("api._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/detect",
                files={"file": ("small.png", _make_png(32, 32), "image/png")},
            )
        assert response.status_code == 422

    def test_valid_image_returns_inference_response(
        self, client: TestClient, mock_pipeline: PCBDefectPipeline
    ) -> None:
        with patch("api._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/detect",
                files={"file": ("pcb.png", _make_png(), "image/png")},
            )
        assert response.status_code == 200
        body = response.json()
        assert "defect_count" in body
        assert "inference_time_ms" in body
        assert "detections" in body
        assert "timestamp" in body
        assert isinstance(body["inference_time_ms"], float)
        assert body["inference_time_ms"] >= 0
        assert body["defect_count"] == len(body["detections"])

    def test_defect_count_matches_detections_list(
        self, client: TestClient, mock_pipeline: PCBDefectPipeline
    ) -> None:
        with patch("api._get_pipeline", return_value=mock_pipeline):
            body = client.post(
                "/detect",
                files={"file": ("pcb.png", _make_png(), "image/png")},
            ).json()
        assert body["defect_count"] == len(body["detections"])
