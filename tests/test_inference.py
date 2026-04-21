"""Tests for the inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference_new import (
    DEFAULT_CLASS_NAMES,
    ImageTooLargeError,
    PCBDefectPipeline,
    validate_image,
)


class TestValidateImage:
    def test_accepts_normal_image(self, sample_image: Image.Image) -> None:
        validate_image(sample_image)

    def test_rejects_too_small(self, small_image: Image.Image) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            validate_image(small_image)

    def test_rejects_too_large(self) -> None:
        huge = Image.fromarray(np.zeros((5000, 5000, 3), dtype=np.uint8))
        with pytest.raises(ImageTooLargeError, match="pixels"):
            validate_image(huge)


class TestGoldenDatabase:
    def test_empty_dir(self, mock_pipeline: PCBDefectPipeline) -> None:
        assert mock_pipeline.find_best_match(Image.new("RGB", (256, 256))) is None

    def test_loads_images_from_dir(
        self, tmp_path: Path, mock_pipeline: PCBDefectPipeline
    ) -> None:
        golden_dir = mock_pipeline._golden_dir
        golden_dir.mkdir(parents=True, exist_ok=True)

        for name in ("ref_01.png", "ref_02.png"):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(golden_dir / name)

        mock_pipeline._golden_db = None  # force rebuild
        assert len(mock_pipeline.golden_db) == 2

    def test_skips_non_image_files(self, mock_pipeline: PCBDefectPipeline) -> None:
        golden_dir = mock_pipeline._golden_dir
        golden_dir.mkdir(parents=True, exist_ok=True)
        (golden_dir / "readme.txt").write_text("not an image")

        mock_pipeline._golden_db = None
        assert len(mock_pipeline.golden_db) == 0


class TestClassifyBatch:
    def test_single_patch(
        self, mock_pipeline: PCBDefectPipeline, sample_image: Image.Image
    ) -> None:
        patch = sample_image.crop((0, 0, 128, 128))
        results = mock_pipeline._classify_batch([patch])
        assert len(results) == 1
        idx, conf = results[0]
        assert idx == 0
        assert conf > 0.9

    def test_batch_of_four(
        self, mock_pipeline: PCBDefectPipeline, sample_image: Image.Image
    ) -> None:
        patch = sample_image.crop((0, 0, 128, 128))
        results = mock_pipeline._classify_batch([patch] * 4)
        assert len(results) == 4
        assert all(idx == 0 for idx, _ in results)


class TestDetectAnomalies:
    def test_identical_images_no_detections(
        self, mock_pipeline: PCBDefectPipeline, sample_image: Image.Image
    ) -> None:
        detections = mock_pipeline.detect_anomalies(sample_image, sample_image.copy())
        assert detections == []

    def test_different_images_produce_detections(
        self, mock_pipeline: PCBDefectPipeline
    ) -> None:
        img_a = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        img_b = Image.fromarray(
            np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        )
        detections = mock_pipeline.detect_anomalies(img_a, img_b)
        assert len(detections) > 0
        assert all(d["label"] in DEFAULT_CLASS_NAMES for d in detections)

    def test_detection_boxes_within_image(
        self, mock_pipeline: PCBDefectPipeline
    ) -> None:
        img_a = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        img_b = Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8))
        detections = mock_pipeline.detect_anomalies(img_a, img_b)

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            assert 0 <= x1 < x2 <= 256
            assert 0 <= y1 < y2 <= 256


class TestDrawDetections:
    def test_no_detections_returns_copy(self, sample_image: Image.Image) -> None:
        result = PCBDefectPipeline.draw_detections(sample_image, [])
        assert result.size == sample_image.size
        assert result is not sample_image

    def test_draws_boxes(self, sample_image: Image.Image) -> None:
        detections = [
            {"box": [10, 10, 50, 50], "label": "spur", "confidence": 0.95},
            {"box": [100, 100, 150, 150], "label": "short", "confidence": 0.88},
        ]
        result = PCBDefectPipeline.draw_detections(sample_image, detections)
        assert np.array(result).sum() != np.array(sample_image).sum()


class TestPipelineRun:
    def test_run_returns_tuple(
        self, mock_pipeline: PCBDefectPipeline, sample_image: Image.Image
    ) -> None:
        # No golden DB -> returns input untouched.
        result_img, anomalies = mock_pipeline.run(sample_image)
        assert isinstance(result_img, Image.Image)
        assert anomalies == []

    def test_run_rejects_tiny_image(
        self, mock_pipeline: PCBDefectPipeline, small_image: Image.Image
    ) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            mock_pipeline.run(small_image)
