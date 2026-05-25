"""Tests for the inference pipeline."""

import numpy as np
import pytest
from PIL import Image

from inference import (
    DEFAULT_CLASS_NAMES,
    ImageTooLargeError,
    PCBDefectPipeline,
    validate_image,
)


def test_run_with_no_golden_refs_returns_input(mock_pipeline, sample_image):
    out, anomalies = mock_pipeline.run(sample_image)
    assert anomalies == []
    assert out.tobytes() == sample_image.tobytes()


def test_run_rejects_tiny(mock_pipeline, small_image):
    with pytest.raises(ValueError, match="smaller than"):
        mock_pipeline.run(small_image)


@pytest.mark.parametrize(
    "size, exc, msg",
    [
        ((64, 64), ValueError, "smaller than"),
        ((5000, 5000), ImageTooLargeError, "pixels"),
    ],
)
def test_validate_image_rejects(size, exc, msg):
    img = Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    with pytest.raises(exc, match=msg):
        validate_image(img)


def test_validate_image_accepts_normal(sample_image):
    validate_image(sample_image)


def test_identical_images_produce_no_detections(mock_pipeline, sample_image):
    assert mock_pipeline.detect_anomalies(sample_image, sample_image.copy()) == []


def test_different_images_produce_detections(mock_pipeline):
    black = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    noisy = Image.fromarray(np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8))

    detections = mock_pipeline.detect_anomalies(black, noisy)
    assert detections
    assert all(d["label"] in DEFAULT_CLASS_NAMES for d in detections)


def test_detection_boxes_stay_within_bounds(mock_pipeline):
    a = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    b = Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8))

    for d in mock_pipeline.detect_anomalies(a, b):
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= 256
        assert 0 <= y1 < y2 <= 256


def test_find_best_match_returns_none_when_no_refs(mock_pipeline):
    assert mock_pipeline.find_best_match(Image.new("RGB", (256, 256))) is None


def test_golden_db_skips_non_image_files(mock_pipeline):
    golden_dir = mock_pipeline.golden_dir
    golden_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ref_01.png", "ref_02.png"):
        Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(
            golden_dir / name
        )
    (golden_dir / "readme.txt").write_text("not an image")

    # Reset the lazy cache so it re-scans the now-populated dir.
    mock_pipeline._golden_db = None
    assert len(mock_pipeline.golden_db) == 2


def test_draw_detections_changes_pixels(sample_image):
    detections = [
        {"box": [10, 10, 50, 50], "label": "spur", "confidence": 0.95},
        {"box": [100, 100, 150, 150], "label": "short", "confidence": 0.88},
    ]
    out = PCBDefectPipeline.draw_detections(sample_image, detections)
    assert not np.array_equal(np.array(out), np.array(sample_image))


def test_draw_detections_empty_returns_copy(sample_image):
    out = PCBDefectPipeline.draw_detections(sample_image, [])
    assert out is not sample_image
    assert out.size == sample_image.size
