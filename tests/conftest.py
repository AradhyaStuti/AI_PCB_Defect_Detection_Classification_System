"""Shared pytest fixtures."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from inference import DEFAULT_CLASS_NAMES, PCBDefectPipeline


@pytest.fixture
def sample_image():
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def small_image():
    """Smaller than the sliding window, should fail validation."""
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.fixture
def mock_pipeline(tmp_path):
    """Pipeline whose 'model' always predicts class 0 with high confidence,
    so we can exercise the surrounding logic without real weights.
    """
    pipeline = PCBDefectPipeline(
        model_path=tmp_path / "fake.pth",
        golden_dir=tmp_path / "golden",
    )

    logits = torch.zeros(1, len(DEFAULT_CLASS_NAMES))
    logits[0, 0] = 10.0

    def fake_forward(x):
        return logits.expand(x.shape[0], -1).clone()

    pipeline.model = MagicMock(side_effect=fake_forward)
    pipeline.class_names = DEFAULT_CLASS_NAMES.copy()
    pipeline.device = torch.device("cpu")
    return pipeline
