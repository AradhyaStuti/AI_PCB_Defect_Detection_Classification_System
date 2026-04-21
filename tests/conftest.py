"""Shared pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from inference_new import DEFAULT_CLASS_NAMES, PCBDefectPipeline

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def sample_image() -> Image.Image:
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture()
def small_image() -> Image.Image:
    """An image smaller than the sliding window. Should fail validation."""
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.fixture()
def mock_pipeline(tmp_path: Path) -> PCBDefectPipeline:
    """Pipeline with a fake model that always predicts class 0."""
    pipeline = PCBDefectPipeline(
        model_path=tmp_path / "fake.pth",
        golden_dir=tmp_path / "golden",
    )

    num_classes = len(DEFAULT_CLASS_NAMES)
    logits = torch.zeros(1, num_classes)
    logits[0, 0] = 10.0

    def fake_forward(x: torch.Tensor) -> torch.Tensor:
        return logits.expand(x.shape[0], -1).clone()

    fake_model = MagicMock()
    fake_model.side_effect = fake_forward
    fake_model.__call__ = fake_forward

    pipeline._model = fake_model
    pipeline._class_names = DEFAULT_CLASS_NAMES.copy()
    pipeline._device = torch.device("cpu")

    return pipeline
