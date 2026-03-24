"""PCB differential defect detection pipeline.

Compares an input PCB image against a golden reference database using
structural similarity, then classifies anomalous regions with a
ResNet-50 classifier trained on DeepPCB defect categories.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TypedDict

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms
from torchvision.ops import nms

from config import GOLDEN_DIR as GOLDEN_IMAGES_DIR, MODEL_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class Detection(TypedDict):
    box: list[int]
    label: str
    confidence: float


class GoldenEntry(TypedDict):
    filename: str
    image: Image.Image
    hash: imagehash.ImageHash


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CLASS_NAMES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

WINDOW_SIZE = 128
STRIDE = WINDOW_SIZE // 4
SIMILARITY_THRESHOLD = 0.95
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.80
NMS_IOU_THRESHOLD = 0.2
BATCH_SIZE = 32
MAX_IMAGE_PIXELS = 4096 * 4096  # ~16 MP safety limit

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ImageTooLargeError(ValueError):
    """Raised when the input image exceeds the pixel budget."""


def validate_image(image: Image.Image) -> None:
    """Reject images that would cause excessive memory use."""
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS:
        raise ImageTooLargeError(
            f"Image is {w}x{h} ({w * h:,} pixels). "
            f"Maximum allowed is {MAX_IMAGE_PIXELS:,} pixels."
        )
    if w < WINDOW_SIZE or h < WINDOW_SIZE:
        raise ValueError(
            f"Image is {w}x{h}, smaller than the {WINDOW_SIZE}x{WINDOW_SIZE} "
            f"sliding window. Provide a larger image."
        )


# ---------------------------------------------------------------------------
# Pipeline (lazy-loaded singleton)
# ---------------------------------------------------------------------------

class PCBDefectPipeline:
    """Encapsulates model, golden DB, and inference logic.

    The heavy resources (model weights, golden image database) are loaded
    lazily on first use so that importing this module is side-effect-free.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        golden_dir: Path = GOLDEN_IMAGES_DIR,
    ) -> None:
        self._model_path = model_path
        self._golden_dir = golden_dir

        self._device: torch.device | None = None
        self._model: torch.nn.Module | None = None
        self._class_names: list[str] | None = None
        self._golden_db: list[GoldenEntry] | None = None

    # -- Lazy properties -----------------------------------------------------

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Using device: %s", self._device)
        return self._device

    @property
    def class_names(self) -> list[str]:
        if self._class_names is None:
            self._load_model()
        assert self._class_names is not None
        return self._class_names

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._load_model()
        assert self._model is not None
        return self._model

    @property
    def golden_db(self) -> list[GoldenEntry]:
        if self._golden_db is None:
            self._golden_db = self._build_golden_database()
        return self._golden_db

    # -- Initialization helpers ----------------------------------------------

    def _load_model(self) -> None:
        logger.info("Loading model from %s", self._model_path)
        checkpoint = torch.load(
            self._model_path, map_location=self.device, weights_only=False,
        )

        class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES.copy())
        if "normal" in class_names:
            class_names.remove("normal")
        self._class_names = class_names

        num_classes = len(self._class_names)
        logger.info("Classes (%d): %s", num_classes, self._class_names)

        classifier = models.resnet50(weights=None)
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, num_classes)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.to(self.device)
        classifier.eval()
        self._model = classifier

    def _build_golden_database(self) -> list[GoldenEntry]:
        db: list[GoldenEntry] = []
        if not self._golden_dir.is_dir():
            logger.warning("Golden image directory not found: %s", self._golden_dir)
            return db

        for path in sorted(self._golden_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = Image.open(path).convert("RGB")
                db.append({
                    "filename": path.name,
                    "image": img,
                    "hash": imagehash.phash(img),
                })
            except OSError:
                logger.warning("Could not load image: %s", path)

        logger.info("Loaded %d golden reference images", len(db))
        return db

    # -- Core inference ------------------------------------------------------

    def find_best_match(self, input_image: Image.Image) -> Image.Image | None:
        if not self.golden_db:
            return None
        input_hash = imagehash.phash(input_image)
        best = min(self.golden_db, key=lambda entry: input_hash - entry["hash"])
        return best["image"]

    def _classify_batch(
        self,
        patches: list[Image.Image],
    ) -> list[tuple[int, float]]:
        """Classify a batch of patches in a single forward pass."""
        tensors = torch.stack([INFERENCE_TRANSFORM(p) for p in patches]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors)
            probs = F.softmax(logits, dim=1)
        confidences, indices = probs.max(dim=1)
        return list(zip(indices.tolist(), confidences.tolist(), strict=True))

    def detect_anomalies(
        self,
        input_image: Image.Image,
        golden_image: Image.Image,
    ) -> list[Detection]:
        if input_image.size != golden_image.size:
            golden_image = golden_image.resize(input_image.size)

        img_w, img_h = input_image.size

        # Phase 1: collect anomalous patches via SSIM
        candidate_patches: list[Image.Image] = []
        candidate_boxes: list[list[int]] = []

        for y in range(0, img_h - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, img_w - WINDOW_SIZE + 1, STRIDE):
                crop_box = (x, y, x + WINDOW_SIZE, y + WINDOW_SIZE)
                patch_input = input_image.crop(crop_box)
                patch_golden = golden_image.crop(crop_box)

                score = ssim(
                    np.asarray(patch_golden.convert("L")),
                    np.asarray(patch_input.convert("L")),
                )
                if score < SIMILARITY_THRESHOLD:
                    candidate_patches.append(patch_input)
                    candidate_boxes.append([x, y, x + WINDOW_SIZE, y + WINDOW_SIZE])

        if not candidate_patches:
            return []

        # Phase 2: batch-classify all candidate patches
        detections: list[Detection] = []

        for batch_start in range(0, len(candidate_patches), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(candidate_patches))
            batch_patches = candidate_patches[batch_start:batch_end]
            batch_boxes = candidate_boxes[batch_start:batch_end]

            results = self._classify_batch(batch_patches)

            for (pred_idx, confidence), box in zip(results, batch_boxes, strict=True):
                if confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                    detections.append({
                        "box": box,
                        "label": self.class_names[pred_idx],
                        "confidence": confidence,
                    })

        if not detections:
            return []

        # Phase 3: NMS
        boxes_t = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
        scores_t = torch.tensor([d["confidence"] for d in detections], dtype=torch.float32)
        keep = nms(boxes_t, scores_t, iou_threshold=NMS_IOU_THRESHOLD)
        return [detections[i] for i in keep.tolist()]

    # -- Visualization -------------------------------------------------------

    @staticmethod
    def draw_detections(
        image: Image.Image,
        detections: list[Detection],
    ) -> Image.Image:
        if not detections:
            return image.copy()

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 32)
        except OSError:
            font = ImageFont.load_default()

        unique_labels = sorted({d["label"] for d in detections})
        cmap = plt.colormaps["hsv"]
        color_map = {
            label: tuple(int(c * 255) for c in cmap(i / max(len(unique_labels), 1))[:3])
            for i, label in enumerate(unique_labels)
        }

        for det in detections:
            box = det["box"]
            color = color_map[det["label"]]
            label_text = f"{det['label']} ({det['confidence']:.2f})"

            draw.rectangle(box, outline=color, width=5)

            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]

            bg_box = [box[0], box[1] - th - 5, box[0] + tw + 10, box[1]]
            draw.rectangle(bg_box, fill=color)
            draw.text((box[0] + 5, box[1] - th - 5), label_text, fill="white", font=font)

        return canvas

    # -- Public entry point --------------------------------------------------

    def run(self, input_image: Image.Image) -> tuple[Image.Image, list[Detection]]:
        """Run the full detection pipeline on a single PCB image.

        Returns:
            A tuple of (annotated image, list of detections).

        Raises:
            ImageTooLargeError: If the image exceeds MAX_IMAGE_PIXELS.
            ValueError: If the image is smaller than the sliding window.
        """
        validate_image(input_image)
        t0 = time.perf_counter()

        golden_ref = self.find_best_match(input_image)
        if golden_ref is None:
            logger.warning("No golden reference found — returning input unchanged")
            return input_image, []

        anomalies = self.detect_anomalies(input_image, golden_ref)
        result_image = self.draw_detections(input_image, anomalies)

        elapsed = time.perf_counter() - t0
        logger.info("Detection complete: %d defects in %.2fs", len(anomalies), elapsed)

        return result_image, anomalies


# ---------------------------------------------------------------------------
# Module-level singleton & backward-compatible API
# ---------------------------------------------------------------------------

_pipeline: PCBDefectPipeline | None = None


def _get_pipeline() -> PCBDefectPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PCBDefectPipeline()
    return _pipeline


def run_inference_on_pil(
    input_image: Image.Image,
) -> tuple[Image.Image, list[Detection]]:
    """Backward-compatible entry point used by app.py."""
    return _get_pipeline().run(input_image)
