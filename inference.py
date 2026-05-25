"""PCB defect detection pipeline.

SSIM-diffs an input against the closest golden reference, then runs a
ResNet-50 classifier (trained on DeepPCB) over the patches that look off.
"""

import colorsys
import logging
import time
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms
from torchvision.ops import nms

from config import GOLDEN_DIR, MODEL_PATH

logger = logging.getLogger(__name__)


DEFAULT_CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# Thresholds tuned on the DeepPCB validation set. Re-check precision/recall
# before changing them.
WINDOW_SIZE = 128
# 75% overlap catches defects that straddle window edges.
STRIDE = WINDOW_SIZE // 4
# SSIM below this = patch is suspect; biased toward over-reporting.
SIMILARITY_THRESHOLD = 0.95
# Most real defects classify > 0.9; this trims the noise.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.80
# Tight: PCB defects are small, don't want one defect counted twice.
NMS_IOU_THRESHOLD = 0.2
# Fits an 8 GB GPU; CPU runs are fine too.
BATCH_SIZE = 32

# Cap at ~16 MP so a huge upload can't blow up memory. Real PCB scans rarely exceed this.
MAX_IMAGE_PIXELS = 4096 * 4096

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageTooLargeError(ValueError):
    pass


def validate_image(image: Image.Image) -> None:
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS:
        raise ImageTooLargeError(
            f"Image is {w}x{h} ({w * h:,} pixels), max is {MAX_IMAGE_PIXELS:,}."
        )
    if w < WINDOW_SIZE or h < WINDOW_SIZE:
        raise ValueError(f"Image is {w}x{h}, smaller than the {WINDOW_SIZE}x{WINDOW_SIZE} window.")


class PCBDefectPipeline:
    """End-to-end detect-and-classify on a single image.

    Model weights and golden references load on first use, so importing
    this module stays cheap.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        golden_dir: Path = GOLDEN_DIR,
    ) -> None:
        self.model_path = model_path
        self.golden_dir = golden_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: torch.nn.Module | None = None
        self.class_names: list[str] | None = None
        self._golden_db: list[dict[str, Any]] | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return

        logger.info("Loading model from %s", self.model_path)
        # weights_only=False because the checkpoint also carries the class_names list.
        # Safe given that we control where these .pth files come from.
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        names = ckpt.get("class_names", DEFAULT_CLASS_NAMES.copy())
        # Some training checkpoints carry a "normal" class; we don't predict that at inference.
        if "normal" in names:
            names.remove("normal")

        net = models.resnet50(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(names))
        net.load_state_dict(ckpt["model_state_dict"])
        net.to(self.device)
        net.eval()

        self.model = net
        self.class_names = names
        logger.info("Loaded %d classes: %s", len(names), names)

    @property
    def golden_db(self) -> list[dict[str, Any]]:
        if self._golden_db is None:
            self._golden_db = self._build_golden_database()
        return self._golden_db

    def _build_golden_database(self) -> list[dict[str, Any]]:
        if not self.golden_dir.is_dir():
            logger.warning("Golden image directory not found: %s", self.golden_dir)
            return []

        db: list[dict[str, Any]] = []
        for path in sorted(self.golden_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                logger.warning("Could not load image: %s", path)
                continue
            db.append(
                {
                    "image": img,
                    "hash": imagehash.phash(img),
                }
            )

        logger.info("Loaded %d golden reference images", len(db))
        return db

    def find_best_match(self, input_image: Image.Image) -> Image.Image | None:
        if not self.golden_db:
            return None
        h = imagehash.phash(input_image)
        return min(self.golden_db, key=lambda e: h - e["hash"])["image"]

    def _classify_batch(self, patches: list[Image.Image]) -> list[tuple[int, float]]:
        self._ensure_loaded()
        assert self.model is not None  # _ensure_loaded sets this
        x = torch.stack([INFERENCE_TRANSFORM(p) for p in patches]).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)
        confs, idxs = probs.max(dim=1)
        return list(zip(idxs.tolist(), confs.tolist(), strict=True))

    def detect_anomalies(
        self,
        input_image: Image.Image,
        golden_image: Image.Image,
    ) -> list[dict[str, Any]]:
        """Returns dicts of {box: [x1, y1, x2, y2], label: str, confidence: float}."""
        if input_image.size != golden_image.size:
            golden_image = golden_image.resize(input_image.size)

        img_w, img_h = input_image.size

        # TODO: this scan is sequential and dominates wall time on big boards.
        # Vectorize the SSIM pass (or move it to skimage's batched API) before profiling.
        patches: list[Image.Image] = []
        boxes: list[list[int]] = []
        for y in range(0, img_h - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, img_w - WINDOW_SIZE + 1, STRIDE):
                crop = (x, y, x + WINDOW_SIZE, y + WINDOW_SIZE)
                patch_in = input_image.crop(crop)
                patch_gold = golden_image.crop(crop)

                score = ssim(
                    np.asarray(patch_gold.convert("L")),
                    np.asarray(patch_in.convert("L")),
                )
                if score < SIMILARITY_THRESHOLD:
                    patches.append(patch_in)
                    boxes.append([x, y, x + WINDOW_SIZE, y + WINDOW_SIZE])

        if not patches:
            return []

        detections: list[dict[str, Any]] = []
        for start in range(0, len(patches), BATCH_SIZE):
            end = start + BATCH_SIZE
            results = self._classify_batch(patches[start:end])
            assert self.class_names is not None  # set inside _classify_batch
            for (idx, conf), box in zip(results, boxes[start:end], strict=True):
                if conf >= CLASSIFIER_CONFIDENCE_THRESHOLD:
                    detections.append(
                        {
                            "box": box,
                            "label": self.class_names[idx],
                            "confidence": conf,
                        }
                    )

        if not detections:
            return []

        boxes_t = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
        scores_t = torch.tensor([d["confidence"] for d in detections], dtype=torch.float32)
        keep = nms(boxes_t, scores_t, iou_threshold=NMS_IOU_THRESHOLD)
        return [detections[i] for i in keep.tolist()]

    @staticmethod
    def draw_detections(
        image: Image.Image,
        detections: list[dict[str, Any]],
    ) -> Image.Image:
        if not detections:
            return image.copy()

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 32)
        except OSError:
            font = ImageFont.load_default()

        # Evenly-spaced HSV hues, one per label. Saturation/value pinned at 1
        # so labels are bright and distinguishable on dark PCB backgrounds.
        labels = sorted({d["label"] for d in detections})
        n = max(len(labels), 1)
        colors = {
            label: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 1.0, 1.0))
            for i, label in enumerate(labels)
        }

        for d in detections:
            x1, y1, _, _ = d["box"]
            color = colors[d["label"]]
            text = f"{d['label']} ({d['confidence']:.2f})"

            draw.rectangle(d["box"], outline=color, width=5)

            tb = draw.textbbox((0, 0), text, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 10, y1], fill=color)
            draw.text((x1 + 5, y1 - text_h - 5), text, fill="white", font=font)

        return canvas

    def run(
        self,
        input_image: Image.Image,
    ) -> tuple[Image.Image, list[dict[str, Any]]]:
        validate_image(input_image)
        t0 = time.perf_counter()

        ref = self.find_best_match(input_image)
        if ref is None:
            logger.warning("No golden reference available, returning input unchanged.")
            return input_image, []

        anomalies = self.detect_anomalies(input_image, ref)
        out = self.draw_detections(input_image, anomalies)

        logger.info(
            "Detection done: %d defects in %.2fs",
            len(anomalies),
            time.perf_counter() - t0,
        )
        return out, anomalies
