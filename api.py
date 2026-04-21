"""FastAPI layer around the PCB defect detection pipeline.

Run with:
    uvicorn api:app --reload
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from config import API_HOST, API_PORT, MAX_UPLOAD_MB, configure_logging
from inference_new import ImageTooLargeError, PCBDefectPipeline

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PCB Defect Detection API",
    description="Upload a PCB image, get back defect boxes and labels.",
    version="1.0.0",
)

_pipeline: PCBDefectPipeline | None = None


def _get_pipeline() -> PCBDefectPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PCBDefectPipeline()
    return _pipeline


class DetectionResult(BaseModel):
    label: str
    confidence: float
    box: list[int]


class InferenceResponse(BaseModel):
    defect_count: int
    inference_time_ms: float
    detections: list[DetectionResult]
    timestamp: str


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@app.post("/detect", response_model=InferenceResponse, tags=["inference"])
async def detect(file: UploadFile = File(...)) -> InferenceResponse:
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB, max is {MAX_UPLOAD_MB} MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc

    t0 = time.perf_counter()
    try:
        _, detections = _get_pipeline().run(image)
    except (ImageTooLargeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed for file: %s", file.filename)
        raise HTTPException(status_code=500, detail="Inference error, see server logs.") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Detected %d defects in %.1f ms (file=%s)", len(detections), elapsed_ms, file.filename
    )

    return InferenceResponse(
        defect_count=len(detections),
        inference_time_ms=round(elapsed_ms, 2),
        detections=[
            DetectionResult(
                label=d["label"],
                confidence=round(d["confidence"], 4),
                box=d["box"],
            )
            for d in detections
        ],
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
