<p align="center">
  <img src="image/README/1768377780977.png" alt="PCB Defect Detection Banner" width="800"/>
</p>

<h1 align="center">AI PCB Defect Detection & Classification System</h1>

<p align="center">
  <strong>Automated defect detection and classification for Printed Circuit Boards using deep learning and image processing.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Model-ResNet50-green" alt="ResNet50"/>
  <img src="https://img.shields.io/badge/Dataset-DeepPCB-orange" alt="DeepPCB"/>
</p>

---

## Overview

An end-to-end system that detects and classifies defects on PCB boards by comparing test images against defect-free golden references. It uses SSIM-based anomaly detection, ResNet-50 classification, and provides both a Streamlit web UI and a FastAPI REST API.

---

## Features

- **SSIM sliding window** detection (128x128, stride 32, threshold 0.95)
- **pHash matching** for automatic golden reference selection
- **ResNet-50** classification trained on DeepPCB (6 defect classes, 50 epochs)
- **Batch inference** — 32 patches per forward pass for GPU efficiency
- **NMS post-processing** (IoU 0.2) to eliminate overlapping detections
- **Streamlit UI** — upload, detect, view annotated results, download outputs
- **FastAPI REST API** — `/health` and `/detect` endpoints with JSON responses
- **Docker + Docker Compose** — containerised deployment with health checks
- **Environment config** — all settings overridable via `PCB_*` env vars
- **Rotating logs** — 10 MB x 5 files, plus console output
- **24 unit tests** with CI/CD via GitHub Actions (lint, typecheck, test, docker)
- **Type-safe** — full type hints, mypy, ruff

---

## How It Works

```
Input Image ──> pHash Match ──> SSIM Comparison ──> Batch CNN Classification ──> NMS ──> Annotated Output
                    │                  │                       │
              Golden DB          Threshold=0.95         Confidence>0.80
```

1. **pHash Matching** — finds the closest golden reference image from the database
2. **SSIM Sliding Window** — compares 128x128 patches; SSIM < 0.95 = anomalous
3. **Batch Classification** — suspicious patches classified by ResNet-50 in batches of 32
4. **NMS** — removes overlapping detections (IoU threshold 0.2)

---

## Dataset & Defect Classes

**DeepPCB Dataset:** [Download from Dropbox](https://www.dropbox.com/scl/fi/4vrtqn7t001yl41oucflu/PCB_DATASET.zip?rlkey=pghz15q2bsg205wynjwsj2c3n&e=2&dl=0)

| Defect | Description |
|---|---|
| **Missing Hole** | A drill hole that should exist but doesn't |
| **Mouse Bite** | Irregular, jagged edges along copper traces |
| **Open Circuit** | A break or gap in a copper trace |
| **Short** | Unintended copper bridging two traces |
| **Spur** | Small unwanted copper protrusion from a trace |
| **Spurious Copper** | Random copper deposits where there shouldn't be any |

**Sample Results:**

| Missing Hole | Spurious Copper |
|---|---|
| ![Missing Hole](image/README/1768378765792.png) | ![Spurious Copper](image/README/1768378792341.png) |

| Spur | Short |
|---|---|
| ![Spur](image/README/1768378926200.png) | ![Short](image/README/1768379000627.png) |

| Open Circuit | Mouse Bite |
|---|---|
| ![Open Circuit](image/README/1768379036675.png) | ![Mouse Bite](image/README/1768379058640.png) |

**Inference Result:**

![Inference Result](image/README/1768378725086.png)

---

## Tech Stack

| Area | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision (ResNet-50) |
| Image Processing | scikit-image (SSIM), ImageHash (pHash), Pillow, NumPy |
| Frontend | Streamlit |
| REST API | FastAPI, uvicorn, Pydantic |
| Containerisation | Docker, Docker Compose |
| Testing | pytest (24 tests), httpx |
| CI/CD | GitHub Actions (lint + typecheck + test + docker) |
| Code Quality | ruff, mypy |

---

## Project Structure

```
├── app.py                  # Streamlit web application
├── api.py                  # FastAPI REST API (/health, /detect)
├── inference_new.py        # Detection pipeline (PCBDefectPipeline class)
├── config.py               # Centralised env-var config
├── requirements.txt        # Python dependencies
├── pyproject.toml          # ruff, mypy, pytest config
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── model/
│   └── best_resnet50_pcb_defects_50epochs.pth
├── PCB_USED/               # Golden reference images
├── logs/                   # Rotating log files
├── tests/
│   ├── conftest.py         # Fixtures & mock model
│   ├── test_inference.py   # 15 pipeline tests
│   └── test_api.py         # 9 API tests
└── .github/workflows/
    └── ci.yml              # CI pipeline
```

---

## Quick Start

### Local

```bash
git clone https://github.com/AradhyaStuti/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System
python -m venv my_virtual_env
my_virtual_env\Scripts\activate        # Windows
# source my_virtual_env/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Place `best_resnet50_pcb_defects_50epochs.pth` inside `model/`.

**Run Streamlit UI:**
```bash
streamlit run app.py
# Open http://localhost:8501
```

**Run REST API:**
```bash
uvicorn api:app --reload
# Docs at http://localhost:8000/docs
```

### Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| REST API | `http://localhost:8000` |
| Streamlit UI | `http://localhost:8501` |
| Swagger Docs | `http://localhost:8000/docs` |

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Detect defects
curl -X POST http://localhost:8000/detect -F "file=@your_pcb_image.jpg"
```

**Response:**
```json
{
  "defect_count": 2,
  "inference_time_ms": 312.5,
  "detections": [
    {"label": "spur", "confidence": 0.9412, "box": [64, 128, 192, 256]},
    {"label": "open_circuit", "confidence": 0.8871, "box": [320, 64, 448, 192]}
  ],
  "timestamp": "2026-03-24T10:00:01+00:00"
}
```

---

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

24 tests (15 pipeline + 9 API). No model file needed — tests use mocks.

---

## Configuration

All settings overridable via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `PCB_MODEL_PATH` | `model/best_resnet50_pcb_defects_50epochs.pth` | Model weights path |
| `PCB_GOLDEN_DIR` | `PCB_USED/` | Golden reference images |
| `PCB_LOG_DIR` | `logs/` | Log file directory |
| `PCB_API_HOST` | `0.0.0.0` | API bind address |
| `PCB_API_PORT` | `8000` | API port |
| `PCB_MAX_UPLOAD_MB` | `10` | Max upload size (MB) |
| `PCB_LOG_LEVEL` | `INFO` | Log level |

---

## What I Did in My Internship

Built an AI-based PCB defect detection system from scratch — dataset preparation to deployment:

- Studied the DeepPCB dataset and the 6 defect categories
- Preprocessed and aligned template-test PCB image pairs
- Implemented image subtraction, thresholding, contour detection, and ROI extraction
- Trained ResNet-50 using transfer learning (50 epochs)
- Evaluated with accuracy/loss curves and confusion matrix
- Designed the sliding window + SSIM comparison approach for defect localization
- Built Streamlit web app with end-to-end inference pipeline
- Added download functionality for annotated images and detection logs

---

## What I Built on My Own (Beyond Internship)

After the internship, I rewrote major parts to make the project production-grade:

### Inference Pipeline Rewrite

| What I Changed | Before | After |
|---|---|---|
| **Paths** | Hardcoded absolute paths | Relative via `config.py` env vars |
| **Model loading** | Loaded at import time | Lazy-loaded on first inference |
| **Inference speed** | One patch at a time | Batch inference (32 patches/pass) |
| **GPU memory** | No `torch.no_grad()` | Proper no_grad context |
| **Error handling** | Bare `except:` | Specific catches, custom exceptions |
| **Logging** | `print()` with emojis | Python `logging` + rotating file handler |
| **SSIM** | `full=True` (wasted memory) | `full=False` |
| **Type safety** | None | Full type annotations + TypedDict |
| **Input validation** | None | 16MP pixel cap, size checks, 10MB limit |
| **Caching** | Model reloaded every click | `@st.cache_resource` singleton |

### Added REST API
- FastAPI with `/health` and `/detect` endpoints
- Pydantic schemas, proper HTTP status codes (400/413/422/500)

### Added Config, Docker, Tests, CI/CD
- `config.py` — centralised env-var settings + rotating file logging
- `Dockerfile` + `docker-compose.yml` — two-service deployment with health checks
- 24 unit tests (pytest) with mock models
- GitHub Actions CI — 4 parallel jobs (lint, typecheck, test, docker build)
- ruff + mypy — zero warnings, full type coverage

---

## Future Scope

- Vision Transformers (ViT) for higher accuracy
- Multi-scale sliding window for varying defect sizes
- Batch processing of multiple PCB images
- Real-time video stream detection
- Prometheus metrics endpoint
- Cloud deployment (AWS/GCP) + Kubernetes

---

## Author

**Aradhya Stuti**

- GitHub: [AradhyaStuti](https://github.com/AradhyaStuti)
- LinkedIn: [aradhya-stuti-9b2b9529a](https://www.linkedin.com/in/aradhya-stuti-9b2b9529a)
