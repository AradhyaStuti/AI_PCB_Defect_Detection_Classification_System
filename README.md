<p align="center">
  <img src="image/README/1768377780977.png" alt="PCB Defect Detection Banner" width="800"/>
</p>

<h1 align="center">AI PCB Defect Detection & Classification System</h1>

<p align="center">
  <strong>An automated system for detecting and classifying defects in Printed Circuit Boards (PCBs) using image processing and deep learning techniques.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Model-ResNet50-green" alt="ResNet50"/>
  <img src="https://img.shields.io/badge/Dataset-DeepPCB-orange" alt="DeepPCB"/>
</p>

---

## Project Statement

The objective is to develop an end-to-end defect detection and classification system for PCBs. The system:

- Detects and localizes defects using comparison with defect-free templates.
- Classifies detected defects into predefined categories using a trained CNN (ResNet-50).
- Provides a user-friendly frontend for image upload and viewing labeled outputs.
- Integrates a modular backend pipeline for processing images and returning annotated results.
- Exports annotated outputs and detection logs for documentation and analysis.

---

## Features

- Automated defect detection using template-based differential comparison (SSIM).
- Transfer learning-based classification using ResNet-50 trained on DeepPCB dataset.
- Perceptual hashing (pHash) for automatic golden reference matching.
- Batch inference pipeline — classifies patches in batches of 32 for GPU efficiency.
- Input validation with pixel budget limits and file size guards.
- Non-Maximum Suppression (NMS) to eliminate overlapping detections.
- Web-based frontend (Streamlit) for uploading images and viewing predictions in real-time.
- Annotated image export and CSV-style log generation for analysis.
- Full test suite (15 unit tests) with CI/CD pipeline via GitHub Actions.
- Production-grade code — type hints, structured logging, lazy model loading, zero hardcoded paths.

---

## Tech Stack

| Area | Tools / Libraries |
|---|---|
| Deep Learning | PyTorch, torchvision (ResNet-50) |
| Image Comparison | scikit-image (SSIM), ImageHash (pHash) |
| Image Processing | Pillow, NumPy |
| Frontend | Streamlit |
| Backend | Python, Modular Inference Pipeline |
| Visualization | Matplotlib (colormaps) |
| Testing | pytest (15 tests) |
| CI/CD | GitHub Actions (lint + typecheck + test) |
| Code Quality | ruff (linter), mypy (type checker) |
| Evaluation | Accuracy, Loss Curves, Confusion Matrix |

---

## Dataset

**DeepPCB Dataset:** [Download from Dropbox](https://www.dropbox.com/scl/fi/4vrtqn7t001yl41oucflu/PCB_DATASET.zip?rlkey=pghz15q2bsg205wynjwsj2c3n&e=2&dl=0)

The model classifies defects into six categories from the DeepPCB benchmark:

| Defect | Description |
|---|---|
| **Missing Hole** | A drill hole that should exist but doesn't |
| **Mouse Bite** | Irregular, jagged edges along copper traces |
| **Open Circuit** | A break or gap in a copper trace |
| **Short** | Unintended copper bridging two traces |
| **Spur** | Small unwanted copper protrusion from a trace |
| **Spurious Copper** | Random copper deposits where there shouldn't be any |

---

## System Workflow

```
AI PCB Defect Detection and Classification System
│
├── 1. Dataset Preparation
│   ├── 1.1 Dataset Collection
│   │   ├── DeepPCB Dataset
│   │   └── Defect-free (Template) Images
│   ├── 1.2 Image Preprocessing
│   │   ├── Image Alignment
│   │   ├── Grayscale Conversion
│   │   ├── Noise Reduction
│   │   └── Normalization
│   └── 1.3 Image Subtraction
│       ├── Template – Test Image Subtraction
│       ├── Thresholding
│       └── Binary Defect Mask Generation
│
├── 2. Defect Localization
│   ├── 2.1 Contour Detection
│   │   ├── Find Defect Contours
│   │   └── Filter Small/Irrelevant Contours
│   ├── 2.2 ROI Extraction
│   │   ├── Bounding Box Generation
│   │   └── Cropped Defect Patches
│   └── 2.3 Defect Visualization
│       ├── Contour Overlay
│       └── Bounding Box Annotation
│
├── 3. Dataset Preparation for Training
│   ├── 3.1 Label Assignment
│   │   ├── Missing Hole
│   │   ├── Spur
│   │   ├── Spurious Copper
│   │   ├── Short
│   │   ├── Open Circuit
│   │   └── Mouse Bite
│   ├── 3.2 Image Resizing
│   │   └── Resize ROIs to 128 x 128
│   └── 3.3 Data Augmentation
│       ├── Rotation
│       ├── Flipping
│       ├── Brightness Adjustment
│       └── Scaling
│
├── 4. Model Training
│   ├── 4.1 Model Selection
│   │   └── Transfer Learning (ResNet50)
│   ├── 4.2 Training Pipeline
│   │   ├── Forward Pass
│   │   ├── Loss Computation
│   │   ├── Backpropagation
│   │   └── Weight Optimization
│   └── 4.3 Model Evaluation
│       ├── Accuracy & Loss Curves
│       ├── Confusion Matrix
│       └── Class-wise Performance
│
├── 5. Inference Pipeline
│   ├── 5.1 Image Upload
│   │   └── Test PCB Image
│   ├── 5.2 Golden Reference Matching
│   │   └── pHash-based Best Match Selection
│   ├── 5.3 Defect Detection
│   │   ├── Sliding Window (128x128, stride 32)
│   │   └── SSIM Comparison (threshold 0.95)
│   ├── 5.4 Defect Classification
│   │   ├── Batch Inference (32 patches/forward pass)
│   │   └── CNN Prediction (confidence > 0.80)
│   └── 5.5 Post-processing
│       ├── Non-Maximum Suppression (IoU 0.2)
│       ├── Bounding Boxes & Labels
│       └── Confidence Scores
│
├── 6. Web Application (Frontend)
│   ├── Streamlit UI
│   │   ├── Image Upload Interface
│   │   ├── Input Validation (16MP limit, 10MB file size)
│   │   ├── Real-time Predictions
│   │   └── Result Visualization
│   └── User Interaction
│       ├── View Annotated Images
│       ├── Detection Results Table
│       └── Download Outputs (PNG + TXT Log)
│
├── 7. Backend Integration
│   ├── PCBDefectPipeline Class
│   │   ├── Lazy Model Loading
│   │   ├── Cached via @st.cache_resource
│   │   ├── Image Processing Module
│   │   ├── Batch Inference Module
│   │   └── Annotation Module
│   └── Logging & Export
│       ├── Structured Logging (Python logging)
│       ├── Prediction Logs (CSV format)
│       └── Annotated Image Export
│
├── 8. Testing & CI/CD
│   ├── Unit Tests (15 tests, pytest)
│   │   ├── Input Validation Tests
│   │   ├── Golden Database Tests
│   │   ├── Batch Classification Tests
│   │   ├── Detection Pipeline Tests
│   │   ├── Visualization Tests
│   │   └── End-to-End Pipeline Tests
│   └── GitHub Actions CI
│       ├── Lint (ruff)
│       ├── Type Check (mypy)
│       └── Test (pytest)
│
└── 9. Final Output
    ├── Annotated PCB Image
    ├── Defect Class Labels
    ├── Confidence Scores
    ├── Detection Log (CSV)
    └── Deployment-Ready Application
```

---

## Project Modules & Milestones

### Milestone 1: Dataset Preparation and Image Processing

**Module 1: Dataset Setup and Image Subtraction**
- Aligned and preprocessed template-test image pairs.
- Applied image subtraction and thresholding to highlight defects.
- **Deliverables:** Cleaned dataset, subtraction scripts, sample defect-highlighted images.

**Module 2: Contour Detection and ROI Extraction**
- Detected contours of defects and extracted ROI for model training.
- **Deliverables:** ROI extraction pipeline, labeled defect samples, visualization of contours.

**Results:**

| Missing Hole | Spurious Copper |
|---|---|
| ![Missing Hole](image/README/1768378765792.png) | ![Spurious Copper](image/README/1768378792341.png) |

| Spur | Short |
|---|---|
| ![Spur](image/README/1768378926200.png) | ![Short](image/README/1768379000627.png) |

| Open Circuit | Mouse Bite |
|---|---|
| ![Open Circuit](image/README/1768379036675.png) | ![Mouse Bite](image/README/1768379058640.png) |

---

### Milestone 2: Model Training and Evaluation

**Module 3: Model Training**
- Used transfer learning with ResNet-50 for defect classification.
- Preprocessed and augmented images (128x128) for training over 50 epochs.
- **Deliverables:** Trained model (`best_resnet50_pcb_defects_50epochs.pth`), accuracy/loss metrics, confusion matrix.

**Module 4: Evaluation and Prediction Testing**
- Tested model on unseen images.
- Compared predictions against ground truth annotations.
- **Deliverables:** Annotated test images, final evaluation report.

**Result after inference:**

![Inference Result](image/README/1768378725086.png)

---

### Milestone 3: Frontend and Backend Integration

**Module 5: Web UI for Image Upload**
- Built Streamlit-based interface for PCB image uploads.
- Displays annotated images with defect labels and confidence scores in real-time.
- Added input validation (10MB upload limit, 16MP pixel cap).

**Module 6: Backend Pipeline for Inference**
- Built `PCBDefectPipeline` class with lazy loading and batch inference.
- Connected backend to Streamlit frontend with `@st.cache_resource` caching.
- **Deliverables:** Full prediction pipeline, annotated outputs, CSV detection logs.

---

### Milestone 4: Testing, Code Quality & CI/CD

**Module 7: Unit Testing**
- Wrote 15 unit tests covering validation, golden DB, batch classification, detection, visualization, and end-to-end pipeline.
- Tests use mock models — no `.pth` file needed to run them.
- All tests pass in under 2 seconds.

**Module 8: Code Quality & CI/CD**
- Configured ruff linter with strict rules (bugbear, simplify, type-checking).
- Added mypy type checking configuration.
- Set up GitHub Actions CI with 3 parallel jobs: lint, typecheck, test.
- Zero lint warnings, full type annotations across codebase.

**Module 9: Documentation & Finalization**
- Comprehensive README with project structure, setup instructions, and results.
- Production-grade codebase with no hardcoded paths, proper error handling, and structured logging.

---

## Project Structure

```
AI_PCB_Defect_Detection_Classification_System/
├── app.py                  # Streamlit web application
├── inference_new.py        # Detection pipeline (PCBDefectPipeline class)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project config (ruff, mypy, pytest)
├── model/
│   └── best_resnet50_pcb_defects_50epochs.pth
├── PCB_USED/               # Golden reference images
│   ├── 01.JPG
│   ├── 04.JPG
│   └── ...
├── tests/
│   ├── conftest.py         # Shared test fixtures & mock model
│   └── test_inference.py   # 15 unit tests
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
└── image/
    └── README/             # Images used in this README
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/AradhyaStuti/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System
```

**2. Create and activate a virtual environment**
```bash
python -m venv my_virtual_env
my_virtual_env\Scripts\activate        # Windows
# source my_virtual_env/bin/activate   # Linux/Mac
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your trained model weights**

Place `best_resnet50_pcb_defects_50epochs.pth` inside the `model/` folder.

---

## How to Use

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your browser.
2. Upload a PCB image (JPG/PNG, max 10MB).
3. Click **Run detection**.
4. View results with bounding boxes, defect labels, and confidence scores.
5. Download the annotated image (PNG) and detection log (TXT).

---

## How to Run Tests

```bash
pip install pytest
pytest tests/ -v
```

All 15 tests pass in under 2 seconds. No model file needed — tests use mock models.

---

## How the Detection Pipeline Works

```
Input Image ──> pHash Match ──> SSIM Comparison ──> Batch CNN Classification ──> NMS ──> Annotated Output
                    │                  │                       │
              Golden DB          Threshold=0.95         Confidence>0.80
```

**Step 1 — Golden Reference Matching:** The system finds the closest matching reference image from the golden database using perceptual hashing (pHash).

**Step 2 — Sliding Window + SSIM:** A 128x128 pixel window slides across both images. Structural similarity (SSIM) is computed for each position. Regions with SSIM < 0.95 are flagged as anomalous.

**Step 3 — Batch Classification:** All suspicious patches are collected and classified through ResNet-50 in batches of 32. This is significantly faster than classifying one patch at a time.

**Step 4 — Non-Maximum Suppression:** Overlapping detections are filtered using NMS (IoU threshold = 0.2) to produce clean final results.

---

## Key Architecture Decisions

- **Lazy Loading** — Model and golden database load only on first inference call. Importing the module has zero side effects.
- **`@st.cache_resource`** — Pipeline singleton survives Streamlit reruns. Model doesn't reload on every button click.
- **Batch Inference** — Patches classified in batches of 32 instead of one-by-one. Major speed improvement on GPU.
- **Input Validation** — 16MP pixel budget and 10MB upload limit prevent OOM crashes.
- **Relative Paths** — All paths are relative to the script directory. No hardcoded absolute paths.
- **Structured Logging** — Python `logging` module instead of print statements.

---

## What I Did During This Internship

This project went through two major phases. The first phase was building the core detection system from scratch. The second phase was refactoring the entire codebase to make it production-ready. Here's a breakdown of my contributions:

### Phase 1 — Built the Core System

- Collected and prepared the DeepPCB dataset with template-test image pairs.
- Implemented image subtraction and thresholding pipeline to highlight defect regions.
- Built contour detection and ROI extraction to isolate individual defects.
- Trained a ResNet-50 classifier using transfer learning on 6 defect categories (50 epochs).
- Created the sliding window + SSIM comparison approach for differential defect detection.
- Built the Streamlit web app for uploading PCB images and viewing results.
- Integrated the inference backend with the frontend for end-to-end predictions.

### Phase 2 — Refactored to Production Quality

The initial codebase worked but had several issues — hardcoded paths, no error handling, model loading on every import, debug prints everywhere. I rewrote the entire pipeline to meet industry standards:

| What I Changed | Before | After |
|---|---|---|
| **File paths** | Hardcoded `C:\Users\User\...` absolute paths | Relative paths using `Path(__file__).parent` |
| **Model loading** | Loaded at import time, crashed if path missing | Lazy-loaded `PCBDefectPipeline` class, loads only when needed |
| **Inference speed** | One patch at a time through the model | Batch inference (32 patches per forward pass) |
| **GPU memory** | No `torch.no_grad()` in detection loop | Proper `no_grad` context, no gradient accumulation |
| **Error handling** | Bare `except:` catching everything | Specific `OSError` handling, custom `ImageTooLargeError` |
| **Logging** | Emoji-filled `print()` statements | Python `logging` module with structured format |
| **SSIM call** | `ssim(full=True)` allocating unused diff image | `full=False` (default), saves memory |
| **Matplotlib** | Deprecated `plt.cm.get_cmap()` | `plt.colormaps["hsv"]` |
| **Softmax** | Computed twice per patch in some cases | Single computation per patch |
| **Type safety** | No type hints anywhere | Full annotations, `TypedDict`, `from __future__ import annotations` |
| **Input validation** | None — could OOM on huge images | 16MP pixel cap, 10MB upload limit, minimum size check |
| **Streamlit caching** | Model reloaded on every rerun | `@st.cache_resource` keeps pipeline alive |
| **Datetime** | Naive `datetime.now()` | Timezone-aware `datetime.now(tz=timezone.utc)` |

### Phase 3 — Added Testing & CI/CD

- Wrote 15 unit tests using pytest with mock models (no `.pth` file dependency).
- Tests cover: input validation, golden database, batch classification, anomaly detection, visualization, and full pipeline.
- Set up `pyproject.toml` with ruff (linter) and mypy (type checker) configurations.
- Created GitHub Actions CI pipeline with 3 parallel jobs: lint, typecheck, test.
- Achieved zero lint warnings across the entire codebase.

---

## Future Scope

- Support Vision Transformers (ViT) for higher accuracy.
- Multi-scale sliding window for defects of varying sizes.
- Batch processing of multiple PCB images.
- Real-time video stream defect detection.
- Docker containerization for easy deployment.
- Cloud deployment (AWS/GCP) for industrial production line use.

---

## Author

**Aradhya Stuti**

- GitHub: [AradhyaStuti](https://github.com/AradhyaStuti)
- LinkedIn: [aradhya-stuti-9b2b9529a](https://www.linkedin.com/in/aradhya-stuti-9b2b9529a)
