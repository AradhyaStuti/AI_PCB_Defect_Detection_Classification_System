<p align="center">
  <img src="image/README/1768377780977.png" alt="PCB Defect Detection Banner" width="800"/>
</p>

<h1 align="center">AI PCB Defect Detection & Classification System</h1>

<p align="center">
  <strong>Automated defect detection and classification for Printed Circuit Boards using deep learning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Model-ResNet50-green" alt="ResNet50"/>
  <img src="https://img.shields.io/badge/Dataset-DeepPCB-orange" alt="DeepPCB"/>
</p>

---

## What is this?

I built this project to solve a real problem in PCB manufacturing — finding defects quickly and accurately. The idea is simple: take a picture of a PCB, compare it against a known-good reference, and let a neural network figure out what went wrong.

The system uses **differential image analysis** (comparing test vs. golden reference using SSIM) combined with a **ResNet-50 classifier** trained on the DeepPCB dataset to detect and categorize six types of defects. Everything runs through a clean Streamlit web interface where you can upload an image, see results instantly, and download annotated outputs.

It's not perfect, but it works well enough to be genuinely useful. I've spent a lot of time refactoring the codebase to make it production-grade — proper error handling, batch inference, input validation, full test coverage, CI pipeline, the whole nine yards.

## Defect Categories

The model classifies defects into six categories from the DeepPCB benchmark:

| Defect | What it looks like |
|---|---|
| **Missing Hole** | A drill hole that should exist but doesn't |
| **Mouse Bite** | Irregular, jagged edges along copper traces |
| **Open Circuit** | A break or gap in a copper trace |
| **Short** | Unintended copper bridging two traces |
| **Spur** | Small unwanted copper protrusion from a trace |
| **Spurious Copper** | Random copper deposits where there shouldn't be any |

## How it works

The detection pipeline has three stages:

**1. Golden Reference Matching** — When you upload a PCB image, the system finds the closest matching reference image from the golden database using perceptual hashing (pHash). This handles slight variations in orientation and lighting.

**2. Sliding Window + SSIM** — A 128x128 pixel window slides across both the input and reference images. For each window position, structural similarity (SSIM) is computed. If the similarity drops below 0.95, that region is flagged as potentially defective.

**3. Batch Classification + NMS** — All suspicious patches are collected and fed through the ResNet-50 classifier in batches of 32 (much faster than one-by-one). Non-maximum suppression cleans up overlapping detections.

```
Input Image ──> pHash Match ──> SSIM Comparison ──> Batch CNN Classification ──> NMS ──> Annotated Output
                    │                  │                       │
              Golden DB          Threshold=0.95         Confidence>0.80
```

## Sample Results

Here are some actual detection results from the system:

| Missing Hole | Spurious Copper |
|---|---|
| ![Missing Hole](image/README/1768378765792.png) | ![Spurious Copper](image/README/1768378792341.png) |

| Spur | Short |
|---|---|
| ![Spur](image/README/1768378926200.png) | ![Short](image/README/1768379000627.png) |

| Open Circuit | Mouse Bite |
|---|---|
| ![Open Circuit](image/README/1768379036675.png) | ![Mouse Bite](image/README/1768379058640.png) |

**After running inference:**

![Inference Result](image/README/1768378725086.png)

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision (ResNet-50) |
| Image Comparison | scikit-image (SSIM), ImageHash (pHash) |
| Web Interface | Streamlit |
| Image Processing | Pillow, NumPy |
| Visualization | Matplotlib (colormaps) |
| Testing | pytest (15 tests) |
| CI/CD | GitHub Actions (lint + typecheck + test) |
| Linting | ruff, mypy |

## Project Structure

```
AI_PCB_Defect_Detection_Classification_System/
├── app.py                  # Streamlit web application
├── inference_new.py        # Detection pipeline (PCBDefectPipeline class)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project config (ruff, mypy, pytest)
├── model/
│   └── best_resnet50_pcb_defects_50epochs.pth   # Trained weights
├── PCB_USED/               # Golden reference images
│   ├── 01.JPG
│   ├── 04.JPG
│   └── ...
├── tests/
│   ├── conftest.py         # Shared test fixtures
│   └── test_inference.py   # 15 unit tests
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
└── image/
    └── README/             # Images used in this README
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- ~500MB disk space (mostly PyTorch)
- GPU optional but recommended for faster inference

### Installation

```bash
# Clone the repo
git clone https://github.com/AradhyaStuti/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System

# Create and activate virtual environment
python -m venv my_virtual_env
my_virtual_env\Scripts\activate        # Windows
# source my_virtual_env/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

If you want to retrain or experiment with the model:

[Download DeepPCB Dataset (Dropbox)](https://www.dropbox.com/scl/fi/4vrtqn7t001yl41oucflu/PCB_DATASET.zip?rlkey=pghz15q2bsg205wynjwsj2c3n&e=2&dl=0)

### Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload a PCB image, hit **Run detection**, and you'll see annotated results with bounding boxes, labels, and confidence scores. You can download both the annotated image and a CSV-style detection log.

### Run Tests

```bash
pip install pytest
pytest tests/ -v
```

All 15 tests should pass in under 2 seconds. They use mock models so you don't need the actual `.pth` file to run them.

## Architecture Decisions

A few things I deliberately chose and why:

- **Lazy loading** — The model and golden database don't load until the first inference call. This means importing the module is instant and side-effect-free, which matters for testing and for Streamlit's rerun model.
- **`@st.cache_resource`** — The pipeline singleton survives Streamlit reruns. Without this, the model would reload from disk every time you click a button.
- **Batch inference** — Instead of feeding patches through the model one at a time (which is painfully slow), I collect all SSIM-flagged patches first, then run them through in batches of 32. Big difference on GPU.
- **Input validation** — There's a 16MP pixel cap and a minimum image size check. Without this, someone could upload a 100MP image and OOM the server.
- **No hardcoded paths** — Everything is relative to the script directory. The old version had `C:\Users\User\...` paths baked in, which obviously breaks on any other machine.

## Known Limitations

I want to be upfront about what this system can and can't do:

- **SSIM comparison is CPU-bound** — The sliding window SSIM loop is pure Python. Vectorizing it with OpenCV's `matchTemplate` would be 3-5x faster, but I haven't gotten to it yet.
- **Requires golden references** — If you don't have a matching reference image in `PCB_USED/`, the system can't detect anything. It's not a standalone object detector.
- **Single-image inference only** — No batch file upload or directory processing yet.
- **Fixed sliding window** — The 128x128 window might miss very large or very small defects. An adaptive multi-scale approach would help.

## Future Scope

- Vision Transformer (ViT) backbone for potentially better accuracy
- Multi-scale sliding window for detecting defects of varying sizes
- Batch processing for multiple PCB images at once
- Real-time video stream detection for production line integration
- Docker containerization for easy deployment
- Cloud deployment (AWS/GCP) for industrial use

---

## Author

**Aradhya Stuti**

- GitHub: [AradhyaStuti](https://github.com/AradhyaStuti)
- LinkedIn: [aradhya-stuti-9b2b9529a](https://www.linkedin.com/in/aradhya-stuti-9b2b9529a)

---

<p align="center">
  Built with lots of coffee and frustration. If you found this useful, a star would make my day.
</p>
