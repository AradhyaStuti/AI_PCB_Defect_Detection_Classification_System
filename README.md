# AI PCB Defect Detection and Classification

A small system for detecting and classifying defects on printed circuit boards (PCBs).
It compares a test image against a defect-free reference, flags regions that look off, and then classifies each region using a CNN.

This started during my internship, and I later cleaned it up so it’s easier to run, test, and deploy.

## What it does

Given a PCB image, the pipeline:

1. Finds the closest “golden” (defect-free) reference from a folder using perceptual hashing
2. Slides a window over the image and compares each patch with the corresponding patch from the reference using SSIM
3. Sends only the mismatched patches through a ResNet-50 classifier trained on the DeepPCB dataset
4. Applies Non-Max Suppression to merge overlapping detections

The output is a list of bounding boxes with labels and confidence scores, along with an annotated version of the input image.

## Defect classes

From the DeepPCB dataset:

* missing_hole
* mouse_bite
* open_circuit
* short
* spur
* spurious_copper

## Tech stack

* PyTorch / torchvision (ResNet-50)
* scikit-image (SSIM), ImageHash (pHash), NumPy, Pillow
* Streamlit (UI), FastAPI (API)
* pytest, ruff, mypy
* Docker + docker-compose
* GitHub Actions for CI

## Project structure

```
app.py              Streamlit UI
api.py              FastAPI endpoints
inference_new.py    Detection pipeline
config.py           Config + env handling
model/              Trained weights (.pth, gitignored)
PCB_USED/           Golden reference images
tests/              pytest suite
```

## Running locally

Clone and install:

```bash
git clone https://github.com/AradhyaStuti/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Run either the UI:

```bash
streamlit run app.py
```

Or the API:

```bash
uvicorn api:app --reload
```

You’ll need:

* trained weights at `model/best_resnet50_pcb_defects_50epochs.pth`
* some reference images inside `PCB_USED/`

Paths can be configured via environment variables (see `.env.example`).

## Docker

```bash
docker-compose up --build
```

Starts:

* FastAPI on `:8000`
* Streamlit on `:8501`

## API

```
GET  /health     Health check  
POST /detect     Upload PCB image and get detections  
```

`/detect` accepts a multipart image and returns:

* defect count
* inference time
* list of `{label, confidence, box}`

## Tests

```bash
pytest -v
```

Tests mock the model, so the weights aren’t required.

## Design notes

Some decisions that might not be obvious:

* **SSIM instead of pixel difference**
  Pixel-wise comparison was too sensitive to lighting and minor variations. SSIM ended up being much more stable for this use case.

* **Sliding window with overlap (~75%)**
  This helps catch defects that fall near window boundaries. It does increase the number of patches quite a bit, but NMS handles the redundancy.

* **Two-stage approach (detect → classify)**
  I didn’t have enough bounding box annotations to train a full detection model like YOLO reliably. Using SSIM for region proposals and a classifier for labeling was simpler and worked well enough.

* **Lazy model loading**
  The ResNet weights are ~90 MB and take a moment to load, so the model initializes only on the first inference call instead of at import time.

## Config

All runtime settings use environment variables prefixed with `PCB_`.
Copy `.env.example` to `.env` and adjust as needed. See `config.py` for details.

## What I worked on during the internship

* Explored the DeepPCB dataset and defect categories
* Built preprocessing and region extraction logic
* Trained the ResNet-50 classifier (`trained.ipynb`)
* Put together a basic Streamlit UI for testing

## What I added afterwards

* Centralized config using environment variables
* Refactored the inference pipeline into a reusable class
* Added batching for faster classification
* Built a FastAPI layer
* Added Docker + docker-compose setup
* Wrote a pytest suite and CI workflow
* Added logging (rotating file handler)

## Limitations

* Assumes the input image is roughly aligned with the reference (rotation/shift can affect SSIM)
* Sliding window approach can be slow on high-resolution images (especially on CPU)
* Performance is tied to the DeepPCB dataset — may not generalize well to very different PCB layouts

## Future work

* Try transformer-based backbones (ViT, Swin)
* Add multi-scale windowing
* Test on real-world PCB images beyond DeepPCB

## Author

Aradhya Stuti

* GitHub: [https://github.com/AradhyaStuti](https://github.com/AradhyaStuti)
* LinkedIn: [https://www.linkedin.com/in/aradhya-stuti-9b2b9529a](https://www.linkedin.com/in/aradhya-stuti-9b2b9529a)
