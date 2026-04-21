# AI PCB Defect Detection and Classification
A small system for finding and labeling defects on printed circuit boards. It
compares a test image against a defect-free reference, flags the regions that
look wrong, and classifies each one with a CNN.

Built during my internship and then cleaned up afterwards so it can actually be
run, tested, and deployed.


## What it does

Given a PCB image, the pipeline:

1. Picks the closest golden (defect-free) reference from a folder of known
   boards, using perceptual hashing.
2. Slides a window across the image and compares each patch to the matching
   patch on the reference using SSIM.
3. Sends the patches that differ through a ResNet-50 classifier trained on the
   DeepPCB dataset.
4. Runs Non-Max Suppression so overlapping detections collapse into one.

Output is a list of boxes with labels and confidences, and an annotated copy of
the input image.


## Defect classes

From the DeepPCB dataset:

- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper


## Stack

- PyTorch / torchvision (ResNet-50)
- scikit-image (SSIM), ImageHash (pHash), NumPy, Pillow
- Streamlit for the UI, FastAPI for the API
- pytest, ruff, mypy
- Docker + docker-compose
- GitHub Actions for CI


## Layout

```
app.py              Streamlit UI
api.py              FastAPI endpoints
inference_new.py    Detection pipeline
config.py           Paths, ports, logging, env vars
model/              Trained weights (.pth, gitignored)
PCB_USED/           Golden reference images
tests/              pytest suite
```


## Running it

Clone and install:

```bash
git clone https://github.com/AradhyaStuti/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System
python -m venv venv
venv\Scripts\activate          # on Windows
pip install -r requirements.txt
```

Then either the UI:

```bash
streamlit run app.py
```

or the API:

```bash
uvicorn api:app --reload
```

You need the trained weights at `model/best_resnet50_pcb_defects_50epochs.pth`
and some golden images in `PCB_USED/`. Paths can be overridden via env vars
(see `.env.example`).


### Docker

```bash
docker-compose up --build
```

Brings up the API on `:8000` and the Streamlit UI on `:8501`.


## API

```
GET  /health     health check
POST /detect     upload a PCB image, get back detections
```

`/detect` takes a multipart file upload and returns JSON with the defect
count, inference time, and a list of `{label, confidence, box}` entries.


## Tests

```bash
pytest -v
```

The tests mock the model, so you don't need the weights to run them.


## Config

All runtime settings are environment variables prefixed with `PCB_`. Copy
`.env.example` to `.env` and edit what you need. See `config.py` for the full
list.


## What I did during the internship

- Got familiar with the DeepPCB dataset and the defect categories
- Wrote the preprocessing and region-extraction logic
- Trained the ResNet-50 classifier (`trained.ipynb`)
- Put together the detection pipeline (`inference.ipynb`)
- Built a Streamlit UI to test it interactively


## What I added afterwards

- Moved config into a single file driven by env vars
- Rewrote the inference pipeline as a class so it loads the model once
- Added batching so classification isn't one-patch-at-a-time
- Added a FastAPI layer and a Dockerfile / compose setup
- Wrote a pytest suite and a GitHub Actions workflow
- Added logging to a rotating file


## Things I'd still like to try

- Transformer-based backbones (ViT, Swin)
- Multi-scale windowing for defects of different sizes
- Real PCB images instead of only the DeepPCB ones


## Author

Aradhya Stuti

- https://github.com/AradhyaStuti
- https://www.linkedin.com/in/aradhya-stuti-9b2b9529a
