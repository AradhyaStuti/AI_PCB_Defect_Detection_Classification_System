##  AI PCB Defect Detection & Classification System

An end-to-end system for detecting and classifying defects in Printed Circuit Boards (PCBs) using image processing and deep learning.

It compares test images with defect-free reference images, identifies anomalies, and classifies them into known defect categories.

---

##  Overview

This project combines **classical image processing** and **deep learning**:

* Uses **SSIM-based comparison** to detect anomalous regions
* Applies **ResNet-50** to classify detected defects
* Provides both a **Streamlit UI** and a **FastAPI backend**

The goal is to build a practical inspection system that can assist in automated PCB quality analysis.

---

##  Features

* SSIM-based sliding window detection
* Automatic reference matching using pHash
* CNN-based classification (ResNet-50 trained on DeepPCB)
* Batch inference for efficient processing
* Non-Max Suppression to refine detections
* Streamlit interface for interactive testing
* FastAPI endpoints for programmatic access
* Configurable via environment variables
* Logging and basic validation checks
* Unit tests with CI pipeline

---

##  How It Works

```
Input Image → Reference Match → SSIM Comparison → CNN Classification → NMS → Output
```

1. **Reference Matching** — finds the closest defect-free image
2. **SSIM Comparison** — identifies regions that differ significantly
3. **Classification** — predicts defect type for detected regions
4. **Post-processing** — removes overlapping detections

---

##  Dataset

Uses the **DeepPCB dataset**, which contains annotated PCB defects across 6 categories:

* Missing Hole
* Mouse Bite
* Open Circuit
* Short
* Spur
* Spurious Copper

---

##  Tech Stack

* **Deep Learning:** PyTorch, torchvision
* **Image Processing:** scikit-image, ImageHash, NumPy
* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Testing:** pytest
* **DevOps:** Docker, GitHub Actions
* **Code Quality:** ruff, mypy

---

##  Project Structure

```
app.py              # Streamlit app
api.py              # FastAPI endpoints
inference_new.py    # Detection pipeline
config.py           # Configuration
model/              # Trained model
tests/              # Unit tests
```

---

##  Setup

### Local

```bash
git clone https://github.com/AradhyaStuti/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System

python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

or

```bash
uvicorn api:app --reload
```

---

### Docker

```bash
docker-compose up --build
```

---

##  API

```bash
GET /health
POST /detect
```

Returns detected defects with labels and bounding boxes.

---

##  Testing

```bash
pytest -v
```

Includes unit tests for both pipeline and API.

---

##  What I Worked On

During my internship, I:

* Explored the DeepPCB dataset and defect categories
* Implemented preprocessing and region extraction logic
* Trained a ResNet-50 model for classification
* Built an end-to-end inference pipeline
* Developed a Streamlit interface for testing

---

## Improvements After Internship

I later refined the project to make it more structured and easier to run:

* Centralized configuration using environment variables
* Improved inference efficiency with batch processing
* Added proper logging and error handling
* Introduced API endpoints for external use
* Added tests and CI setup

---

##  Future Work

* Explore transformer-based models
* Improve detection for varying defect sizes
* Support batch image processing
* Extend to real-time inspection

---

##  Author

**Aradhya Stuti**

* GitHub: [https://github.com/AradhyaStuti](https://github.com/AradhyaStuti)
* LinkedIn: [https://www.linkedin.com/in/aradhya-stuti-9b2b9529a](https://www.linkedin.com/in/aradhya-stuti-9b2b9529a)

---
