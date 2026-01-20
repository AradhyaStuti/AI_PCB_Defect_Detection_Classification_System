""AI PCB Defect Detection and Classification System""

An automated system for detecting and classifying defects in Printed Circuit Boards (PCBs) using image processing and deep learning techniques. The system leverages reference-based image subtraction, contour extraction, and CNN-based classification to identify and label defects. A fully functional web application allows users to upload PCB images and receive annotated outputs highlighting defects.

---

📌 Project Statement
The objective is to develop an end-to-end defect detection and classification system for PCBs. The system will:
- Detect and localize defects using comparison with defect-free templates.
- Classify detected defects into predefined categories using a trained CNN.
- Provide a user-friendly frontend for image upload and viewing labeled outputs.
- Integrate a backend pipeline for processing images and returning annotated results.
- Export annotated outputs and logs for documentation and analysis.

---

✨ Features
- Automated defect detection using template-based subtraction and contour extraction.
- Transfer learning-based classification using CNN models (ResNet).
- Preprocessing and augmentation of PCB images for robust model training.
- Web-based frontend for uploading images and viewing predictions in real-time.
- Annotated image export and log generation for analysis.

---

🛠️ Tech Stack
| Area               | Tools / Libraries                        |
|-------------------|-----------------------------------------|
| Image Processing   | OpenCV                                   |
| Neural Networks    | PyTorch                                  |
| Dataset            | DeepPCB                                   |
| Frontend           | Streamlit                                |
| Backend            | Python, Modularized Inference Pipeline  |
| Evaluation         | Accuracy, Loss, Confusion Matrix        |

---
Dataset: https://www.dropbox.com/scl/fi/4vrtqn7t001yl41oucflu/PCB_DATASET.zip?rlkey=pghz15q2bsg205wynjwsj2c3n&e=2&dl=0

![1768377780977](image/README/1768377780977.png)

---
System Workflow

AI PCB Defect Detection and Classification System
│
├── 1. Dataset Preparation
│   │
│   ├── 1.1 Dataset Collection
│   │   ├── DeepPCB Dataset
│   │   └── Defect-free (Template) Images
│   │
│   ├── 1.2 Image Preprocessing
│   │   ├── Image Alignment
│   │   ├── Grayscale Conversion
│   │   ├── Noise Reduction
│   │   └── Normalization
│   │
│   └── 1.3 Image Subtraction
│       ├── Template – Test Image Subtraction
│       ├── Thresholding
│       └── Binary Defect Mask Generation
│
├── 2. Defect Localization
│   │
│   ├── 2.1 Contour Detection
│   │   ├── Find Defect Contours
│   │   └── Filter Small/Irrelevant Contours
│   │
│   ├── 2.2 ROI Extraction
│   │   ├── Bounding Box Generation
│   │   └── Cropped Defect Patches
│   │
│   └── 2.3 Defect Visualization
│       ├── Contour Overlay
│       └── Bounding Box Annotation
│
├── 3. Dataset Preparation for Training
│   │
│   ├── 3.1 Label Assignment
│   │   ├── Missing Hole
│   │   ├── Spur
│   │   ├── Spurious Copper
│   │   ├── Short
│   │   ├── Open Circuit
│   │   └── Mouse Bite
│   │
│   ├── 3.2 Image Resizing
│   │   └── Resize ROIs to 128 × 128
│   │
│   └── 3.3 Data Augmentation
│       ├── Rotation
│       ├── Flipping
│       ├── Brightness Adjustment
│       └── Scaling
│
├── 4. Model Training
│   │
│   ├── 4.1 Model Selection
│   │   └── Transfer Learning (ResNet18 / ResNet50)
│   │
│   ├── 4.2 Training Pipeline
│   │   ├── Forward Pass
│   │   ├── Loss Computation
│   │   ├── Backpropagation
│   │   └── Weight Optimization
│   │
│   └── 4.3 Model Evaluation
│       ├── Accuracy & Loss Curves
│       ├── Confusion Matrix
│       └── Class-wise Performance
│
├── 5. Inference Pipeline
│   │
│   ├── 5.1 Image Upload
│   │   ├── Template Image
│   │   └── Test PCB Image
│   │
│   ├── 5.2 Defect Detection
│   │   ├── Image Subtraction
│   │   └── Contour Extraction
│   │
│   ├── 5.3 Defect Classification
│   │   ├── ROI Cropping
│   │   └── CNN Prediction
│   │
│   └── 5.4 Output Annotation
│       ├── Bounding Boxes
│       ├── Defect Labels
│       └── Confidence Scores
│
├── 6. Web Application (Frontend)
│   │
│   ├── Streamlit UI
│   │   ├── Image Upload Interface
│   │   ├── Real-time Predictions
│   │   └── Result Visualization
│   │
│   └── User Interaction
│       ├── View Annotated Images
│       └── Download Outputs
│
├── 7. Backend Integration
│   │
│   ├── Modular Inference Pipeline
│   │   ├── Image Processing Module
│   │   ├── Model Inference Module
│   │   └── Annotation Module
│   │
│   └── Logging & Export
│       ├── Prediction Logs
│       └── Annotated Image Export
│
└── 8. Final Output
    │
    ├── Annotated PCB Image
    ├── Defect Class Labels
    ├── Performance Metrics
    └── Deployment-Ready Application


📁 Project Modules & Milestones

**Milestone 1: Dataset Preparation and Image Processing**

    **Module 1: Dataset Setup and Image Subtraction**
- Align and preprocess template-test image pairs.
- Apply image subtraction and thresholding to highlight defects.
- **Deliverables:** Cleaned dataset, subtraction scripts, sample defect-highlighted images.

    **Module 2: Contour Detection and ROI Extraction**
- Detect contours of defects and extract ROI for model training.
- **Deliverables:** ROI extraction pipeline, labeled defect samples, visualization of contours.
RESULTS:
MISSING HOLE:
![1768378765792](image/README/1768378765792.png)
SPURIOUS COPPER:
![1768378792341](image/README/1768378792341.png)
SPUR:
![1768378926200](image/README/1768378926200.png)
SHORT:
![1768379000627](image/README/1768379000627.png)
OPEN CIRCUIT:
![1768379036675](image/README/1768379036675.png)
MOUSE BITE:
![1768379058640](image/README/1768379058640.png)
---

**Milestone 2: Model Training and Evaluation**
    **Module 3: Model Training**
- Use transfer learning (ResNet) for defect classification.
- Preprocess and augment images (128×128) for training.
- **Deliverables:** Trained model, accuracy/loss metrics, confusion matrix.

    **Module 4: Evaluation and Prediction Testing**
- Test model on unseen images.
- Compare predictions against ground truth annotations.
- **Deliverables:** Annotated test images, final evaluation report.
RESULT AFTER INFERENCE:
![1768378725086](image/README/1768378725086.png)
---

**Milestone 3: Frontend and Backend Integration**
    **Module 5: Web UI for Image Upload**
- Streamlit-based interface for template and test image uploads.
- Display annotated images with defect labels in real-time.

    **Module 6: Backend Pipeline for Inference**
- Modularized image processing and model inference.
- Connect backend to frontend upload inputs.
- **Deliverables:** Full prediction pipeline, annotated outputs, logs.

---

**Milestone 4: Finalization**
    **Module 7: Testing, Evaluation & Export**
- Export annotated images and prediction logs.
- Optimize pipeline for speed and performance.

    **Module 8: Documentation & Presentation**

---

🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/AI_PCB_Defect_Detection_Classification_System.git
cd AI_PCB_Defect_Detection_Classification_System

2. **Create and activate a virtual environment**

3. **Install dependencies**
pip install -r requirements.txt

4. **Add your trained model weights in the models/ folder (e.g., resnet50.pth)**
5. ** Install streamlit**
6. 🚀 How to Use

         streamlit run app.py
         Upload PCB images via the frontend.
         View results with bounding boxes and defect labels.
         Download outputs and logs for further analysis.

🔮 Future Scope:

    Support advanced models like Vision Transformers for higher accuracy.

    Batch processing of PCB images.

    Real-time video stream defect detection.

    Cloud deployment for industrial production line use.    


AUTHOR: Aradhya Stuti
GitHub: https://github.com/AradhyaStuti
LinkedIn: https://www.linkedin.com/in/aradhya-stuti-9b2b9529a    
