
# 🧠 Brain Tumor Detection & Classification using VGG16 and YOLOv8  
Automated MRI Analysis Using Deep Learning (Classification + Localization)

This project provides a dual deep-learning pipeline for analyzing Brain Tumors (BT) from MRI scans using:

- **VGG16 (Transfer Learning)** → Tumor **Classification**
- **YOLOv8** → Tumor **Detection & Localization** with bounding boxes

The system identifies tumor types and visually marks the region of interest on MRI images.

---

## 📌 Features  
- ✔ CNN-based classification  
- ✔ YOLOv8 tumor localization  
- ✔ High performance on MRI datasets  
- ✔ F1-score, Confusion Matrix, mAP evaluation  
- ✔ GPU-accelerated training support  
- ✔ Ready-to-run scripts for training and inference  

---

## 📁 Project Structure

```
Deep-MRIC/
├── data/
│   ├── raw_dataset/          # Raw MRI images organized by class
│   │   ├── NO_TUMOR/
│   │   ├── GLIOMA/
│   │   ├── MENINGIOMA/
│   │   └── PITUITARY/
│   ├── vgg16_classification/  # Preprocessed images for VGG16 (generated)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── yolov8/
│       ├── images/
│       └── labels/
├── models/
├── scripts/
│   ├── classify_vgg16.py
│   ├── classify_infer.py
│   ├── yolo_infer.py
│   ├── balance_dataset.py
│   └── evaluate_classification.py
├── runs/
├── preprocess_vgg16.py        # Python script for data preprocessing
├── preprocess_vgg16.ipynb     # Jupyter notebook for interactive preprocessing
├── tumor_data.yaml
├── requirements.txt            # pip requirements (Windows & Linux compatible)
├── environment.yml             # conda environment file (Windows & Linux compatible)
└── README.md
```

---

# 🚀 Getting Started

## 1️⃣ Prerequisites  
- Python **3.8+** (3.8 to 3.11 recommended)
- NVIDIA GPU with CUDA support (recommended for training)
- pip or conda package manager
- Git (for cloning the repository)

---

## 2️⃣ Installation

### Option A: Using pip (Recommended for most users)

#### Windows:
```bash
# Clone the repository
git clone https://github.com/<your_username>/Deep-MRIC.git
cd Deep-MRIC

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Linux/macOS:
```bash
# Clone the repository
git clone https://github.com/<your_username>/Deep-MRIC.git
cd Deep-MRIC

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using conda (Recommended for data science workflows)

#### Windows & Linux:
```bash
# Clone the repository
git clone https://github.com/<your_username>/Deep-MRIC.git
cd Deep-MRIC

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate deep-mric
```

### Verify Installation

To verify that all packages are installed correctly:

```bash
# Check Python version (should be 3.8+)
python --version

# Check if key packages are installed
python -c "import cv2, numpy, sklearn, tqdm; print('✅ All packages installed successfully!')"

# For Jupyter notebook support
jupyter --version
```

---

## 3️⃣ Data Preprocessing

Before training the VGG16 model, you need to preprocess your raw MRI images. The preprocessing script will:
- Crop brain regions (remove black background)
- Resize images to 224×224 (VGG16 input size)
- Perform stratified train/val/test split (70/15/15)

### Step 1: Organize Raw Data

Place your raw MRI images in the following structure:

```
data/raw_dataset/
├── NO_TUMOR/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── GLIOMA/
│   ├── image1.jpg
│   └── ...
├── MENINGIOMA/
│   └── ...
└── PITUITARY/
    └── ...
```

**Supported image formats:** `.jpg`, `.jpeg`, `.png`

### Step 2: Run Preprocessing

#### Option A: Using Python Script
```bash
python preprocess_vgg16.py
```

#### Option B: Using Jupyter Notebook (Interactive)
```bash
# Start Jupyter Notebook
jupyter notebook

# Open preprocess_vgg16.ipynb and run all cells
```

The preprocessing will create the following structure:

```
data/vgg16_classification/
├── train/
│   ├── NO_TUMOR/
│   ├── GLIOMA/
│   ├── MENINGIOMA/
│   └── PITUITARY/
├── val/
│   └── [same class folders]
└── test/
    └── [same class folders]
```

**Note:** You can modify the configuration in the script/notebook:
- `RAW_DATA_DIR`: Input directory path
- `VGG_OUTPUT_DIR`: Output directory path
- `SPLIT_RATIO`: Train/Val/Test split ratios
- `CLASSES`: List of class names

---

# 📦 Dataset Setup

## A) VGG16 Classification

After running the preprocessing script (`preprocess_vgg16.py` or `preprocess_vgg16.ipynb`), your data will be organized as:

```
data/vgg16_classification/
├── train/
│   ├── NO_TUMOR/
│   ├── GLIOMA/
│   ├── MENINGIOMA/
│   └── PITUITARY/
├── val/
│   ├── NO_TUMOR/
│   ├── GLIOMA/
│   ├── MENINGIOMA/
│   └── PITUITARY/
└── test/
    ├── NO_TUMOR/
    ├── GLIOMA/
    ├── MENINGIOMA/
    └── PITUITARY/
```

**Note:** If you already have preprocessed data in a different structure, you may need to adjust the paths in your training scripts.

---

## B) YOLOv8 Detection

YOLO expects:

### Image folders:

```
data/yolov8/images/train/
data/yolov8/images/val/
```

### Label folders (same filenames, `.txt` format):

```
data/yolov8/labels/train/
data/yolov8/labels/val/
```

### Example YOLO label:

```
0 0.52 0.41 0.33 0.44
```

### tumor_data.yaml:

```yaml
train: data/yolov8/images/train
val: data/yolov8/images/val

nc: 1
names: ["tumor"]
```

---

# 🔥 Training

## 1️⃣ Train VGG16 — Classification

**Important:** Make sure you've run the preprocessing script first (see [Data Preprocessing](#3️⃣-data-preprocessing) section).

```bash
python scripts/classify_vgg16.py \
    --data_dir data/vgg16_classification \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --model_save_path models/vgg16_classifier_best.pth
```

**Windows users:** If the backslash continuation doesn't work, use:
```bash
python scripts/classify_vgg16.py --data_dir data/vgg16_classification --epochs 30 --batch_size 64 --learning_rate 1e-4 --model_save_path models/vgg16_classifier_best.pth
```

---

## 2️⃣ Train YOLOv8 — Detection

```bash
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=tumor_data.yaml \
    epochs=50 \
    imgsz=640 \
    name=yolov8_tumor_detection
```

---

# 🔍 Inference / Testing

## 1️⃣ Classification (VGG16)

```bash
python scripts/classify_infer.py \
    --image_path data/test_images/sample_mri.png \
    --model_path models/vgg16_classifier_best.pth
```

---

## 2️⃣ Detection (YOLOv8)

```bash
yolo task=detect mode=predict \
    model=runs/detect/yolov8_tumor_detection/weights/best.pt \
    source=data/test_images/sample_mri.png \
    conf=0.25
```

Output saved in:

```
runs/detect/predict/
```

---

# 📊 Evaluation

## VGG16 Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## YOLOv8 Metrics

* mAP@0.5
* mAP@0.5:0.95
* IoU
* Precision-Recall curves

Run YOLO evaluation:

```bash
yolo mode=val model=runs/detect/yolov8_tumor_detection/weights/best.pt data=tumor_data.yaml
```

---

# 🛠 Requirements

All dependencies are listed in `requirements.txt` and `environment.yml`. The main packages include:

## Core Dependencies
- **opencv-python** (≥4.5.0) - Image processing and computer vision
- **numpy** (≥1.21.0) - Numerical computing
- **scikit-learn** (≥1.0.0) - Machine learning utilities (train/test split)
- **tqdm** (≥4.62.0) - Progress bars

## Deep Learning (for training)
- **torch** (≥1.12.0) - PyTorch framework
- **torchvision** - PyTorch vision utilities
- **ultralytics** - YOLOv8 implementation

## Jupyter Notebook Support
- **jupyter** (≥1.0.0) - Jupyter notebook environment
- **ipykernel** (≥6.0.0) - Jupyter kernel
- **matplotlib** (≥3.5.0) - Plotting and visualization
- **ipywidgets** (≥7.6.0) - Interactive widgets

## Installation

Install all requirements using one of the methods in the [Installation](#2️⃣-installation) section above.

**Quick install:**
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
```

---

# 🧪 Results

| Model      | Task           | Best Metric                       |
| ---------- | -------------- | --------------------------------- |
| **VGG16**  | Classification | High accuracy & F1-score          |
| **YOLOv8** | Detection      | High mAP & precise bounding boxes |
---


# 📜 License

This project is free for research and educational use.

---

# 🐛 Troubleshooting

## Common Issues

### Issue: `ModuleNotFoundError` when running scripts
**Solution:** Make sure your virtual environment is activated and all dependencies are installed:
```bash
# Activate venv
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: OpenCV installation fails
**Solution:** Try installing with conda instead:
```bash
conda install -c conda-forge opencv
```

### Issue: Preprocessing script can't find images
**Solution:** 
- Check that your raw data is in `data/raw_dataset/` with class subfolders
- Verify image file extensions are `.jpg`, `.jpeg`, or `.png`
- Check file permissions (especially on Linux)

### Issue: Jupyter notebook not starting
**Solution:** 
- Install Jupyter: `pip install jupyter ipykernel`
- Or use conda: `conda install jupyter ipykernel`
- Start with: `jupyter notebook`

### Issue: CUDA/GPU not detected (for training)
**Solution:**
- Install PyTorch with CUDA support: Visit [pytorch.org](https://pytorch.org) for installation instructions
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

# ⭐ Support

If this project helps you, please **⭐ star the repo** on GitHub!

For issues, questions, or contributions, please open an issue on GitHub.


