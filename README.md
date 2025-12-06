
# 🧠 Brain Tumor Detection & Classification using Deep Learning  
Automated MRI Analysis Using CNN, RNN-LSTM, VGG16, and YOLOv8

This project provides a comprehensive deep-learning pipeline for analyzing Brain Tumors (BT) from MRI scans. The system can classify brain MRI images into 4 categories:

- **NO_TUMOR**: Healthy brain (no tumor detected)
- **GLIOMA**: Glioma tumor type
- **MENINGIOMA**: Meningioma tumor type
- **PITUITARY**: Pituitary tumor type

## 🎯 Project Overview

This project implements multiple deep learning models for brain tumor classification:

1. **CNN (Convolutional Neural Network)** - Custom CNN architecture for image classification
2. **RNN-LSTM (Recurrent Neural Network with LSTM)** - Hybrid CNN-LSTM model combining spatial feature extraction with sequential processing
3. **VGG16 (Transfer Learning)** - Pre-trained VGG16 model for classification
4. **YOLOv8** - Object detection and localization with bounding boxes

The system provides comprehensive evaluation metrics including confusion matrices, classification reports, and training visualizations for reporting purposes.

## 🔄 Project Workflow

```
1. Data Preparation
   └──> Organize raw MRI images into class folders
   └──> Run preprocessing script/notebook
   └──> Generate train/val/test splits

2. Model Training (Choose one or both)
   ├──> train_cnn.ipynb
   │    └──> Train CNN model
   │    └──> Generate evaluation metrics
   │    └──> Save model and visualizations
   │
   └──> train_rnn_lstm.ipynb
        └──> Train CNN-LSTM hybrid model
        └──> Generate evaluation metrics
        └──> Save model and visualizations

3. Model Evaluation
   └──> Compare test accuracy
   └──> Review confusion matrices
   └──> Analyze classification reports
   └──> Compare training curves

4. Reporting
   └──> Use generated plots and metrics
   └──> Include in research reports/presentations
```

---

## 📌 Features  
- ✔ **CNN Model** - Custom convolutional neural network for brain tumor classification
- ✔ **RNN-LSTM Hybrid Model** - CNN feature extractor + LSTM sequential processing
- ✔ **VGG16 Transfer Learning** - Pre-trained model for classification
- ✔ **YOLOv8** - Tumor detection and localization with bounding boxes
- ✔ Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- ✔ Confusion matrices and classification reports
- ✔ Training/validation curves visualization
- ✔ Sample predictions visualization
- ✔ GPU-accelerated training support (CUDA)
- ✔ Jupyter notebook-based training pipeline
- ✔ Cross-platform support (Windows & Linux)  

---

## 📁 Project Structure

```
Deep-MRIC/
├── data/
│   ├── raw_dataset/              # Raw MRI images organized by class
│   │   ├── NO_TUMOR/
│   │   ├── GLIOMA/
│   │   ├── MENINGIOMA/
│   │   └── PITUITARY/
│   ├── vgg16_classification/     # Preprocessed images (generated after preprocessing)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── dataset_metadata.csv      # Dataset metadata with paths and labels
│   └── yolov8/                   # YOLOv8 dataset (if using detection)
│       ├── images/
│       └── labels/
├── models/                        # Saved model checkpoints (generated after training)
│   ├── cnn_brain_tumor_classifier.pth
│   ├── rnn_lstm_brain_tumor_classifier.pth
│   ├── cnn_training_history.csv
│   ├── rnn_lstm_training_history.csv
│   └── [evaluation plots and reports]
├── preprocess_vgg16.py           # Python script for data preprocessing
├── preprocess_vgg16.ipynb         # Jupyter notebook for interactive preprocessing
├── train_cnn.ipynb                # CNN model training notebook
├── train_rnn_lstm.ipynb          # RNN-LSTM model training notebook
├── requirements.txt               # pip requirements (Windows & Linux compatible)
├── environment.yml                # conda environment file (Windows & Linux compatible)
└── README.md
```

---

# 🚀 Getting Started

## 1️⃣ Prerequisites  

### System Requirements
- **Python**: 3.8 to 3.11 (3.8+ recommended)
- **Operating System**: Windows 10/11 or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: At least 5GB free space for datasets and models
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster training, but CPU training is also supported)

### Software Requirements
- **Git** - For cloning the repository
- **pip** or **conda** - Package manager
- **Jupyter Notebook** - For running training notebooks (included in installation)

### GPU Setup (Optional but Recommended)
- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **CUDA Toolkit** 11.0 or higher
- **cuDNN** (included with PyTorch installation)

---

## 2️⃣ Installation

### Option A: Using pip (Recommended for most users)

#### Windows Installation:

1. **Open Command Prompt or PowerShell** (Run as Administrator if needed)

2. **Clone the repository:**
```cmd
git clone https://github.com/chaklader17/Deep-MRIC.git
cd Deep-MRIC
```

3. **Create virtual environment:**
```cmd
python -m venv venv
```

4. **Activate virtual environment:**
```cmd
venv\Scripts\activate
```

5. **Upgrade pip (recommended):**
```cmd
python -m pip install --upgrade pip
```

6. **Install PyTorch with CUDA (if you have NVIDIA GPU):**
   - Visit [PyTorch Installation](https://pytorch.org/get-started/locally/)
   - Select your CUDA version and copy the installation command
   - Example for CUDA 11.8:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

7. **Install other dependencies:**
```cmd
pip install -r requirements.txt
```

#### Linux Installation:

1. **Open Terminal**

2. **Clone the repository:**
```bash
git clone https://github.com/chaklader17/Deep-MRIC.git
cd Deep-MRIC
```

3. **Create virtual environment:**
```bash
python3 -m venv venv
```

4. **Activate virtual environment:**
```bash
source venv/bin/activate
```

5. **Upgrade pip:**
```bash
python -m pip install --upgrade pip
```

6. **Install PyTorch with CUDA (if you have NVIDIA GPU):**
   - Visit [PyTorch Installation](https://pytorch.org/get-started/locally/)
   - Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

7. **Install other dependencies:**
```bash
pip install -r requirements.txt
```

### Option B: Using conda (Recommended for data science workflows)

#### Windows Installation:

1. **Open Anaconda Prompt or Command Prompt**

2. **Clone the repository:**
```cmd
git clone https://github.com/chaklader17/Deep-MRIC.git
cd Deep-MRIC
```

3. **Create conda environment:**
```cmd
conda env create -f environment.yml
```

4. **Activate the environment:**
```cmd
conda activate deep-mric
```

5. **Install PyTorch with CUDA (if needed):**
```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Linux Installation:

1. **Open Terminal**

2. **Clone the repository:**
```bash
git clone https://github.com/chaklader17/Deep-MRIC.git
cd Deep-MRIC
```

3. **Create conda environment:**
```bash
conda env create -f environment.yml
```

4. **Activate the environment:**
```bash
conda activate deep-mric
```

5. **Install PyTorch with CUDA (if needed):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Verify Installation

#### Windows:
```cmd
# Check Python version (should be 3.8+)
python --version

# Check if key packages are installed
python -c "import cv2, numpy, sklearn, tqdm, torch, torchvision, pandas, seaborn; print('✅ All packages installed successfully!')"

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# For Jupyter notebook support
jupyter --version
```

#### Linux:
```bash
# Check Python version (should be 3.8+)
python3 --version

# Check if key packages are installed
python3 -c "import cv2, numpy, sklearn, tqdm, torch, torchvision, pandas, seaborn; print('✅ All packages installed successfully!')"

# Check PyTorch and CUDA
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# For Jupyter notebook support
jupyter --version
```

---

## 3️⃣ Data Preprocessing

Before training any model, you need to preprocess your raw MRI images. The preprocessing script will:
- Crop brain regions (remove black background using contour detection)
- Resize images to 224×224 (standard input size for deep learning models)
- Perform stratified train/val/test split (70/15/15) to ensure balanced class distribution
- Create organized directory structure for easy data loading

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

**Windows:**
```cmd
# Make sure virtual environment is activated
venv\Scripts\activate

# Run preprocessing script
python preprocess_vgg16.py
```

**Linux:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run preprocessing script
python3 preprocess_vgg16.py
```

#### Option B: Using Jupyter Notebook (Interactive)

**Windows:**
```cmd
# Activate virtual environment
venv\Scripts\activate

# Start Jupyter Notebook
jupyter notebook

# Open preprocess_vgg16.ipynb and run all cells
```

**Linux:**
```bash
# Activate virtual environment
source venv/bin/activate

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
- `RAW_DATA_DIR`: Input directory path (default: `data/raw_dataset/`)
- `VGG_OUTPUT_DIR`: Output directory path (default: `data/vgg16_classification/`)
- `SPLIT_RATIO`: Train/Val/Test split ratios (default: [0.70, 0.15, 0.15])
- `CLASSES`: List of class names (default: ['NO_TUMOR', 'GLIOMA', 'MENINGIOMA', 'PITUITARY'])

**After preprocessing**, you'll have:
- Organized train/val/test splits in `data/vgg16_classification/`
- `data/dataset_metadata.csv` with image paths and labels
- Ready-to-use dataset for training

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

## How the Project Works

### Workflow Overview:

1. **Data Preprocessing** → Organize and preprocess raw MRI images
2. **Model Training** → Train CNN and/or RNN-LSTM models using Jupyter notebooks
3. **Model Evaluation** → Generate confusion matrices, classification reports, and visualizations
4. **Model Comparison** → Compare performance metrics between different models

### Training Process:

The training notebooks (`train_cnn.ipynb` and `train_rnn_lstm.ipynb`) follow this process:

1. **Data Loading**: Load preprocessed images from `data/vgg16_classification/`
2. **Data Augmentation**: Apply transformations (rotation, flipping, etc.) for training
3. **Model Definition**: Define CNN or CNN-LSTM architecture
4. **Training Loop**: Train model with validation monitoring
5. **Model Saving**: Save best model based on validation accuracy
6. **Evaluation**: Test on test set and generate metrics
7. **Visualization**: Create plots for training curves, confusion matrices, and sample predictions

---

## 1️⃣ Train CNN Model

**Important:** Make sure you've run the preprocessing script first (see [Data Preprocessing](#3️⃣-data-preprocessing) section).

### Windows:

1. **Activate your virtual environment:**
```cmd
venv\Scripts\activate
```

2. **Start Jupyter Notebook:**
```cmd
jupyter notebook
```

3. **Open `train_cnn.ipynb`** in the browser

4. **Run all cells** (Cell → Run All) or run cells sequentially

### Linux:

1. **Activate your virtual environment:**
```bash
source venv/bin/activate
```

2. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

3. **Open `train_cnn.ipynb`** in the browser

4. **Run all cells** (Cell → Run All) or run cells sequentially

### What the Notebook Does:

- Loads and preprocesses data with augmentation
- Defines a custom CNN architecture (4 convolutional blocks + fully connected layers)
- Trains the model with early stopping and learning rate scheduling
- Evaluates on test set and generates:
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - Training/validation curves
  - Sample predictions visualization
- Saves model to `models/cnn_brain_tumor_classifier.pth`

---

## 2️⃣ Train RNN-LSTM Model

The RNN-LSTM notebook trains a hybrid CNN-LSTM model that combines:
- **CNN layers** for spatial feature extraction
- **LSTM layers** for sequential processing of features

### Windows:

1. **Activate your virtual environment:**
```cmd
venv\Scripts\activate
```

2. **Start Jupyter Notebook:**
```cmd
jupyter notebook
```

3. **Open `train_rnn_lstm.ipynb`** in the browser

4. **Run all cells** (Cell → Run All)

### Linux:

1. **Activate your virtual environment:**
```bash
source venv/bin/activate
```

2. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

3. **Open `train_rnn_lstm.ipynb`** in the browser

4. **Run all cells** (Cell → Run All)

### What the Notebook Does:

- Loads and preprocesses data (same as CNN notebook)
- Defines CNN-LSTM hybrid architecture:
  - CNN feature extractor (3 convolutional blocks)
  - LSTM layers to process CNN features as sequences
  - Fully connected layers for classification
- Trains the model with the same training pipeline as CNN
- Generates the same evaluation metrics and visualizations
- Saves model to `models/rnn_lstm_brain_tumor_classifier.pth`

### Model Comparison:

Both notebooks generate comprehensive reports that can be compared:
- Test accuracy
- Per-class metrics (precision, recall, F1-score)
- Confusion matrices
- Training curves

---

## 3️⃣ Train VGG16 — Classification (Legacy)

**Note:** This section is for the original VGG16 training script. The CNN and RNN-LSTM notebooks are the recommended approach.

**Important:** Make sure you've run the preprocessing script first.

### Windows:
```cmd
python scripts/classify_vgg16.py --data_dir data/vgg16_classification --epochs 30 --batch_size 64 --learning_rate 1e-4 --model_save_path models/vgg16_classifier_best.pth
```

### Linux:
```bash
python scripts/classify_vgg16.py \
    --data_dir data/vgg16_classification \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --model_save_path models/vgg16_classifier_best.pth
```

---

## 4️⃣ Train YOLOv8 — Detection (Optional)

YOLOv8 is used for tumor detection and localization with bounding boxes.

### Windows:
```cmd
yolo task=detect mode=train model=yolov8n.pt data=tumor_data.yaml epochs=50 imgsz=640 name=yolov8_tumor_detection
```

### Linux:
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

## Using Trained Models

After training, you can use the saved models for inference on new images. The training notebooks include evaluation on the test set, but you can also load the models for custom inference.

### Loading and Using Trained Models

**Example code (can be added to notebooks):**

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load trained CNN model
model = BrainTumorCNN(num_classes=4)
checkpoint = torch.load('models/cnn_brain_tumor_classifier.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path/to/image.jpg')
image_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}% confidence)")
```

---

## YOLOv8 Detection (Optional)

### Windows:
```cmd
yolo task=detect mode=predict model=runs/detect/yolov8_tumor_detection/weights/best.pt source=data/test_images/sample_mri.png conf=0.25
```

### Linux:
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

## Model Evaluation Metrics

Both CNN and RNN-LSTM notebooks automatically generate comprehensive evaluation metrics:

### Classification Metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro/micro averages
- **Recall**: Per-class and macro/micro averages
- **F1-Score**: Per-class and macro/micro averages
- **Confusion Matrix**: Visual heatmap showing classification performance
- **Classification Report**: Detailed per-class metrics saved to text file

### Visualizations Generated:
- **Training Curves**: Loss and accuracy plots for training and validation
- **Confusion Matrix**: Heatmap visualization saved as PNG
- **Sample Predictions**: 10 sample test images with predictions and confidence scores

### Output Files:

After training, the following files are saved in the `models/` directory:

**CNN Model:**
- `cnn_brain_tumor_classifier.pth` - Trained model checkpoint
- `cnn_training_history.csv` - Training history (loss, accuracy per epoch)
- `cnn_training_curves.png` - Training/validation curves
- `cnn_confusion_matrix.png` - Confusion matrix visualization
- `cnn_classification_report.txt` - Detailed classification report
- `cnn_sample_predictions.png` - Sample predictions visualization

**RNN-LSTM Model:**
- `rnn_lstm_brain_tumor_classifier.pth` - Trained model checkpoint
- `rnn_lstm_training_history.csv` - Training history
- `rnn_lstm_training_curves.png` - Training/validation curves
- `rnn_lstm_confusion_matrix.png` - Confusion matrix visualization
- `rnn_lstm_classification_report.txt` - Detailed classification report
- `rnn_lstm_sample_predictions.png` - Sample predictions visualization

### Comparing Models:

To compare CNN vs RNN-LSTM performance:
1. Train both models using their respective notebooks
2. Check the test accuracy printed in each notebook
3. Compare confusion matrices (saved as PNG files)
4. Review classification reports (saved as text files)
5. Compare training curves to see convergence behavior

## YOLOv8 Metrics (Optional)

For YOLOv8 detection model:
* mAP@0.5
* mAP@0.5:0.95
* IoU
* Precision-Recall curves

### Run YOLO evaluation:

**Windows:**
```cmd
yolo mode=val model=runs/detect/yolov8_tumor_detection/weights/best.pt data=tumor_data.yaml
```

**Linux:**
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
- **torch** (≥1.12.0) - PyTorch framework for deep learning
- **torchvision** (≥0.13.0) - PyTorch vision utilities and datasets
- **pandas** (≥1.3.0) - Data manipulation and analysis
- **seaborn** (≥0.12.0) - Statistical data visualization (for confusion matrices)
- **ultralytics** - YOLOv8 implementation (optional, for detection)

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

## Model Performance

After training, you can compare the performance of different models:

| Model      | Architecture | Task           | Output Files                      |
| ---------- | ------------ | -------------- | --------------------------------- |
| **CNN**    | Custom CNN   | Classification | `models/cnn_*` files              |
| **RNN-LSTM** | CNN-LSTM Hybrid | Classification | `models/rnn_lstm_*` files         |
| **VGG16**  | Transfer Learning | Classification | Model checkpoint files            |
| **YOLOv8** | Object Detection | Detection      | `runs/detect/` directory          |

### Typical Performance Metrics:

- **Accuracy**: Both CNN and RNN-LSTM models typically achieve high accuracy (>85%)
- **Per-Class Metrics**: Detailed precision, recall, and F1-score for each tumor type
- **Training Time**: CNN typically trains faster than RNN-LSTM due to simpler architecture
- **Model Size**: CNN models are generally smaller than RNN-LSTM models

### Reporting:

All generated files (confusion matrices, classification reports, training curves) can be directly used in research reports and presentations.

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
- **Windows**: Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- **Linux**: Verify GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`
- If CUDA is not available, training will automatically use CPU (slower but functional)

### Issue: Jupyter notebook kernel not found
**Solution:**
- **Windows**: `python -m ipykernel install --user --name=deep-mric`
- **Linux**: `python3 -m ipykernel install --user --name=deep-mric`
- Restart Jupyter notebook and select the kernel

### Issue: Out of memory during training
**Solution:**
- Reduce `BATCH_SIZE` in the notebook configuration (try 16 or 8)
- Close other applications to free up GPU/CPU memory
- Use CPU training if GPU memory is insufficient (slower but works)

### Issue: Training is very slow
**Solution:**
- Ensure CUDA is properly installed and GPU is being used
- Check GPU utilization: `nvidia-smi` (Linux) or Task Manager → Performance (Windows)
- Reduce batch size if memory constrained
- Consider using a smaller model architecture

---

# ⭐ Support

If this project helps you, please **⭐ star the repo** on GitHub!

For issues, questions, or contributions, please open an issue on GitHub.


