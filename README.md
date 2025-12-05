Here is the **entire README as ONE clean markdown block**, ready to copy-paste into GitHub with **no breaks, no meta text, no extra notes**.

---

```markdown
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

brain-tumor-detection/
├── data/
│   ├── classification/
│   │   ├── train/
│   │   └── val/
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
├── tumor_data.yaml
├── requirements.txt
└── README.md

````

---

# 🚀 Getting Started

## 1️⃣ Prerequisites  
- Python **3.8+**  
- NVIDIA GPU with CUDA support (recommended)  
- pip + virtual environment  

---

## 2️⃣ Installation

```bash
git clone https://github.com/<your_username>/<your_repo>.git
cd <your_repo>

python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
````

---

# 📦 Dataset Setup

## A) VGG16 Classification

Organize MRI images into:

```
data/classification/
├── train/
│   ├── meningioma/
│   ├── glioma/
│   └── pituitary/
└── val/
    ├── meningioma/
    ├── glioma/
    └── pituitary/
```

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

```bash
python scripts/classify_vgg16.py \
    --data_dir data/classification \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --model_save_path models/vgg16_classifier_best.pth
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

```
torch>=1.12.0
torchvision
ultralytics
scikit-learn
matplotlib
numpy
pandas
tqdm
```

---

# 🧪 Results

| Model      | Task           | Best Metric                       |
| ---------- | -------------- | --------------------------------- |
| **VGG16**  | Classification | High accuracy & F1-score          |
| **YOLOv8** | Detection      | High mAP & precise bounding boxes |

Add your own screenshots, training curves, and predictions.

---

# 🤝 Contributing

Pull requests are welcome.

---

# 📜 License

This project is free for research and educational use.

---

# ⭐ Support

If this project helps you, please **⭐ star the repo** on GitHub!

```

---

If you want, I can also generate a **banner image**, a **GIF demo**, or **sample output plots** for your README.
```
