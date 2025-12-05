# Deep-MRIC
🧠 Brain Tumor Detection and Classification using CNNs and YOLOv8Project OverviewThis project focuses on the automated analysis of Brain Tumors (BT) from Magnetic Resonance Imaging (MRI) scans using established and state-of-the-art deep learning architectures. The approach is dual-faceted:Classification: Using the VGG16 architecture via Transfer Learning to classify the presence and type of tumor.Detection/Localization: Employing YOLOv8 (You Only Look Once, v8) to accurately localize the tumor within the MRI scan by drawing a bounding box around the region of interest.AttributeDetailTopicBrain Tumor Classification and LocalizationArchitecturesVGG16 (Classification), YOLOv8 (Detection/Localization)Dataset SourceKaggle Brain Tumor MRI Classification/Detection DatasetPrimary GoalHigh accuracy in both tumor classification and bounding box localization.🚀 Getting StartedPrerequisitesEnsure you have a modern Python environment and the required hardware setup.Python 3.8+NVIDIA GPU (Recommended for accelerated training)CUDA ToolkitInstallationClone the repository and install the necessary dependencies. Note that YOLOv8 requires the ultralytics library.

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Install all dependencies (see requirements.txt below)
pip install -r requirements.txt
Dataset SetupDownload the Kaggle Brain Tumor MRI Classification/Detection Dataset.For VGG16 (Classification): Place the labeled images in a folder structure like data/classification/train/class_A/ and data/classification/val/class_B/.For YOLOv8 (Detection): The data must be in the YOLO format, typically with images in one folder and corresponding .txt annotation files (with coordinates: class_id x_center y_center width height) in another. Place this structure under data/yolov8/.💻 Running the ExperimentsThe project consists of two distinct machine learning pipelines.1. VGG16 for Classification (Type of Tumor/Presence)This script fine-tunes a pre-trained VGG16 model on your classification dataset.Bash# Run the training script for VGG16 Classification
python classify_vgg16.py \
    --data_dir data/classification \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --model_save_path models/vgg16_classifier_best.pth
2. YOLOv8 for Tumor Detection and LocalizationThis script utilizes the powerful ultralytics library to train the YOLOv8 model to draw precise bounding boxes around the tumor area, providing localization.Bash# Run the training script for YOLOv8 Detection
# Note: You may need a YAML configuration file (e.g., tumor_data.yaml) 
# pointing to your image and label directories for YOLOv8 training.
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/yolov8/tumor_data.yaml \
    epochs=50 \
    imgsz=640 \
    name=yolov8_tumor_detection_run
3. Inference and VisualizationTo test the trained YOLOv8 model on new images and visualize the detected bounding boxes:Bash# Run inference using the best trained YOLOv8 model weights
yolo task=detect mode=predict \
    model=runs/detect/yolov8_tumor_detection_run/weights/best.pt \
    source=data/test_images/sample_mri.png \
    conf=0.25
📊 Evaluation and MetricsThe project tracks separate metrics for the two tasks:VGG16 Classification MetricsAccuracyPrecision, Recall, F1-Score (Crucial for class imbalance)Confusion MatrixYOLOv8 Detection MetricsmAP (mean Average Precision) @ 0.5 (Standard metric for object detection)mAP @ 0.5:0.95 (More stringent metric covering a range of IoU thresholds)IoU (Intersection over Union)🛠️ Required Dependencies (requirements.txt)Copy and paste the following into a file named requirements.txt:# Deep Learning Frameworks
torch>=1.12.0
torchvision

# YOLOv8 Library
ultralytics

# Performance Metrics and Utilities
scikit-learn
matplotlib
numpy
pandas
tqdm
