Of course! Here is an English version of your README.md file, professionally formatted and ready for GitHub. I've translated the content, improved the structure for clarity, and added some common elements like badges to make it look polished.

---

# SAM-YOLO Auto Annotator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Ultralytics-orange.svg)](https://pytorch.org/)

An automated data annotation and training tool that combines Meta's **Segment Anything Model 2 (SAM 2)** with **YOLOv8**. This project is designed to accelerate the dataset creation and model training pipeline for visuo-tactile sensors or any other computer vision segmentation task.

## Key Features

*   **Semi-Automatic Annotation**: Leverage the power of SAM 2 to generate high-quality segmentation masks with just a few clicks.
*   **Integrated YOLOv8 Training**: Seamlessly convert annotated data into YOLO format and initiate the training process.
*   **Interactive Interface**: A simple graphical interface for collecting samples and fine-tuning annotations with point-based prompts.

## Directory Structure

```
sam-yolo-annotator/
├── README.md                 # Project documentation
├── sam2_annotator.py         # Core script for SAM 2-assisted annotation
├── train_yolo_seg.py         # Script for training the YOLOv8 segmentation model
├── dataset/                  # Dataset directory
│   ├── data.yaml             # YOLO dataset configuration file
│   ├── raw_images/           # Raw images collected from the camera
│   ├── images/               # Processed images for training
│   └── labels/               # Training labels in YOLO .txt format
├── runs/                     # Directory for training logs and model outputs
├── sam2/                     # SAM 2 library and model weights
└── sam3/                     # Configuration files for the SAM 2 environment
```

## Environment Setup

This project requires two separate Python environments to avoid dependency conflicts.

1.  **Annotation Environment (`sam3`)**: For running SAM 2.
2.  **Training Environment (`yolo_training`)**: For training the YOLOv8 model.

### 1. Configure SAM 2 Environment

Please refer to the official SAM 2 documentation to install its dependencies. A setup with PyTorch and CUDA is highly recommended.

```bash
# Example: Create and activate a new conda environment
conda create -n sam3 python=3.10
conda activate sam3

# Install required dependencies (e.g., PyTorch, torchvision, etc.)
# pip install ...
```
**Note**: The environment is named `sam3` for historical reasons, but it is set up to run SAM 2. You can also adapt it for SAM 3, which may require applying for access on Hugging Face.

Before the first use, you must download the SAM 2 model weights:
```bash
cd sam2/checkpoints
./download_ckpts.sh
```

### 2. Configure YOLO Training Environment
Create and activate a separate environment for training.

```bash
# Example using conda
conda create -n yolo_training python=3.10
conda activate yolo_training

# Install Ultralytics for YOLOv8
pip install ultralytics
```

## Usage Guide

### Step 1: Data Collection

Collect raw images from a connected camera.

```bash
# Activate the SAM environment
conda activate sam3

# Run the collection script
python sam2_annotator.py collect --camera 0 --output ./dataset/raw_images
```
**Controls:**
*   `Spacebar`: Save the current frame.
*   `q`: Quit.

### Step 2: Interactive Annotation

Use the SAM 2-powered annotator to create segmentation masks.

```bash
# Make sure the SAM environment is active
conda activate sam3

# Run the annotation script
python sam2_annotator.py annotate --input ./dataset/raw_images --output ./dataset
```
**Controls:**
*   **Left-click**: Add a **foreground point** to include an area in the mask.
*   **Right-click**: Add a **background point** to exclude an area from the mask.
*   **`s`**: Save the current annotation and move to the next image.
*   **`n` / `p`**: Navigate to the **n**ext / **p**revious image without saving.
*   **`r`**: **R**eset all points for the current image.
*   **`q`**: Quit the annotator.

The script will automatically save the annotated image to `dataset/images/` and the corresponding label to `dataset/labels/` in YOLO format.

### Step 3: Model Training

Switch to the training environment and start the YOLOv8 training process. A new experiment folder (e.g., `tactile_seg2`, `tactile_seg3`, ...) will be created under `runs/segment/` for each training run.

```bash
# Activate the YOLO environment
conda activate yolo_training

# Run the training script
python train_yolo_seg.py
```

### Step 4: Deploy the Model

After training is complete, the script will print the path to the best-performing model (e.g., `runs/segment/tactile_segX/weights/best.pt`). Manually copy this model file to your target application project.

**Example:**
```bash
# Copy the best model to a robotics project in the GIP_ws workspace
cp runs/segment/tactile_seg10/weights/best.pt /home/user/GIP_ws/src/perception/best.pt
```

## License

This project is licensed under the [MIT License](LICENSE).