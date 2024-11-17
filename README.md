# Overview

This project aims to segment sea turtles from photographs using computer vision and deep learning techniques. We employed two advanced segmentation models: U-Net and DeepLabv3+. The objective was to accurately identify the head, flippers, and carapace of sea turtles in the given dataset, which contains diverse images from different environmental conditions.

# Dataset

We used the SeaTurtleID2022 dataset, which includes 8,729 images spanning 13 years of observations. Each image is annotated to provide detailed segmentations of key body parts of the turtles. This dataset is crucial for training and evaluating the performance of our segmentation models.
The dataset can be downloaded from this link: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022

# Models

**1. U-Net**

**Description:** U-Net is an encoder-decoder architecture with skip connections that preserves spatial hierarchies.

**Purpose:** It was selected for its effectiveness in handling small datasets while maintaining high spatial accuracy, particularly useful for segmenting detailed anatomical features like the flippers and carapace.

**2. DeepLabv3+**

**Description:** DeepLabv3+ is an encoder-decoder model that uses Atrous Spatial Pyramid Pooling (ASPP) to extract multi-scale context and refine segmentation maps for sharp boundaries.

**Purpose:** It was chosen for its ability to handle complex and cluttered environments with high accuracy, making it ideal for segmenting multiple overlapping turtles.

# Install Dependencies

Use the following commands to install the required libraries and packages:

```bash
# Clone the repository
git clone https://github.com/arygupta04/visioneerium.git

# Navigate to the directory
cd visioneerium

# Install libraries and packages
# PyTorch and related libraries
pip install torch torchvision torchaudio

# Albumentations and image processing
pip install albumentations opencv-python-headless pillow matplotlib

# TQDM for progress bars
pip install tqdm

# Segmentation models (includes pre-trained models for PyTorch)
pip install segmentation-models-pytorch

# PyCOCOTools for working with COCO dataset annotations
pip install pycocotools

# Pandas for data handling
pip install pandas

# simplecrf library for CRF (Conditional Random Fields) post-processing
pip install simplecrf

# Timm for pre-trained models
pip install timm
```

# Running the project
In the terminal, write:

```bash
python train.py
```

**Note:** Make sure to adjust the file paths in the scripts to match your directory structure and data locations before running them.

