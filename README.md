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

# Install the required packages
pip install -r requirements.txt
```

# Download the Dataset

Download the SeaTurtleID2022 dataset from the following link: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022. Extract the contents of the zip file into the `data` directory. Either keep at the same level as the `src` directory or update the paths in the scripts accordingly.

# Running the project

In the terminal, write:

```bash
python -m src.unet.train
```

If you want to train the DeepLabv3+ model, run:

```bash
python -m src.deeplabv3.train
```
