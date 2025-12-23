# Speeding Up AI Image Forensics: A CLIP--MLP Pipeline for Real vs. Synthetic Image Detection

This repository contains the official implementation of the paper
**"Speeding Up AI Image Forensics: A CLIP-MLP Pipeline for Real
vs. Synthetic Detection"**.\
(publication pending)
The project introduces a lightweight, efficient, and highly accurate
framework for detecting AI-generated images using frozen **CLIP ViT-L/14
embeddings** and a compact **MLP classifier**.

## Overview

Modern generative models (Midjourney, DALL·E 3, SDXL, etc.) produce
photo-realistic images that challenge existing forensic tools.\
This project provides a fast and robust detection pipeline with:

-   **Pretrained CLIP ViT-L/14** for feature extraction\
-   **Lightweight MLP classifier** (no CLIP fine-tuning required)\
-   **Fast inference** (\~162 ms/image)\
-   **High classification accuracy** (\> 98%)\
-   **Cross-domain robustness** evaluated on the **Chameleon** dataset\
-   **Generalization to common distortions** (JPEG, blur, resolution
    changes)

##  Repository Structure

    .
    ├── efficientnetv2b3.ipynb
    ├── experiments_CHAMELEON.ipynb
    ├── experiments_Image_Transformation.ipynb
    ├── performance_comparison.ipynb
    ├── requirements.txt
    └── README.md

##  Methodology

### 1. Feature Extraction --- CLIP ViT-L/14

-   Images processed using CLIP's preprocessing pipeline\
-   Converted into **768-dimensional embeddings**\
-   CLIP remains **frozen** during training

### 2. MLP Classifier

-   Layers: **512 → 256 → 256**\
-   BatchNorm + ReLU + Dropout\
-   Sigmoid output\
-   Trained 40 epochs using AdamW

### 3. Datasets

-   **AI vs Human Generated Images (Kaggle)** --- 80,000 images\
-   **Chameleon Dataset** --- high-fidelity synthetic images

## Results

### Primary Performance

  Model                        Accuracy     F1
  ---------------------------- ------------ ------------
  **CLIP + MLP**               **0.9852**   **0.9851**
  
 - CLIP + Logistic Regression   0.9774       0.9773
   
 - CLIP + XGBoost               0.9537       0.9535
   
 - EfficientNetV2B3             0.8796       0.8888

### Runtime Efficiency

  Model            Training Time   Inference
  ---------------- --------------- -----------
  **CLIP + MLP**   **22.5 min**    **2 sec**
  
  CNN Baseline     6 hours         600 sec

##  Installation

``` bash
pip install -r requirements.txt
```

The full research paper is included as:\
**realvfakeconfpaper.pdf**

## 📜 License

MIT License
