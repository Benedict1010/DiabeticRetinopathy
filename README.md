# Multimodal Diabetic Retinopathy Detection

### Combining Retinal Fundus Images and Clinical Data for Disease Severity Prediction

## Overview

This project explores a multimodal machine learning approach for diabetic retinopathy severity prediction using retinal fundus images and clinical information.

The objective is to investigate how visual features extracted from retinal images can be combined with structured clinical information to support disease severity classification.

## Problem Statement

Diabetic retinopathy is a diabetes-related eye condition that can lead to vision impairment.

Traditional image-based approaches primarily analyze retinal fundus images. This project explores combining retinal image features with clinical data to investigate a multimodal prediction approach.

## Proposed Architecture

The project follows a modular machine learning pipeline.

```text
Retinal Fundus Images       Clinical Data
        │                        │
        ▼                        ▼
 Image Preprocessing      Clinical Preprocessing
        │                        │
        ▼                        ▼
 EfficientNet Model       Clinical Feature Processing
        │                        │
        ▼                        │
 Image Feature Extraction │
        │                        │
        └──────────┬─────────────┘
                   ▼
             Feature Fusion
                   │
                   ▼
             Classification
                   │
                   ▼
          Severity Prediction
                   │
                   ▼
             Model Evaluation
```

*This represents the proposed architecture; not all stages are implemented or integrated yet.*

## Technology Stack

| Category         | Technologies                  |
| ---------------- | ----------------------------- |
| Programming      | Python                        |
| Deep Learning    | PyTorch                       |
| Image Models     | EfficientNet                  |
| Image Processing | OpenCV                        |
| Data Processing  | NumPy, Pandas                 |
| Frontend         | React, Vite                   |
| Development      | Jupyter Notebook, Git, GitHub |

## Dataset

The project references publicly available retinal fundus image datasets, including:

* APTOS 2019 Blindness Detection Dataset.
* Other suitable retinal image datasets for experimentation.

Dataset files are not included in this repository.

## Project Modules

### 1. Dataset Management

* Organizing retinal fundus images.
* Managing image labels and dataset metadata.
* Preparing data for machine learning experiments.

### 2. Image Preprocessing

* Image resizing.
* Image cropping.
* Image normalization.
* Data augmentation.

### 3. Deep Learning Model

* Exploring EfficientNet-based image feature extraction.
* Preparing the image classification pipeline.

### 4. Clinical Data Processing

* Preparing structured clinical information.
* Handling missing values and feature transformations.

### 5. Multimodal Feature Fusion

* Exploring the integration of image and clinical features.
* Preparing a combined prediction architecture.

### 6. Evaluation

* Model performance evaluation.
* Classification metrics.
* Prediction analysis.

## Repository Structure

```text
DiabeticRetinopathy/
│
├── assets/
├── frontend/
├── model/
│   ├── datasets.py
│   ├── efficientnet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
│
├── notebooks/
│
├── preprocessing/
│   ├── augment.py
│   ├── crop.py
│   ├── normalize.py
│   ├── preprocess.py
│   └── resize.py
│
└── test_install.py
```

Some modules are still under development.



## Future Improvements

* Complete the clinical preprocessing pipeline.
* Implement multimodal feature fusion.
* Complete model training and evaluation modules.
* Integrate model inference with the frontend.
* Add reproducible experiments and performance reports.
* Implement model explainability using Grad-CAM.

## Important Disclaimer

This project is developed for academic and research purposes. It is not a clinically validated diagnostic system and must not be used for medical diagnosis or treatment decisions.

## Repository

https://github.com/Benedict1010/DiabeticRetinopathy
