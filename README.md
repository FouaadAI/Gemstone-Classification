# Gemstone Classification Using Deep Learning

This repository contains a deep learning-based image classification project that identifies various types of gemstones using a convolutional neural network (ResNet50). The project is implemented in Python using PyTorch and is designed to run in Google Colab.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Requirements](#requirements)
- [Author](#author)

---

## Overview

The goal of this project is to classify gemstone images into one of 87 categories. It uses a pretrained ResNet50 model with fine-tuning, early stopping, and evaluation metrics such as accuracy and confusion matrices to measure performance.

---

## Dataset

The dataset is organized into three directories:
- /train: Training images organized in subfolders by class.
- /test: Testing images organized similarly.
- /validation: (optional) Used if not using a split from training data.

*Note*: The dataset must be uploaded as a ZIP archive and is automatically extracted within the notebook.

---

## Model Architecture

- *Base Model*: ResNet50 pretrained on ImageNet.
- *Modifications*: Replaced the final fully connected layer to match the number of gemstone classes (87).
- *Loss Function*: CrossEntropyLoss
- *Optimizer*: Adam
- *Device*: Automatically uses GPU if available.

---

## Training Details

- *Image Size*: 224x224
- *Batch Size*: 32
- *Epochs*: 50
- *Learning Rate*: 1e-4
- *Early Stopping*: Stops if no improvement for 5 consecutive epochs.
- *Best Model*: Automatically saved as best_model.pt.

---

## Evaluation

The model is evaluated on the test set using:
- Final test accuracy
- Full confusion matrix
- Reduced confusion matrix for the top 10 most confused classes
- Visual plots of training/validation loss and accuracy

---

## How to Use

1. Upload the dataset ZIP file to the Colab environment.
2. Run the notebook cell-by-cell.
3. Use the interactive widget to upload custom gemstone images and receive predictions.
4. View results via plots and confusion matrices.

---

## Results

- *Test Accuracy*: ~X.XX% (Replace with your final accuracy)
- *Top Confused Classes*: Displayed in a heatmap using seaborn.
- *Live Demo*: Image upload tool inside the notebook allows real-time classification.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- tqdm
- seaborn
- scikit-learn
- ipywidgets (for image upload tool)

You can install required libraries in Colab using:
```bash
!pip install torch torchvision matplotlib tqdm seaborn scikit-learn ipywidgets
