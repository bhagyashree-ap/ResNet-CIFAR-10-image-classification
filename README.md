# ResNet-CIFAR-10-image-classification

![Trainable Parameters](https://img.shields.io/badge/Trainable%20Parameters-4.7M-blue)
![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-97.82%25-brightgreen)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Implementation](https://img.shields.io/badge/Implementation-From%20Scratch-purple)

This repo is a PyTorch implementation of a parameter efficient CIFAR-10 image classifier using a Pre-Activation Residual Network, rotation based auxiliary supervision, and self-distillation.

The project focuses on achieving high classification accuracy while staying below a 5-million trainable parameter budget. It includes controlled training comparisons, checkpoint based recovery, GPU optimized training, and test time augmentation.

The implementation is contained in a Jupyter notebook and was trained from scratch without pretrained weights or external training datasets.

## Overview

* Employs residual shortcut connections to improve gradient flow and feature propagation.
* Combines primary classification with auxiliary rotation prediction during training.
* Uses a multi-objective optimization strategy combining classification, auxiliary supervision, and KL-divergence losses.
* Applies strong data augmentation techniques:
    * Random Cropping
    * Random Horizontal Flipping
    * TrivialAugmentWide
    * Cutout
* Implements a Cosine Learning Rate Scheduler with five epochs of linear warm-up followed by cosine decay.
* Uses FP16 Automatic Mixed Precision (AMP) to improve GPU training efficiency.
* Optimizes memory usage through channels-last memory formatting.
* Supports checkpoint-based training recovery to resume interrupted experiments.
* Includes controlled training comparisons to evaluate the contribution of different training strategies.
* Combines predictions from six deterministic transformations during inference using Test-Time Augmentation (TTA).
* Reports 4,706,674 trainable parameters and 97.82% validation accuracy.

## Dataset

Contains: 
* Labeled training images - 50,000
* Unlabeled evaluation images - 2,000
* Image resolution - 32 × 32
* Color channels - 3 (RGB)
* Number of classes - 10

![Airplane](https://img.shields.io/badge/Class-Airplane-green)
![Automobile](https://img.shields.io/badge/Class-Automobile-green)
![Bird](https://img.shields.io/badge/Class-Bird-green)
![Cat](https://img.shields.io/badge/Class-Cat-green)
![Deer](https://img.shields.io/badge/Class-Deer-green)
![Dog](https://img.shields.io/badge/Class-Dog-green)
![Frog](https://img.shields.io/badge/Class-Frog-green)
![Horse](https://img.shields.io/badge/Class-Horse-green)
![Ship](https://img.shields.io/badge/Class-Ship-green)
![Truck](https://img.shields.io/badge/Class-Truck-green)

## Validation split

The notebook creates a fixed, stratified validation split.

* Training - 45,000 Images
* Validation - 5,000 Images
* Total labeled data - 50,000 Images

The split uses NumPy seed 1123 and selects 500 validation images per class. This gives every class equal representation in the validation set and makes the 200-epoch and 300-epoch comparisons consistent. After selecting the best training schedule, the notebook retrains the model using all 50,000 labeled images.

## Architecture

The model uses a shared convolutional feature extractor with two output heads. The main classification head predicts the original image's class. The auxiliary SLA head predicts the joint class-and-rotation label.

<img width="400" height="565" alt="image" src="https://github.com/user-attachments/assets/e154232d-e07d-4ec7-9840-0ee55cfdcca4" />

The main classifier learns the standard image classification task. The auxiliary classifier receives rotated versions of the same image and learns to predict both the object category and the rotation angle. This encourages the shared feature extractor to learn more informative representations. The auxiliary head also provides a teacher signal to the main classifier through KL-divergence-based self-distillation. During inference, both heads contribute to the final prediction.

## Model Architecture
The network is a CIFAR-sized Pre-Activation ResNet with three residual stages.

### Architecture Configuration

| Component | Configuration |
|:---|:---|
| Input | `3 × 32 × 32` |
| Initial Convolution | `3 × 3, 64 channels, stride 1` |
| Stage 1 | `4 Residual Blocks, 64 channels` |
| Stage 2 | `4 Residual Blocks, 128 channels` |
| Stage 3 | `3 Residual Blocks, 256 channels` |
| Downsampling | `Stride 2` at the first block of Stages 2 and 3 |
| Normalization | BatchNorm |
| Activation | ReLU |
| Pooling | Adaptive Average Pooling |
| Feature Dimension | `256` |
| Main Classification Head | `256 → 10` |
| SLA Classification Head | `256 → 40` |
| Trainable Parameters | **4,706,674** |

The network contains 11 residual blocks in total.

### Feature extraction

The input image passes through the initial convolution and the three residual stages. The spatial resolution decreases as the number of feature channels increases.

<img width="578" height="498" alt="generated-image" src="https://github.com/user-attachments/assets/c2e91821-e5bb-4f14-be78-d38d3ac56c5d" />
