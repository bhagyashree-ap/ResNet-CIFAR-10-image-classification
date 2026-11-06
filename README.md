# ResNet-CIFAR-10 Image Classification

![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-blue)
![Architecture](https://img.shields.io/badge/Architecture-PreAct_ResNet-blue)
![Trainable Parameters](https://img.shields.io/badge/Trainable_Parameters-4.7M-informational)
![Validation Accuracy](https://img.shields.io/badge/Validation_Accuracy-97.82%25-brightgreen)

![Auxiliary Supervision](https://img.shields.io/badge/Auxiliary_Task-Rotation_SLA-blueviolet)
![Self-Distillation](https://img.shields.io/badge/Self_Distillation-KL_Divergence-purple)
![Training](https://img.shields.io/badge/Training-From_Scratch-orange)
![Inference](https://img.shields.io/badge/Inference-6_View_TTA-red)

![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![GPU](https://img.shields.io/badge/GPU-NVIDIA_A100_40GB-green)
![Precision](https://img.shields.io/badge/Precision-FP16_AMP-yellow)
![Epochs](https://img.shields.io/badge/Epochs-300-lightgrey)

This repo is a PyTorch implementation of a parameter efficient CIFAR-10 image classifier using a Pre-Activation Residual Network, rotation based auxiliary supervision and self-distillation. The project focuses on achieving high classification accuracy while staying below a 5 million trainable parameter budget. The implementation is contained in a Jupyter notebook and was trained from scratch without pretrained weights or external training datasets.

## Overview

* Combines primary classification with auxiliary rotation prediction during training.
* Uses a multi-objective optimization strategy combining classification, auxiliary supervision and KL-divergence losses.
* Applies strong data augmentation techniques:
    * Random Cropping
    * Random Horizontal Flipping
    * TrivialAugmentWide
    * Cutout
* Implements a Cosine Learning Rate Scheduler with five epochs of linear warm-up followed by cosine decay.
* Uses FP16 Automatic Mixed Precision (AMP) to improve GPU training efficiency.
* Supports checkpoint-based training recovery to resume interrupted experiments.
* Combines predictions from six deterministic transformations during inference using Test-Time Augmentation (TTA).

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

### Validation Split

The notebook creates a fixed, stratified validation split.

* Training - 45,000 Images
* Validation - 5,000 Images
* Total labeled data - 50,000 Images

The split uses NumPy seed 1123 and selects 500 validation images per class. This gives every class equal representation in the validation set and makes the 200-epoch and 300-epoch comparisons consistent. After selecting the best training schedule, the notebook retrains the model using all 50,000 labeled images.

## Architecture

The model uses a shared convolutional feature extractor with two output heads. The main classification head predicts the original image's class. The auxiliary SLA head predicts the joint class-and-rotation label.

<img width="400" height="565" alt="image" src="https://github.com/user-attachments/assets/e154232d-e07d-4ec7-9840-0ee55cfdcca4" />

The auxiliary classifier receives rotated versions of the same image and learns to predict both the object category and the rotation angle. The auxiliary head also provides a teacher signal to the main classifier through KL-divergence-based self-distillation. During inference, both heads contribute to the final prediction.

The network is a CIFAR-sized Pre-Activation ResNet with three residual stages.

### Configuration

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

### Feature Extraction

The input image passes through the initial convolution and the three residual stages. The spatial resolution decreases as the number of feature channels increases.

<img width="578" height="498" alt="generated-image" src="https://github.com/user-attachments/assets/c2e91821-e5bb-4f14-be78-d38d3ac56c5d" />

### Global Average Pooling

Instead of flattening the complete final feature map, the network applies adaptive average pooling.

For a feature map (X) with spatial dimensions (H \times W), global average pooling produces:

<img width="203" height="85" alt="image" src="https://github.com/user-attachments/assets/039a954e-7a7b-4e85-a688-8124eca74603" />

This produces one value per channel, resulting in a 256-dimensional feature vector. It reduces the number of parameters required by the classifier and provides a compact representation.

### Pre-Activation Residual Blocks
The architecture uses pre-activation residual blocks. Unlike a conventional residual block that commonly applies an activation after adding the shortcut, this implementation applies BatchNorm and ReLU before the convolutional layers.

#### Residual block structure

<img width="578" height="498" alt="image" src="https://github.com/user-attachments/assets/8da84eab-a57a-4dc1-b243-bd0af01f323e" />

If the input and output dimensions match, the shortcut is an identity connection. If the dimensions differ, the network uses a 1 × 1 projection convolution with the appropriate stride. The projection operates on the activated input.

#### Mathematical formulation
Let:
<img width="147" height="56" alt="image" src="https://github.com/user-attachments/assets/9ba8f4cc-c615-47ec-9980-e2cc0125bbd9" />
The residual branch is:
<img width="280" height="53" alt="image" src="https://github.com/user-attachments/assets/fbf52c14-91c3-49d8-b51d-791e02240959" />
The output is:
<img width="139" height="38" alt="image" src="https://github.com/user-attachments/assets/cde62478-0047-4baf-830b-6dc8e3f4f009" />

There is no additional ReLU immediately after the residual addition in this implementation; the complete network applies BatchNorm and ReLU after the residual stages instead.

* Helps gradients flow through deeper networks.
* Provides direct identity paths for information propagation.
* Makes optimization of deeper residual networks easier.
* Uses projection shortcuts when spatial dimensions or channel counts change.

## Supervised Learning with Rotations (SLA)
One of the main techniques used in this project is rotation-based auxiliary supervision. Instead of training only on the original image, the network also processes four rotated versions of every training image.

### Rotation transformations
For each original image x, the notebook generates:

<img width="230" height="47" alt="image" src="https://github.com/user-attachments/assets/ebb26e28-5a2b-4f37-ac7c-af61bc0d2d5e" />

Each image has four corresponding rotation labels:
| Rotation | Rotation index |
|----------|----------------|
| 0°       | 0              |
| 90°      | 1              |
| 180°     | 2              |
| 270°     | 3              |

The model processes four views per original image for the auxiliary task. For a batch of 256 original images, this produces: 256 × 4 = 1024 rotated images for the SLA feature extraction step.

### Joint classification

The joint classification approach combines the image class and rotation class into a single classification target.

* Image classes: 10
* Rotation classes: 4
* Total output classes: 10 \times 4 = 40
* SLA head output: 40 logits.

The joint target is calculated as:

<img width="187" height="44" alt="image" src="https://github.com/user-attachments/assets/7806ac86-cfd9-46cd-8e6d-cb2b33dd41fc" />

Example: 
If the original image class is 3 and the rotation index is 2:

<img width="151" height="31" alt="image" src="https://github.com/user-attachments/assets/2fb05abf-278d-4894-89cb-8df251dff864" />

Therefore, the joint label is 14, representing the combination of image class 3 and rotation class 2. This allows the model to predict both image identity and rotation using a single 40-class classification task.

### Rotation Aligned Teacher Prediction

The SLA logits are reshaped into **(B,4,10,4)** where **(B)** is the original batch size.

The second dimension represents the four image rotations. The third dimension represents the ten object classes. The fourth dimension represents the four predicted rotations. The implementation extracts rotation-aligned logits for each view and averages them to create a teacher logit vector:

<img width="177" height="73" alt="image" src="https://github.com/user-attachments/assets/c3b4d92b-631a-498f-81d2-b9640c283ab4" />

The teacher logits are detached from the computation graph before the distillation loss is calculated. This means the teacher signal does not receive gradients from the distillation term.

## Training Objective

The total loss combines three terms:

<img width="200" height="34" alt="image" src="https://github.com/user-attachments/assets/e043bef1-debf-4337-a4e7-25de8b9f0733" />

| Term | Source | Formula |
|------|--------|---------|
| `L_joint` | SLA head (40-way class+rotation) | CE with label smoothing (ε = 0.1) |
| `L_class` | Main head (10-way CIFAR-10) | CE with label smoothing (ε = 0.1) |
| `L_distill` | SLA head → main head | T² · KL(P_teacher ∥ P_student) |

### Self Distillation
Rotation aligned SLA logits are averaged into teacher logits (detached) and both teacher and student are softened with temperature T before the KL loss:

```python
loss_distill = F.kl_div(
    F.log_softmax(class_logits / T, dim=1),
    F.softmax(teacher_logits / T, dim=1),
    reduction="batchmean"
) * (T ** 2)
```

**Defaults:** λ = 1.0, T = 1.0

## Data Preprocessing & Augmentation

The default training recipe, **`trivial`**, uses the following pipeline:

PIL conversion → reflection padding (4 px) → random crop (32 × 32) → random horizontal flip → TrivialAugmentWide → tensor conversion → normalization → Cutout (12 × 12).

| Setting | Value |
|---|---|
| Normalization mean (RGB) | `(0.4914, 0.4822, 0.4465)` |
| Normalization standard deviation (RGB) | `(0.2470, 0.2435, 0.2616)` |
| Cutout | Sets a 12 × 12 region to zero after normalization |
| Validation / test preprocessing | Tensor conversion and normalization only |

Test-time augmentation is applied separately during final inference.

## Training Configuration

![Optimizer: SGD](https://img.shields.io/badge/Optimizer-SGD-blue)
![Mixed Precision: FP16](https://img.shields.io/badge/Mixed_Precision-FP16-green)

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD with Nesterov momentum |
| Momentum | 0.9 |
| Training / evaluation batch size | 256 / 128 |
| Base learning rate | 0.2 |
| Training schedules compared | 200 and 300 epochs |
| Weight decay | 0.0005 for parameters with more than one dimension; zero for biases and normalization parameters |
| Label smoothing | 0.1 |
| Distillation temperature / weight | 1.0 / 1.0 |
| Mixed precision | FP16 |
| Training seed | 123 |
| Validation split seed | 1123 |

### Learning Rate Schedule

The learning rate increases linearly to 0.2 over the first five epochs,
then follows cosine decay toward zero. The schedule updates at every
optimizer step.

## GPU & Performance

![GPU: NVIDIA A100](https://img.shields.io/badge/GPU-NVIDIA_A100_40GB-76B900)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Device: CUDA](https://img.shields.io/badge/Device-CUDA-blue)

The notebook requires a CUDA-compatible GPU and stops if CUDA is unavailable.
The recorded run used an NVIDIA A100-SXM4-40GB in the NYU HPC environment.

| Optimization | Implementation |
|---|---|
| Mixed precision | FP16 automatic mixed precision with GradScaler |
| GPU computation | Channels-last memory format and TF32 |
| Optimizer | Fused SGD when supported; standard SGD fallback |
| Data transfers | Pinned memory and non-blocking transfers |
| Data loading | Persistent workers and cached NumPy arrays |
| Training batch size | 256 original images |

Recorded final retraining speed: approximately 10.2 seconds per epoch on the A100. Runtime varies with data loading, GPU allocation and system load.

## Experiments & Results

Both experiments use the same architecture and fixed 45,000-image training /
5,000-image validation split.

| Configuration | Epochs | Validation accuracy |
|---|---:|---:|
| SLA baseline | 200 | 97.74% |
| SLA longer training - selected | 300 | 97.82% |

Optional configurations include higher distillation temperature, Stochastic
Weight Averaging, SLA with color permutations, AugMix + JSD and
Squeeze-and-Excitation. These are disabled by default; no measured results
are reported for them.

### Final Training

The selected SLA configuration was retrained from scratch on all 50,000
labeled images for 300 epochs, using a model with 4,706,674 trainable parameters.

The 97.82% validation accuracy belongs to the model-selection experiment,
not the final model retrained on all labeled data.

## Inference & Test-Time Augmentation

Inference uses one trained model in evaluation mode, with no test-time
fine-tuning or weight updates.

For each view, the main classification probabilities and rotation-aligned
SLA classification probabilities are added. These scores are then combined
using six deterministic views:

| View | Weight |
|---|---:|
| Original | 0.25 |
| Horizontal flip | 0.25 |
| Original shifted by (−1, −1) | 0.125 |
| Original shifted by (+1, +1) | 0.125 |
| Flipped image shifted by (−1, −1) | 0.125 |
| Flipped image shifted by (+1, +1) | 0.125 |

Shifts use reflection padding. The class with the highest weighted combined
score is selected. This is test-time augmentation, not an ensemble of
independently trained models.

## Checkpoints

Checkpoints are saved every 10 epochs and after the final epoch. They include
the model, optimizer, scheduler, mixed-precision scaler, training configuration,
current epoch and optional SWA state.

| Setting | Value / behavior |
|---|---|
| Resume training | `FORCE_RETRAIN = False` with a matching configuration |
| Restart training | `FORCE_RETRAIN = True` |
| Configuration mismatch | Stops to prevent loading incompatible weights |
| Training / final seed | 123 |
| Validation split seed | 1123 |

Seeds are initialized for Python, NumPy, PyTorch, CUDA and data-loader workers.
Checkpoints do not preserve every random-number-generator state, so resumed
training is not guaranteed to match uninterrupted training bit for bit.
