# CIFAR-10 CNN Project: Comprehensive Code & Architecture Explanation

> **Purpose of this Document:**  
> This file provides a practical, section-by-section explanation of the code, design choices, mathematical foundations, and implementation details of [CIFAR10.ipynb](file:///c:/Users/KAIF/OneDrive/Desktop/CIFAR/CIFAR10.ipynb). It is structured to serve as a study guide for project presentation, GitHub repository documentation, and B.Tech viva examination preparation.

---

## Table of Contents
1. [Section 1: Problem Understanding & Strategy](#1-problem-understanding--strategy)
2. [Section 2: Imports, Environment Setup & Dataset Loading](#2-imports-environment-setup--dataset-loading)
   - [2.1 Libraries & Device Selection](#21-libraries--device-selection)
   - [2.2 Reproducibility & Random Seeds](#22-reproducibility--random-seeds)
   - [2.3 Data Transformations (Augmentation vs. Deterministic)](#23-data-transformations-augmentation-vs-deterministic)
   - [2.4 Dataset Loading (trainset, trainset_eval, testset)](#24-dataset-loading-trainset-trainset_eval-testset)
3. [Section 3: Data Understanding & Pixel Properties](#3-data-understanding--pixel-properties)
4. [Section 4: Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Section 5: CNN Model Architecture & Pipeline Setup](#5-cnn-model-architecture--pipeline-setup)
   - [5.1 Layer-by-Layer Architecture & Spatial Math](#51-layer-by-layer-architecture--spatial-math)
   - [5.2 Role of Batch Normalization](#52-role-of-batch-normalization)
   - [5.3 Role of Dropout](#53-role-of-dropout)
   - [5.4 PyTorch Parameter Calculation](#54-pytorch-parameter-calculation)
   - [5.5 Skorch NeuralNetClassifier Integration](#55-skorch-neuralnetclassifier-integration)
   - [5.6 Scikit-Learn Pipeline Construction](#56-scikit-learn-pipeline-construction)
6. [Section 6: Baseline CNN Training & Visualization](#6-baseline-cnn-training--visualization)
7. [Section 7: Validation Setup & Optuna Hyperparameter Optimization](#7-validation-setup--optuna-hyperparameter-optimization)
   - [7.1 Validation Strategy: Holdout vs. Cross-Validation](#71-validation-strategy-holdout-vs-cross-validation)
   - [7.2 Slicing train_subset and val_subset](#72-slicing-train_subset-and-val_subset)
   - [7.3 Initial Validation Evaluation](#73-initial-validation-evaluation)
   - [7.4 Optuna Objective Function & Search Space](#74-optuna-objective-function--search-space)
   - [7.5 Running the Optuna Study](#75-running-the-optuna-study)
8. [Section 8: Final Retraining on Full Training Data](#8-final-retraining-on-full-training-data)
9. [Section 9: Final Test Evaluation & Global Metrics](#9-final-test-evaluation--global-metrics)
10. [Section 10: Detailed Classification Evaluation](#10-detailed-classification-evaluation)
    - [10.1 Classification Report (Precision, Recall, F1)](#101-classification-report-precision-recall-f1)
    - [10.2 Confusion Matrix Heatmap](#102-confusion-matrix-heatmap)
11. [Section 11: Error Analysis & Visual Predictions](#11-error-analysis--visual-predictions)
12. [Section 12: Model Persistence (Saving the Checkpoint)](#12-model-persistence-saving-the-checkpoint)
13. [Section 13: Conclusion](#13-conclusion)

---

## 1. Problem Understanding & Strategy

### A. What the Code Does
Establishes the problem domain: 10-class supervised image classification on CIFAR-10 ($32 \times 32 \times 3$ RGB images) into mutually exclusive categories: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`. It outlines the structured deep-learning pipeline:
$$\text{Data Loading} \rightarrow \text{Data Understanding} \rightarrow \text{EDA} \rightarrow \text{Preprocessing} \rightarrow \text{Baseline} \rightarrow \text{Stratified Validation} \rightarrow \text{Optuna Tuning} \rightarrow \text{Final Retraining} \rightarrow \text{Evaluation} \rightarrow \text{Model Persistence}$$

### B. Why We Have It
Provides clear theoretical motivation. Unlike 1D tabular data, 2D images possess strong local spatial correlations. Nearby pixels are statistically related, requiring convolutional kernel filters that preserve spatial topology.

### C. Important Concepts
- **Input Dimensions:** $C \times H \times W = 3 \times 32 \times 32 = 3072$ pixel values per sample.
- **Output:** 10 raw logits, where $\arg\max_k (\text{logit}_k)$ determines the predicted class label.
- **Strict Evaluation Protocol:** The official test dataset (10,000 images) remains untouched during model building, validation, and hyperparameter tuning, reserved exclusively for the final evaluation of the locked model.

---

## 2. Imports, Environment Setup & Dataset Loading

### 2.1 Libraries & Device Selection

#### A. What the Code Does
```python
import os, random, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import optuna
```
Selects computation device:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### B. Why We Have It
- **PyTorch (`torch`, `torch.nn`, `torch.optim`):** Manages tensor computations, autograd backward passes, and neural network layers.
- **Torchvision (`torchvision.transforms`, `CIFAR10`):** Manages dataset downloading, caching, and preprocessing transformations.
- **Scikit-Learn:** Provides dataset splitting (`train_test_split`), pipeline encapsulation (`Pipeline`, `clone`), and evaluation metrics (`classification_report`, `confusion_matrix`).
- **Skorch (`NeuralNetClassifier`):** Integrates PyTorch models with scikit-learn's API, eliminating manual training loop boilerplate.
- **Optuna:** Provides Bayesian hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) algorithm.
- **Device Selection:** Uses an NVIDIA CUDA GPU when available, reducing epoch time to ~12 seconds compared to several minutes on CPU.

---

### 2.2 Reproducibility & Random Seeds

#### A. What the Code Does
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

#### B. Why We Have It
Deep learning uses pseudo-random numbers for weight initialization (Kaiming normal), DataLoader shuffling, and data augmentations. Setting a global seed ensures deterministic data splits and consistent experimental runs across demonstrations.

---

### 2.3 Data Transformations (Augmentation vs. Deterministic)

#### A. What the Code Does
```python
# Training transformation: stochastic augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Evaluation transformation: clean, deterministic preprocessing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

#### B. Why We Have It
1. **`RandomCrop(32, padding=4)`:** Pads the image by 4 pixels and randomly crops back to 32×32.
2. **`RandomHorizontalFlip()`:** Randomly flips images horizontally with 50% probability, teaching left-right symmetry.
3. **`ToTensor()`:** Converts PIL images ($H \times W \times C$, range $[0, 255]$) into PyTorch FloatTensors ($C \times H \times W$, range $[0.0, 1.0]$).
4. **`Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`:** Centers pixel values around zero into range $[-1.0, 1.0]$:
   $$\text{Output} = \frac{\text{Input} - 0.5}{0.5}$$
   Centering prevents vanishing/exploding gradients during early training.

#### C. Separation Principle
- Training data uses stochastic augmentations to expose the network to visual variations and curb overfitting.
- Validation and test sets **must use deterministic transforms** (`test_transform`). Applying random crops or flips during validation would add noise to evaluation scores, obscuring actual model generalization.

---

### 2.4 Dataset Loading (trainset, trainset_eval, testset)

#### A. What the Code Does
```python
trainset = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
trainset_eval = CIFAR10(root="./data", train=True, download=True, transform=test_transform)
testset = CIFAR10(root="./data", train=False, download=True, transform=test_transform)
```

#### B. Why We Have It
- `trainset` and `trainset_eval` point to the **same underlying binary files** in `./data`.
- `trainset` applies stochastic training augmentations.
- `trainset_eval` applies deterministic evaluation preprocessing, used to create `val_subset` and measure deterministic training accuracy.
- `testset` contains the 10,000 official test images with deterministic preprocessing.

---

## 3. Data Understanding & Pixel Properties

### A. What the Code Does
Inspects dataset properties:
- Training images: `50,000` | Test images: `10,000`
- Raw image data shape: `(50000, 32, 32, 3)` with type `uint8`
- Pixel intensities range: `[0, 255]` with global mean: `120.71`
- Class counts: Exactly `5,000` images per class (perfectly balanced 10-class dataset).

### B. Why We Have It
Verifies dataset integrity before modeling. Confirming equal class representation proves that standard classification accuracy is a fair and valid metric.

---

## 4. Exploratory Data Analysis (EDA)

### A. What the Code Does
Generates and saves three diagnostic plots:
1. `cifar10_class_distribution.png`: Confirms equal sample distribution across all 10 classes.
2. `cifar10_sample_images.png`: Visualizes sample images from each class at $32 \times 32$ resolution.
3. `cifar10_pixel_distribution.png`: Plots RGB channel histograms:
   - Red: Mean $= 125.31$, Std $= 62.99$
   - Green: Mean $= 122.95$, Std $= 62.09$
   - Blue: Mean $= 113.87$, Std $= 66.70$

### B. Why We Have It
Justifies zero-centered normalization ($0.5$ mean, $0.5$ std) and visualizes the low-resolution nature of CIFAR-10, demonstrating why fine-grained visual discrimination is non-trivial.

---

## 5. CNN Model Architecture & Pipeline Setup

### 5.1 Layer-by-Layer Architecture & Spatial Math

#### A. Code Implementation
```python
class CNN(nn.Module):
    def __init__(self, fc_size=256, dropout_rate=0.3, activation=nn.ReLU):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: Conv -> BatchNorm -> Activation -> MaxPool (32x32 -> 16x16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation(),
            nn.MaxPool2d(2),

            # Stage 2: Conv -> BatchNorm -> Activation -> MaxPool (16x16 -> 8x8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation(),
            nn.MaxPool2d(2),

            # Stage 3: Conv -> BatchNorm -> Activation -> MaxPool (8x8 -> 4x4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, fc_size),
            nn.BatchNorm1d(fc_size),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

#### B. Spatial Dimension Math
Spatial dimension formula:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

1. **Input:** $3 \times 32 \times 32$
2. **Stage 1:**
   - `Conv2d(3, 32, K=3, P=1, S=1)` $\rightarrow 32 \times 32 \times 32$
   - `BatchNorm2d(32)` $\rightarrow 32 \times 32 \times 32$
   - `MaxPool2d(2, S=2)` $\rightarrow 32 \times 16 \times 16$
3. **Stage 2:**
   - `Conv2d(32, 64, K=3, P=1)` $\rightarrow 64 \times 16 \times 16$
   - `BatchNorm2d(64)` $\rightarrow 64 \times 16 \times 16$
   - `MaxPool2d(2)` $\rightarrow 64 \times 8 \times 8$
4. **Stage 3:**
   - `Conv2d(64, 128, K=3, P=1)` $\rightarrow 128 \times 8 \times 8$
   - `BatchNorm2d(128)` $\rightarrow 128 \times 8 \times 8$
   - `MaxPool2d(2)` $\rightarrow 128 \times 4 \times 4$
5. **Flattening:**
   - $128 \times 4 \times 4 = 2048$ features.
6. **Classifier Head:**
   - `Linear(2048, fc_size)` $\rightarrow$ `BatchNorm1d(fc_size)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(dropout_rate)` $\rightarrow$ `Linear(fc_size, 10)`.

---

### 5.2 Role of Batch Normalization

#### A. Mathematical Formulation
For mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$:
$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
where $\gamma, \beta$ are learnable affine parameters.

#### B. Practical Value
- Stabilizes layer activations throughout training.
- Reduces internal covariate shift, allowing stable learning with higher learning rates.
- Injects minor batch-level noise that provides mild regularization.

---

### 5.3 Role of Dropout

#### A. Mechanism
During training (`model.train()`), Dropout randomly zeroes out activations with probability $p$ ($0.4$) and scales remaining activations by $\frac{1}{1-p}$ (inverted dropout). During evaluation (`model.eval()`), Dropout is deactivated and all units forward activations without scaling.

#### B. Why It Was Added
In the original CNN, `Linear(2048, 256)` contained **524,544 parameters** (>84% of total weights). Without Dropout, dense neurons co-adapted to memorize training noise, causing a 21% generalization gap. Dropout forces dense units to learn independent, redundant representations.

---

### 5.4 PyTorch Parameter Calculation

#### A. Reported Value
```python
_model_summary = CNN()
total_trainable_params = sum(p.numel() for p in _model_summary.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_trainable_params:,}")
```
**Result:** `621,322` parameters.

#### B. Layer Breakdown
1. Conv1: $(3 \times 3 \times 3 + 1) \times 32 = 896$ + BatchNorm1: $2 \times 32 = 64 \rightarrow \mathbf{960}$
2. Conv2: $(32 \times 3 \times 3 + 1) \times 64 = 18,496$ + BatchNorm2: $2 \times 64 = 128 \rightarrow \mathbf{18,624}$
3. Conv3: $(64 \times 3 \times 3 + 1) \times 128 = 73,856$ + BatchNorm3: $2 \times 128 = 256 \rightarrow \mathbf{74,112}$
4. FC1: $2048 \times 256 + 256 = 524,544$ + BatchNorm1d: $2 \times 256 = 512 \rightarrow \mathbf{525,056}$
5. FC2: $256 \times 10 + 10 = \mathbf{2,570}$
$$\text{Sum} = 960 + 18,624 + 74,112 + 525,056 + 2,570 = \mathbf{621,322}$$

---

### 5.5 Skorch NeuralNetClassifier Integration

#### A. Code Implementation
```python
net = NeuralNetClassifier(
    module=CNN,
    module__fc_size=256,
    module__dropout_rate=0.3,
    module__activation=nn.ReLU,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    optimizer__weight_decay=0.0001,
    lr=0.001,
    max_epochs=12,
    batch_size=64,
    device=device,
    train_split=None,
    warm_start=False,
    verbose=0
)
```

#### B. Purpose
- Encapsulates the PyTorch model inside a scikit-learn estimator interface (`fit`, `predict`).
- Manages mini-batch iteration, device transfer, backpropagation, and evaluation mode switching internally.
- `train_split=None`: Prevents internal random splitting, as we use an external stratified holdout.
- `warm_start=False`: Guarantees fresh model initialization for every independent training run.

---

### 5.6 Scikit-Learn Pipeline Construction
```python
cnn_pipeline = Pipeline([("model", net)])
```
Encapsulates the model under the step name `"model"`, allowing clean parameter setting via `model__module__fc_size`, `model__module__dropout_rate`, `model__lr`, and `model__optimizer__weight_decay`.

---

## 6. Baseline CNN Training & Visualization

### A. What the Code Does
Trains the baseline regularized CNN on `trainset` for 12 epochs and evaluates deterministic training accuracy on `trainset_eval`.

### B. Observed Baseline Results
- **Epoch 1 Training Loss:** `1.2934`
- **Final Epoch 12 Training Loss:** `0.5620`
- **Loss Reduction:** `0.7315`
- **Deterministic Training Accuracy:** `85.10%`

### C. Technical Role
Confirms that the regularized network learns effectively without training-set memorization.

---

## 7. Validation Setup & Optuna Hyperparameter Optimization

### 7.1 Validation Strategy: Holdout vs. Cross-Validation

#### A. Methodology
```python
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    stratify=targets,
    random_state=SEED
)
```

#### B. Justification
- CIFAR-10 contains 50,000 images. Running 5-fold cross-validation across 10 Optuna trials would train $5 \times 10 = 50$ deep neural networks (>300 epochs), creating prohibitive computational cost.
- A 10% stratified holdout split provides **5,000 validation images** (500 per class), which offers a low-variance, statistically robust evaluation signal.

---

### 7.2 Slicing train_subset and val_subset
```python
train_subset = Subset(trainset, train_indices)       # 45,000 images with augmentation
val_subset = Subset(trainset_eval, val_indices)     # 5,000 images with deterministic transforms
val_true = targets[val_indices]
```
Prevents data augmentation leakage into validation evaluation.

---

### 7.3 Initial Validation Evaluation
```python
validation_pipeline = clone(cnn_pipeline)
validation_pipeline.fit(train_subset)
val_predictions = validation_pipeline.predict(val_subset)

val_accuracy = accuracy_score(val_true, val_predictions)
val_f1 = f1_score(val_true, val_predictions, average="macro")
```
- **Validation Accuracy:** `80.28%`
- **Validation Macro F1:** `0.8035`
Generalization gap: $85.10\% - 80.28\% = 4.82\%$ (reduced from >21%).

---

### 7.4 Optuna Objective Function & Search Space
```python
def objective(trial):
    torch.manual_seed(SEED + trial.number)
    np.random.seed(SEED + trial.number)

    fc_size = trial.suggest_categorical("fc_size", [128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 0.0005, 0.002, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.001, log=True)

    trial_pipeline = clone(cnn_pipeline)
    trial_pipeline.set_params(
        model__module__fc_size=fc_size,
        model__module__dropout_rate=dropout_rate,
        model__lr=lr,
        model__optimizer__weight_decay=weight_decay,
        model__max_epochs=6
    )

    trial_pipeline.fit(train_subset)
    predictions = trial_pipeline.predict(val_subset)
    return accuracy_score(val_true, predictions)
```
- Searches `fc_size`, `dropout_rate`, `lr`, and `weight_decay` in standard decimal notation.
- Each trial starts from fresh weights via `torch.manual_seed(SEED + trial.number)` and `clone()`.
- Uses 6 epochs per trial to quickly identify promising configurations.

---

### 7.5 Running the Optuna Study
```python
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=10)
```
- **Best Validation Accuracy Found:** `78.60%`
- **Selected Hyperparameters:**
  - `fc_size`: `256`
  - `dropout_rate`: `0.4`
  - `lr`: `0.001147`
  - `weight_decay`: `0.000021`

---

## 8. Final Retraining on Full Training Data
```python
optimized_pipeline = clone(cnn_pipeline)
optimized_pipeline.set_params(
    model__module__fc_size=best_params["fc_size"],
    model__module__dropout_rate=best_params["dropout_rate"],
    model__lr=best_params["lr"],
    model__optimizer__weight_decay=best_params["weight_decay"],
    model__max_epochs=15
)
optimized_pipeline.fit(trainset)
```
Retraining on all 50,000 training images ($45,000 \rightarrow 50,000$) maximizes feature exposure before testing.
- **Epoch 1 Training Loss:** `1.3062` $\rightarrow$ **Epoch 15 Training Loss:** `0.5296`
- **Total Loss Reduction:** `0.7765`
- **Training Time:** `184.4s` on CUDA

---

## 9. Final Test Evaluation & Global Metrics
```python
test_predictions = optimized_pipeline.predict(testset)
test_true = np.array(testset.targets)
```
- **Observed Test Accuracy:** **`83.14%`**
- **Observed Test Precision (Macro):** **`0.8320`**
- **Observed Test Recall (Macro):** **`0.8314`**
- **Observed Test Macro F1:** **`0.8309`**
Absolute gain over the unregularized baseline (73.51%): **`+9.63%`**.

---

## 10. Detailed Classification Evaluation

### 10.1 Classification Report
```
              precision    recall  f1-score   support

    airplane     0.8064    0.8790    0.8411      1000
  automobile     0.8980    0.9330    0.9152      1000
        bird     0.7522    0.7590    0.7556      1000
         cat     0.6802    0.6890    0.6846      1000
        deer     0.8126    0.8280    0.8202      1000
         dog     0.8164    0.7070    0.7578      1000
        frog     0.8578    0.8930    0.8751      1000
       horse     0.9009    0.8450    0.8720      1000
        ship     0.9159    0.8820    0.8986      1000
       truck     0.8796    0.8990    0.8892      1000

    accuracy                         0.8314     10000
   macro avg     0.8320    0.8314    0.8309     10000
weighted avg     0.8320    0.8314    0.8309     10000
```
- Strongest classes: `automobile` ($0.9152$ F1), `ship` ($0.8986$ F1), `truck` ($0.8892$ F1).
- Most challenging class: `cat` ($0.6846$ F1), primarily due to confusion with dogs.

---

### 10.2 Confusion Matrix Heatmap
Saved to `images/cifar10_confusion_matrix.png`. Visualizes high diagonal accuracy and highlights the concentration of errors along semantic boundaries (mammals vs. mammals, vehicles vs. vehicles).

---

## 11. Error Analysis & Visual Predictions
Plots 10 randomly sampled test images color-coded in green (correct) or red (error), saved to `images/cifar10_sample_predictions.png`. Provides qualitative confirmation of classification patterns.

---

## 12. Model Persistence (Saving the Checkpoint)
```python
model_checkpoint = {
    "model_state_dict": optimized_pipeline.named_steps["model"].module_.state_dict(),
    "best_params": best_params
}
torch.save(model_checkpoint, "model/cifar10_cnn_model.pt")
```
Saves the lightweight PyTorch `state_dict` (~2.5 MB) rather than full pickled objects, ensuring robust portability.

---

## 13. Conclusion
The project successfully developed a CNN-based image classification system for the CIFAR-10 dataset. The original three-stage convolutional structure was retained while improving the training pipeline with data augmentation, Batch Normalization, Dropout regularization, a dedicated stratified holdout validation split, and Optuna-based hyperparameter selection. The final locked model achieved an observed test accuracy of **83.14%** and a macro F1-score of **0.8309** on the 10,000-image CIFAR-10 test set, demonstrating effective generalization across the ten CIFAR-10 classes. Overall, the project provides a complete, reproducible workflow spanning data exploration, model training, evaluation, error analysis, and checkpoint persistence.
