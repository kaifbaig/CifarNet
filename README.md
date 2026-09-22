# CIFAR-10 Image Classification Using CNN

An end-to-end deep learning project for multiclass image classification on the CIFAR-10 dataset using a regularized Convolutional Neural Network (CNN) integrated with `skorch`, `scikit-learn`, and `Optuna`.

---

## Overview
This project establishes a clean, reproducible deep learning pipeline for classifying $32 \times 32$ RGB images into 10 mutually exclusive categories. Built upon a 3-stage feedforward convolutional architecture, the project addresses the problem of severe dense-layer overfitting by introducing Batch Normalization, Dropout regularization, clean transformation separation, and Bayesian hyperparameter optimization.

The implementation pairs **PyTorch** for GPU-accelerated modeling with **Skorch** to expose a native **Scikit-Learn Pipeline** interface, enabling structured machine-learning workflows without manual training loop boilerplate.

---

## Objective
- Develop a lightweight CNN capable of high generalization performance on CIFAR-10 while remaining simple enough for transparent viva defense and college project presentation.
- Systematically diagnose and resolve the large generalization gap (over $21\%$) observed in unregularized baselines.
- Maintain a disciplined workflow where the test set is reserved for final evaluation and not used for hyperparameter selection.

---

## Dataset
- **Name:** CIFAR-10 (Canadian Institute for Advanced Research)
- **Total Images:** 60,000 RGB images ($32 \times 32$ pixels across 3 channels: Red, Green, Blue)
- **Splits:** 50,000 training images, 10,000 test images
- **Classes (10 Balanced Categories, 6,000 images/class):**
  `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- **Source:** Automatically managed and cached via `torchvision.datasets.CIFAR10`.

---

## Approach / Workflow
The project implements a structured deep learning experimental workflow:

$$\text{Data Loading} \rightarrow \text{Data Understanding \& EDA} \rightarrow \text{Data Preprocessing} \rightarrow \text{CNN Baseline} \rightarrow \text{Stratified Holdout Validation} \rightarrow \text{Optuna Optimization} \rightarrow \text{Final Retraining} \rightarrow \text{Final Test Evaluation} \rightarrow \text{Error Analysis} \rightarrow \text{Model Checkpoint Saving}$$

*(Note: In accordance with standard deep learning practice on large image datasets, a stratified holdout validation split is used rather than cross-validation).*

---

## CNN Architecture
The architecture uses a 3-stage convolutional backbone with normalization and regularization:

```
Input (3 x 32 x 32)
  │
  ├── Stage 1: Conv2d(3 -> 32, 3x3, P=1)  -> BatchNorm2d(32)  -> ReLU -> MaxPool2d(2, S=2)  [32 x 16 x 16]
  ├── Stage 2: Conv2d(32 -> 64, 3x3, P=1) -> BatchNorm2d(64)  -> ReLU -> MaxPool2d(2, S=2)  [64 x 8 x 8]
  ├── Stage 3: Conv2d(64 -> 128, 3x3, P=1)-> BatchNorm2d(128) -> ReLU -> MaxPool2d(2, S=2)  [128 x 4 x 4]
  │
  ├── Flatten (128 * 4 * 4 = 2048)
  │
  └── Classifier Head:
        ├── Linear(2048 -> fc_size)
        ├── BatchNorm1d(fc_size)
        ├── ReLU
        ├── Dropout(dropout_rate)
        └── Linear(fc_size -> 10)
```

- **Total Trainable Parameters:** `621,322` (calculated and reported directly by PyTorch)
  - Convolutional stages: $960 + 18,624 + 74,112 = 93,696$ parameters
  - Dense classification head: $525,056 + 2,570 = 527,626$ parameters
  - Total: $93,696 + 527,626 = \mathbf{621,322}$

---

## Data Preprocessing
The pipeline maintains a strict separation between stochastic training augmentations and deterministic evaluation preprocessing:

* **Training Transformations (`train_transform`):**
  - `RandomCrop(32, padding=4)`: Pads the image by 4 pixels and randomly crops back to 32×32.
  - `RandomHorizontalFlip()`: Randomly mirrors images horizontally with 50% probability.
  - `ToTensor()`: Scales pixel intensities from $[0, 255]$ to $[0.0, 1.0]$.
  - `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`: Centers pixel values around zero into range $[-1.0, 1.0]$.

* **Evaluation Transformations (`test_transform`):**
  - `ToTensor()` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` (completely deterministic, ensuring consistent validation and test measurements).

---

## Validation Strategy
- **Split Configuration:** 10% stratified holdout split created using scikit-learn's `train_test_split(..., test_size=0.10, stratify=targets, random_state=SEED)`.
- **Subset Division:**
  - `train_subset`: 45,000 images sourced from `trainset` (with stochastic data augmentation).
  - `val_subset`: 5,000 images sourced from `trainset_eval` (with clean, deterministic evaluation transforms).
- **Justification over 5-Fold Cross-Validation:**
  In small tabular datasets, 5-fold cross-validation is needed to control fold variance. In CIFAR-10 with 50,000 images, running 5-fold CV across 10 Optuna trials would require training 50 separate deep neural networks (over 300 epochs), creating prohibitive computational cost. A stratified sample of 5,000 images provides an ample, low-variance evaluation signal for hyperparameter selection.

---

## Hyperparameter Optimization
A controlled 10-trial Bayesian optimization study was conducted using Optuna's `TPESampler(seed=SEED)`:
- **Search Space:**
  - `fc_size`: Categorical $\in \{128, 256\}$
  - `dropout_rate`: Float $\in [0.2, 0.5]$ (step = $0.1$)
  - `lr`: Log-float $\in [0.0005, 0.002]$
  - `weight_decay`: Log-float $\in [0.00001, 0.001]$
- **Trial Setup:** Fresh model weights per trial (`torch.manual_seed(SEED + trial.number)`, `clone(cnn_pipeline)`), 6 epochs per trial on `train_subset`, evaluated on `val_subset`.
- **Selected Configuration (Best Found during Search):**
  - **FC Size:** `256`
  - **Dropout Rate:** `0.4`
  - **Learning Rate:** `0.001147`
  - **Weight Decay:** `0.000021`

---

## Results
All values below represent the actual measured results from the executed notebook:

| Metric / Stage | Measured Value |
| :--- | :--- |
| **Total Trainable Parameters** | `621,322` |
| **Baseline Training Loss (12 Epochs)** | `1.2934` $\rightarrow$ `0.5620` |
| **Baseline Training Accuracy** | `85.10%` |
| **Validation Accuracy (Holdout)** | `80.28%` |
| **Validation Macro F1** | `0.8035` |
| **Optuna Best Validation Accuracy** | `78.60%` |
| **Final Retraining Loss (15 Epochs)** | `1.3062` $\rightarrow$ `0.5296` |
| **Observed Final Test Accuracy** | **`83.14%`** |
| **Observed Final Test Precision (Macro)** | **`0.8320`** |
| **Observed Final Test Recall (Macro)** | **`0.8314`** |
| **Observed Final Test Macro F1** | **`0.8309`** |

*(Note: Earlier unregularized baseline reached 73.51% test accuracy; the regularized pipeline achieved an observed absolute gain of **+9.63%**).*

---

## Project Structure
The repository structure reflects the actual files present in the project folder:

```
cifar10-cnn-project/
│
├── CIFAR10.ipynb                 # Complete, executed end-to-end Jupyter Notebook
├── CIFAR10_CODE_EXPLANATION.md   # Comprehensive viva and technical code reference
├── README.md                     # Project documentation & instructions
├── requirements.txt              # Pinned core dependencies
├── .gitignore                    # Standard Python/PyTorch/Jupyter ignore rules
│
├── images/                       # Generated figures (300 DPI)
│   ├── baseline_training_loss.png
│   ├── cifar10_class_distribution.png
│   ├── cifar10_confusion_matrix.png
│   ├── cifar10_pixel_distribution.png
│   ├── cifar10_sample_images.png
│   ├── cifar10_sample_predictions.png
│   └── optimized_training_loss.png
│
├── model/                        # Saved model weights (ignored by Git)
│   └── cifar10_cnn_model.pt      # PyTorch checkpoint (state_dict + best_params)
│
└── data/                         # CIFAR-10 binary cache (ignored by Git)
    └── cifar-10-batches-py/
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd CIFAR
   ```

2. **Create and activate a virtual environment:**
   - On Windows:
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Notebook
1. Launch Jupyter Lab or VS Code:
   ```bash
   jupyter lab
   ```
2. Open `CIFAR10.ipynb`.
3. Select the Python kernel from your virtual environment.
4. Run all cells sequentially. If CIFAR-10 is not already present in `./data`, `torchvision` downloads it automatically. GPU acceleration (CUDA) is detected and utilized automatically when available.


---

## Limitations
- **Image Resolution:** At $32 \times 32$ pixels, fine textures are heavily downsampled, limiting fine-grained discrimination between similar animals (`cat` vs. `dog`).
- **Model Depth:** The 3-stage sequential architecture has ~621K parameters. It is deliberately lightweight and does not match the multi-million parameter capacity of deep residual architectures (e.g., ResNet-50).
- **Uniform Receptive Fields:** Uses fixed $3 \times 3$ convolutions without multi-scale receptive field mechanisms.

---

## Future Improvements
- **Residual Connections:** Introducing skip/identity connections to allow deeper networks without vanishing gradients.
- **Learning Rate Scheduling:** Adding Cosine Annealing or OneCycleLR schedules to refine late-stage parameter convergence.
- **Data Augmentation Enhancements:** Applying Cutout or Mixup data augmentation to improve decision boundaries between visually similar categories.
