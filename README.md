# Osteoporosis & Low BMD Screening from Hand/Wrist X-Rays via Shallow Neural Network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use%20Only-lightgrey.svg)]()
  
> **Student**: Earan Oshada 
> **Model**: Multi-Layer Perceptron (ReLU MLP with Adam Optimizer)

---

## Project Overview

Osteoporosis is a systemic skeletal condition characterized by reduced Bone Mineral Density (BMD) and micro-architectural deterioration of bone tissue, significantly increasing fracture risk in aging populations. Standard clinical diagnosis relies on Dual-Energy X-ray Absorptiometry (DXA), which is expensive and often inaccessible in primary healthcare settings.

This project investigates the feasibility of an intelligent screening system that classifies bone health into **Normal** vs. **Low BMD** (osteopenia and osteoporosis) directly from plain hand/wrist radiographs and patient demographic metadata (age, sex).

### Clinical Classification & WHO Diagnostic Criteria
According to World Health Organization (WHO) standards, diagnostic categories are established via the **T-score**, measuring patient BMD deviations relative to a young healthy reference population:

$$T\text{-score} = \frac{\text{BMD}_{\text{patient}} - \mu_{\text{reference}}}{\sigma_{\text{reference}}}$$

Where the assignment reference parameters are $\mu_{\text{reference}} = 0.86\text{ g/cm}^2$ and $\sigma_{\text{reference}} = 0.12\text{ g/cm}^2$.

| T-Score Range | Clinical Category | Target Binary Class |
| :--- | :--- | :--- |
| $T \ge -1.0$ | Normal Bone Mass | **0 (Normal)** |
| $-2.5 < T < -1.0$ | Osteopenia (Low Bone Mass) | **1 (Low BMD)** |
| $T \le -2.5$ | Osteoporosis | **1 (Low BMD)** |

---

## Architecture & Machine Learning Pipeline

```
  [Hand/Wrist Radiograph]      --> Resize (64x64) --> HOG Extraction (1764)
                                                                 │
  [Patient Age, Gender]        --> Normalization / Encoding ─────┼──> [1766 Feature Vector]
                                                                 │
                                                       StandardScaler (Train stats)
                                                                 │
                                                    ┌────────────┴────────────┐
                                                    ▼                         ▼
                                           [Refined Random Forest]    [Shallow MLP (NN)]
                                            100 Trees, Depth=8         Dense(32) -> Dense(32) -> Dense(1)
                                            Balanced Weights           Class Weights (1:15.7)
                                                    │                         │
                                                    └────────────┬────────────┘
                                                                 ▼
                                                  [Held-out Test Set Evaluation]
                                                  Accuracy, Precision, Recall, F1
```

### 1. Data Ingestion & Imbalance Profile
The dataset consists of 539 samples partitioned into a strict 3-way evaluation split:
* **Training Set ($n=377$)**: 365 Normal (96.8%), 12 Low BMD (3.2%)
* **Validation Set ($n=54$)**: 51 Normal (94.4%), 3 Low BMD (5.6%)
* **Held-Out Test Set ($n=108$)**: 98 Normal (90.7%), 10 Low BMD (9.3%)

### 2. Feature Extraction Pipeline
* **Spatial Feature Extraction**: X-rays are standardized to $64 \times 64$ pixels in grayscale. Histogram of Oriented Gradients (HOG) is extracted with 9 orientation bins, $8 \times 8$ pixels per cell, and $2 \times 2$ cells per block ($L_2\text{-Hys}$ normalization), yielding **1764 texture/gradient descriptors**.
* **Clinical Metadata**: Age (converted from months to years) and Gender (binary encoded: $M=1, F=0$).
* **Input Dimension**: $1764 + 2 = 1766$ fused features.
* **Leakage-Free Normalization**: A `StandardScaler` is fitted exclusively on the training set and applied downstream to validation and test matrices.

### 3. Model Architecture
* **Input Layer**: 1766 nodes
* **Hidden Layer 1**: 32 units, ReLU activation
* **Hidden Layer 2**: 32 units, ReLU activation
* **Output Layer**: 1 unit, Sigmoid activation ($\hat{y} \in [0, 1]$)
* **Optimization**: Adam ($\eta = 0.001$), Binary Cross-Entropy loss
* **Imbalance Mitigation**: Dynamic loss weighting assigning $15.71\times$ higher penalty to minority Low BMD misclassifications, combined with an Early Stopping monitor ($\text{patience}=15$).

---

## Experimental Results & Comparison

Both models were finalized using validation-set hyperparameter tuning and evaluated strictly once on the held-out test set ($n=108$):

| Model Architecture | Key Configuration | Accuracy | Precision | Recall (Low BMD) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Refined Random Forest** | $n_{\text{trees}}=100$, Depth=8, $\tau=0.50$ | 90.74% | 0.00% | **0.00%** | 0.00% |
| **Shallow Neural Network (MLP)** | Units: [32, 32], $\eta=0.001$, $\tau=0.50$ | 66.67% | 3.57% | **10.00%** | 5.26% |

### Key Findings
1. **The Majority Class Trap**: The Random Forest achieved high accuracy ($90.74\%$) purely by predicting all samples as "Normal", completely failing clinical utility ($0\%$ Recall).
2. **Minority Signal Recovery**: The Shallow Neural Network successfully broke the majority-class collapse through class weighting, detecting positive Low BMD cases ($10\%$ Recall on test, $66.7\%$ Recall during validation).
3. **Clinical Limitations**: A 10% test recall is insufficient for standalone screening. Downsampling to $64 \times 64$ removes microscopic bone trabeculae patterns required for precise BMD estimation. Future work requires higher-resolution patch-based deep CNNs (ResNet/DenseNet) or direct BMD regression.


---

## Repository Structure

```
├── Project_Implementation_2.1_Notebook.ipynb   # Main executable implementation notebook
├── requirements.txt                           # Pinned dependencies
├── earan_individual_nn_predictions.csv         # Exported soft probabilities for ensemble
├── figures/                                   # Generated diagnostic visualisations
│   ├── dataset_distribution.png               # Train/val/test class breakdown
│   ├── rf_confusion_matrices.png              # Random forest validation/test confusion matrices
│   ├── nn_training_curves.png                 # Loss and accuracy progression
│   ├── final_confusion_matrices.png           # Side-by-side test confusion matrices
│   └── model_comparison_test.png              # Benchmark bar chart
└── README.md                                  # Project documentation
```

---

## Getting Started & Reproducibility

### Installation
Clone this repository and install dependencies in a clean virtual environment:
```bash
git clone https://github.com/<your-username>/osteoporosis-screening-nn.git
cd osteoporosis-screening-nn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Expected Dependencies (`requirements.txt`)
```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
opencv-python>=4.8.0
tensorflow>=2.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Execution
Place the university-provided datasets in the project root:
```
├── train_groundtruth_BMD.csv
├── val_groundtruth_BMD.csv
├── Test_BMD.csv
└── Intelligent System X-rays/
```
Open Jupyter and run all cells sequentially:
```bash
jupyter notebook Project_Implementation_2.1_Notebook.ipynb
```

---
