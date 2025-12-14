Here's a clean, comprehensive README:

```markdown
# SafetyLens: Multi-Task Transformers for Content Safety Detection

A comprehensive NLP project comparing transformer-based models and traditional baselines for detecting unsafe content across multiple safety dimensions using the DICES-350 dataset.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [Explainability Methods](#explainability-methods)
- [Results](#results)
- [Key Findings](#key-findings)

---

## Overview

This project evaluates and compares different model architectures for binary content safety classification:
- **Logistic Regression** (TF-IDF baseline)
- **Single-Task Transformer** (RoBERTa-base)
- **Multi-Task Transformer** (2-task and 4-task variants)

We analyze both **model performance** (accuracy, F1, PR-AUC) and **explainability** (SHAP, LIME, Integrated Gradients) to understand what drives safety predictions.

---

## Project Structure

```
SafetyLens/
├── data/
│   ├── raw/                          # Original DICES dataset
│   └── processed/                    # Preprocessed binary labels
│       └── dices_350_binary.csv
│
├── models/                           # Model architectures
│   ├── logistic_regression.py
│   ├── single_task_transformer.py
│   └── multi_task_transformer.py
│
├── explainability/                   # Explainability methods
│   ├── integrated_gradients.py       # IG for transformers
│   ├── SHAP.py                       # SHAP for all models
│   └── LIME.py                       # LIME explanations
│
├── scripts/                          # Executable scripts
│   ├── train_logistic_regression.py
│   ├── train_multitask.py
│   ├── run_integrated_gradients.py
│   ├── run_shap.py
│   ├── compare_ig.py
│   ├── compare_shap.py
│   └── compare_models.py
│
├── src/                              # Core utilities
│   ├── data_preprocessing.py         # Data loading and preprocessing
│   └── visualization.py              # Figure generation
│
├── notebooks/                        # Jupyter notebooks
│   ├── eda.ipynb                     # Exploratory data analysis
│   ├── integrated_gradients.ipynb
│   ├── multitask_2_head_training.ipynb
│   └── multitask_4_head_training.ipynb
│
├── results/                          # All outputs
│   ├── models/                       # Trained model checkpoints
│   ├── evaluation/                   # Comparison CSVs
│   ├── figures/                      # Visualizations
│   ├── ig/                          # Integrated Gradients results
│   ├── shap/                        # SHAP results
│   └── lime/                        # LIME explanations
│
└── docs/                            # Documentation
    ├── hypotheses.md
    └── init_findings.md
```

---

## Setup

### Requirements
- Python 3.11+
- CUDA-compatible GPU (optional, for faster training)

### Installation

```bash
# Clone repository
git clone https://github.com/atremante26/nlp_final_project.git
cd nlp_final_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing

```bash
# Generate processed binary labels
python src/data_preprocessing.py
```

This creates `data/processed/dices_350_binary.csv` with binary safety labels for four dimensions:
- `Q_overall_binary` - Overall safety
- `Q2_harmful_binary` - Harmful content
- `Q3_bias_binary` - Bias/stereotypes  
- `Q6_policy_binary` - Policy violations

---

## Usage

### Training Models

#### Logistic Regression Baseline
```bash
python -m scripts.train_logistic_regression \
  --mod_out results/models/logistic_regression_model.pkl \
  --preds_out results/logistic_regression/test_preds.csv
```

#### Multi-Task Transformer (2-Task)
```bash
python -m scripts.train_multitask \
  --tasks 2 \
  --ckpt_out results/models/best_multitask_2.pt \
  --preds_out results/multi_task_transformer/test_predictions_2.csv
```

#### Multi-Task Transformer (4-Task)
```bash
python -m scripts.train_multitask \
  --tasks 4 \
  --ckpt_out results/models/best_multitask_4.pt \
  --preds_out results/multi_task_transformer/test_predictions_4.csv
```

---

### Running Explainability Analysis

#### Integrated Gradients (Transformers)

**2-Task Model:**
```bash
python -m scripts.run_integrated_gradients \
    --ckpt results/models/best_multitask_2.pt \
    --data_csv data/processed/dices_350_binary.csv \
    --model_type multitask \
    --task Q2_harmful \
    --n_examples 30 \
    --n_steps 50 \
    --out_csv results/ig/ig_2task_q2.csv
```

**4-Task Model:**
```bash
python -m scripts.run_integrated_gradients \
    --ckpt results/models/best_multitask_4.pt \
    --data_csv data/processed/dices_350_binary.csv \
    --model_type multitask \
    --task Q2_harmful \
    --n_examples 30 \
    --n_steps 50 \
    --out_csv results/ig/ig_4task_q2.csv
```

**Single-Task Model:**
```bash
python -m scripts.run_integrated_gradients \
    --ckpt results/models/best_singletask.pt \
    --data_csv data/processed/dices_350_binary.csv \
    --model_type single_task \
    --task Q_overall \
    --n_examples 30 \
    --n_steps 50 \
    --out_csv results/ig/ig_single_q.csv
```

#### SHAP Analysis

**Logistic Regression:**
```bash
python -m scripts.run_shap \
    --ckpt results/models/logistic_regression_model.pkl \
    --data_csv data/processed/dices_350_binary.csv \
    --vectorizer results/models/tfidf_vectorizer.pkl \
    --model_type logreg \
    --task Q_overall \
    --out_csv results/shap/logreg_shap_q_overall.csv
```

**Multi-Task Transformer:**
```bash
python -m scripts.run_shap \
    --ckpt results/models/best_multitask_4.pt \
    --data_csv data/processed/dices_350_binary.csv \
    --model_type multitask \
    --task Q2_harmful \
    --out_csv results/shap/multi_shap_q2_harmful.csv
```

**Single-Task Transformer:**
```bash
python -m scripts.run_shap \
    --ckpt results/models/best_singletask.pt \
    --data_csv data/processed/dices_350_binary.csv \
    --model_type single_task \
    --task Q_overall \
    --out_csv results/shap/single_shap_q_overall.csv
```

---

### Comparing Results

#### Compare Model Performance
```bash
python -m scripts.compare_models
```

#### Compare Integrated Gradients Results
```bash
python -m scripts.compare_ig \
    --results_2task results/ig/ig_2task_q2.csv \
    --results_4task results/ig/ig_4task_q2.csv \
    --results_single results/ig/ig_single_q.csv \
    --output results/evaluation/ig_comparison.csv
```

#### Compare SHAP Results
```bash
python -m scripts.compare_shap \
    --logreg results/shap/logreg_shap_q_overall.csv \
    --single results/shap/single_shap_q_overall.csv \
    --multitask results/shap/multi_shap_q2_harmful.csv \
    --output results/evaluation/shap_comparison.csv
```

#### Generate Visualizations
```bash
python src/visualization.py
```

---

## Models

### Logistic Regression Baseline
- **Features:** TF-IDF (20,000 features, unigrams + bigrams)
- **Purpose:** Traditional ML baseline for comparison
- **Explainability:** SHAP with linear explainer

### Single-Task Transformer
- **Architecture:** RoBERTa-base fine-tuned for binary classification
- **Tasks:** Predicts one safety dimension (Q_overall)
- **Explainability:** Integrated Gradients for token attributions

### Multi-Task Transformer
- **Architecture:** Shared RoBERTa encoder with task-specific classification heads
- **Variants:**
  - **2-Task:** Q_overall + Q2_harmful
  - **4-Task:** All four safety dimensions
- **Loss:** BCEWithLogitsLoss with pos_weight for class imbalance
- **Explainability:** Integrated Gradients per task head

---

## Explainability Methods

### Integrated Gradients (IG)
- **Purpose:** Token-level attributions for transformer models
- **Implementation:** Captum library with 50 integration steps
- **Output:** Top-k tokens per prediction with attribution scores
- **Quality Metric:** Convergence delta (<0.05 = good)

### SHAP (SHapley Additive exPlanations)
- **Purpose:** Feature importance for all model types
- **Variants:**
  - Linear explainer for logistic regression
  - Partition explainer for transformers
- **Output:** SHAP values per feature/token

### LIME (Local Interpretable Model-Agnostic Explanations)
- **Purpose:** Local explanations via perturbation
- **Status:** Sample explanations generated, not used in final analysis
- **Location:** `results/lime/`

---

## Results

### Model Performance (Q_overall)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|-------|----------|-----------|--------|-----|--------|
| Logistic Regression | - | - | - | 0.422 | 0.414 |
| Single-Task Transformer | 0.700 | 0.750 | 0.273 | 0.543 | 0.660 |
| Multi-Task (2-Task) | 0.733 | 0.400 | 0.286 | 0.469 | 0.366 |
| Multi-Task (4-Task) | 0.633 | 0.250 | 0.286 | 0.461 | 0.369 |

### Explainability Quality

**Integrated Gradients Convergence (30 examples, 50 steps):**
- 2-Task: Mean Δ = 0.280, 26.7% good convergence
- 4-Task: Mean Δ = 0.201, 26.7% good convergence  
- Single: Mean Δ = 0.199, 13.3% good convergence

**SHAP Feature Importance:**
- LogReg: 44 unique features, top: "response don"
- Transformers: Token-level, contextual attributions

### Generated Outputs

All results saved to `results/`:
- **Models:** `.pt` checkpoints in `results/models/`
- **Predictions:** CSV files per model in respective directories
- **Comparisons:** `results/evaluation/` (ig_comparison.csv, shap_comparison.csv)
- **Figures:** `results/figures/` (performance charts, distributions, confusion matrices)
- **IG Results:** `results/ig/` (token attributions per model)
- **SHAP Results:** `results/shap/` (feature importance per model)

---

## Key Findings

### Hypothesis Testing

**H1: Transformer Superiority** - Single-task transformer significantly outperforms logistic regression (F1: 0.543 vs 0.422)

**H2: Multi-Task Benefits** - Multi-task learning did not improve performance; 4-task model showed degraded performance

**H3: Class Imbalance Effects** - Severe class imbalance significantly hampers multi-task learning

**H4: Salient Token Attribution** - Multi-task models successfully identify meaningful harmful content tokens

**H5: Diffuse Attribution** - Low-confidence predictions show more distributed attributions across tokens

### Model Behavior Insights

- **Single-task:** Most conservative (13.3% predicted unsafe), highest precision
- **2-task multi-task:** Best accuracy (73.3%), balanced predictions
- **4-task multi-task:** Most aggressive (26.7% predicted unsafe), struggles with severe imbalance

### Explainability Insights

- **Token filtering:** 100% success rate removing special tokens (<s>, </s>, <pad>)
- **Multi-task advantages:** Higher confidence predictions enable better salient token analysis
- **Method suitability:** SHAP optimal for linear models, IG optimal for transformers

-