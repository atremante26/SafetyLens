## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)

---

## Overview

This project evaluates and compares different model architectures for binary content safety classification:
- **Logistic Regression** (TF-IDF baseline)
- **Single-Task Transformer** (RoBERTa-base)
- **Multi-Task Transformer** (2-task and 4-task variants)

We analyze both **model performance** (accuracy, F1, PR-AUC) and **explainability** (SHAP, LIME, Integrated Gradients) to understand what drives safety predictions.

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
Run `notebooks/eda.ipynb`. This creates `data/processed/dices_350_binary.csv` with binary safety labels for four dimensions:
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
