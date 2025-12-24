# SafetyLens

**Multi-Model Content Safety Detection with Explainable AI**

Academic research project comparing model architectures and explainability methods for content safety classification in conversational AI contexts.

**🌐 Live Demo:** https://atremante26.github.io/SafetyLens/

---

## 🎯 Overview

This project implements and compares three model architectures for content safety detection, evaluated on the DICES-350 dataset:

1. **Logistic Regression** (TF-IDF baseline) - F1: 0.422
2. **Single-Task RoBERTa** (binary classifier) - F1: 0.543  
3. **Multi-Task RoBERTa** (2-head & 4-head variants) - F1: 0.469, 0.461

Each model is analyzed using three explainability methods:
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations)
- **Integrated Gradients** (gradient-based attribution)

**Key Research Finding:** Single-task models outperformed multi-task variants under severe class imbalance. When training on rare safety dimensions (bias: 9.1% unsafe, policy: 11.4% unsafe), multi-task learning degraded performance compared to focused single-task approaches.

**Web Application:** An interactive interface is provided for model comparison and real-time explainability visualization. See [Interactive Demo](#-interactive-demo-optional) section.

---

## 📊 Research Results

### Model Performance

| Model | F1 Score | Precision | Recall | Architecture |
|-------|----------|-----------|--------|--------------|
| Logistic Regression | 0.422 | 0.419 | 0.425 | TF-IDF + sklearn |
| Single-Task RoBERTa | **0.543** | 0.531 | 0.556 | Binary classifier |
| Multi-Task 2-head | 0.469 | 0.458 | 0.481 | Overall + Harmful |
| Multi-Task 4-head | 0.461 | 0.449 | 0.473 | All dimensions |

### Key Findings

1. **Class imbalance significantly impacts multi-task learning**
   - Multi-task models struggled with rare safety dimensions
   - Q3_bias: Only 9.1% unsafe examples in training data
   - Q6_policy: Only 11.4% unsafe examples
   - Shared representations failed to capture low-frequency patterns

2. **Single-task models excel in focused classification**
   - 16% improvement over baseline (0.422 → 0.543 F1)
   - Better handling of class imbalance through focused optimization
   - Simpler architecture reduces overfitting on small positive classes

3. **Explainability reveals model reasoning differences**
   - Logistic regression focuses on lexical features (keywords)
   - Transformers capture contextual patterns and implicit sentiment
   - Token attributions differ substantially across methods (LIME vs SHAP vs IG)

### Safety Dimensions

- **Q_overall**: General safety assessment across all categories
- **Q2_harmful**: Harmful, offensive, or inappropriate content  
- **Q3_bias**: Stereotypes, bias, and discriminatory language
- **Q6_policy**: Platform policy violations and terms of service

---

## 🔬 Methodology

### Dataset

**DICES-350**: Diverse Conversational AI Safety Examples
- 350 conversation turns from human-AI dialogues
- Multiple safety annotations per example
- Severe class imbalance across dimensions
- Conversation-level train/test split to prevent data leakage

### Data Preprocessing

Run `notebooks/eda.ipynb` to process the raw dataset. This creates `data/processed/dices_350_binary.csv` with binary safety labels for four dimensions:

- `Q_overall_binary` - Overall safety
- `Q2_harmful_binary` - Harmful content
- `Q3_bias_binary` - Bias/stereotypes  
- `Q6_policy_binary` - Policy violations

Key preprocessing steps:
- Binary label creation from multi-class annotations
- Conversation-level data splitting
- Class distribution analysis
- Text normalization and cleaning

### Model Architectures

**Logistic Regression Baseline:**
- TF-IDF vectorization (max 5000 features)
- L2 regularization
- Class weight balancing

**Single-Task RoBERTa:**
- RoBERTa-base pretrained weights
- Binary classification head
- Fine-tuned on Q_overall dimension
- Max sequence length: 512 tokens

**Multi-Task RoBERTa:**
- Shared RoBERTa encoder
- Separate classification heads per task
- 2-head variant: Q_overall + Q2_harmful
- 4-head variant: All safety dimensions
- Max sequence length: 256 tokens
- Weighted BCE loss for class imbalance

### Explainability Methods

**LIME (Local Interpretable Model-agnostic Explanations):**
- Perturbation-based local approximation
- 500 perturbed samples per prediction
- Linear model fitted to local neighborhood
- Works with all model types

**SHAP (SHapley Additive exPlanations):**
- Game-theoretic feature attribution
- Exact computation for linear models
- Kernel SHAP for transformers
- Measures marginal contribution of features

**Integrated Gradients:**
- Gradient-based attribution method
- Path integration from baseline to input
- 50 integration steps
- Transformer-specific (requires gradients)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Repository Structure
```
SafetyLens/
├── data/
│   ├── raw/                  # Original DICES-350 dataset
│   └── processed/            # Processed binary labels
├── notebooks/
│   └── eda.ipynb            # Data exploration and preprocessing
├── scripts/
│   ├── train_logistic_regression.py
│   ├── train_singletask.py
│   ├── train_multitask.py
│   ├── run_integrated_gradients.py
│   ├── run_shap.py
│   ├── compare_models.py
│   ├── compare_ig.py
│   └── compare_shap.py
├── models/
│   ├── checkpoints/          # Trained model files
│   └── multi_task_transformer.py
├── results/
│   ├── models/              # Saved checkpoints
│   ├── ig/                  # Integrated Gradients results
│   ├── shap/               # SHAP analysis results
│   └── evaluation/         # Comparison outputs
└── app/                    # Interactive web interface (optional)
```

---

## 📝 Usage

### 1. Data Preprocessing
```bash
# Run EDA notebook to create processed dataset
jupyter notebook notebooks/eda.ipynb

# Output: data/processed/dices_350_binary.csv
```

### 2. Training Models

#### Logistic Regression Baseline
```bash
python -m scripts.train_logistic_regression \
  --mod_out results/models/logistic_regression_model.pkl \
  --preds_out results/logistic_regression/test_preds.csv
```

#### Single-Task Transformer
```bash
python -m scripts.train_singletask \
  --ckpt_out results/models/best_singletask.pt \
  --preds_out results/single_task/test_predictions.csv
```

#### Multi-Task Transformer (2 Tasks)
```bash
python -m scripts.train_multitask \
  --tasks 2 \
  --ckpt_out results/models/best_multitask_2.pt \
  --preds_out results/multi_task_transformer/test_predictions_2.csv
```

#### Multi-Task Transformer (4 Tasks)
```bash
python -m scripts.train_multitask \
  --tasks 4 \
  --ckpt_out results/models/best_multitask_4.pt \
  --preds_out results/multi_task_transformer/test_predictions_4.csv
```

### 3. Explainability Analysis

#### Integrated Gradients (Transformer Models)

**Multi-Task 2-Head:**
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

**Multi-Task 4-Head:**
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

**Single-Task:**
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

### 4. Comparing Results

#### Compare Model Performance
```bash
python -m scripts.compare_models

# Output: Comparative F1, precision, recall across all models
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

# Creates plots in results/figures/
```

---

## 🌐 Interactive Demo (Optional)

An interactive web application is provided for real-time model comparison and explainability visualization. This is a supplementary tool for exploring the research results.

### Quick Start with Docker
```bash
# Download pre-trained models
mkdir -p models/checkpoints && cd models/checkpoints
gdown 1NSGLiM2M8l_h2N0m0DDKjVWsh2aQlnL4  # logreg (~5 MB)
gdown 1WHjq8UaTlRb2SqudGZCv5RiUQL42NQSB  # vectorizer (~10 MB)
gdown 1DX2oY2zPX7DgH6_F2j6BxvL6CNAd1IUH  # singletask (~500 MB)
gdown 1FZdKRT3E4mISFQ2HnRCODxXJQkF1xeBf  # multitask_2 (~500 MB)
gdown 11AQtrY6veTF_j337g2f9YaSBxipfsdor  # multitask_4 (~500 MB)
cd ../..

# Start application
./docker-start.sh
```

**Access:** http://localhost:5173/SafetyLens/

**Live Demo:** https://atremante26.github.io/SafetyLens/

### Manual Setup

**Backend:**
```bash
cd app/backend
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd app/frontend
npm install
npm run dev
```

---

## 📚 Technical Stack

### Research Pipeline
- **PyTorch 2.1.0** - Deep learning framework
- **Transformers** (Hugging Face) - RoBERTa implementation
- **scikit-learn 1.8.0** - Logistic regression baseline
- **Captum** - Integrated Gradients
- **LIME** - Model-agnostic explainability
- **SHAP** - Shapley value computation

### Web Application
- **FastAPI** - Backend REST API
- **React 18** - Frontend interface
- **Docker** - Containerized deployment

---

## 📖 Paper Highlights

### Research Questions

**RQ1:** How do multi-task and single-task transformer architectures compare for content safety detection?

**Finding:** Single-task models achieve 16% higher F1 score (0.543 vs 0.469) when class imbalance is severe. Multi-task learning fails to leverage shared representations when some tasks have <10% positive examples.

**RQ2:** How do different explainability methods (LIME, SHAP, IG) compare in revealing model reasoning?

**Finding:** Methods show complementary strengths:
- LIME: Best for model-agnostic local explanations
- SHAP: Most faithful for linear models (game-theoretic guarantees)
- IG: Most precise for transformers (direct gradient attribution)

**RQ3:** What token-level patterns distinguish safe from unsafe content across models?

**Finding:** Logistic regression relies on explicit keywords while transformers capture implicit contextual cues. Bias and policy violations show lower attribution consistency than explicit harmful content.

### Methodology Contributions

- **Conversation-level data splitting** to prevent leakage
- **Weighted loss functions** for class imbalance handling
- **Comparative analysis framework** for explainability methods
- **Multi-dimensional safety evaluation** across 4 safety aspects

### Limitations & Future Work

- Small dataset size (350 examples) limits generalization
- English-only conversations
- Binary classification (does not capture severity)
- Static evaluation (no temporal dynamics)

**Future directions:**
- Data augmentation for rare classes
- Focal loss for better imbalance handling
- Multilingual safety detection
- Conversation-level contextual models

---

## 📝 Citation
```bibtex
@misc{tremante2024safetylens,
  author = {Tremante, Andrew},
  title = {SafetyLens: Multi-Model Content Safety Detection with Explainable AI},
  year = {2024},
  institution = {Amherst College},
  course = {Natural Language Processing},
  url = {https://github.com/atremante26/SafetyLens}
}
```

---

## 📄 License

Academic project completed for Amherst College COSC-243: Natural Language Processing (Fall 2024).

---
