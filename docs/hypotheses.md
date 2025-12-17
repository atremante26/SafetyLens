# Research Hypotheses

This project evaluates transformer-based models for **binary safety detection** (Safe = 0, Unsafe = 1) across multiple safety dimensions derived from the DICES-350 dataset. We study both **model performance** and **model explainability** using multiple attribution methods.

---

## Performance Hypotheses

### H1: Transformer Superiority over Linear Baselines - CONFIRMED

**Hypothesis:**  
Transformer-based models will significantly outperform a logistic regression baseline across all evaluated safety dimensions.

**Rationale:**
- Logistic regression relies on shallow lexical features and lacks contextual understanding.
- Transformer models capture semantic nuance, negation, and long-range dependencies.
- Pre-trained language models encode prior knowledge relevant to safety and harmful content.

**Results:**
- **Single-task RoBERTa substantially outperformed logistic regression on Q_overall:**
  - F1: 0.543 vs 0.422 (+29% improvement) 
  - PR-AUC: 0.660 vs 0.414 (+59% improvement) 
- Effect is substantial and consistent across both metrics

**Status: STRONGLY CONFIRMED**

---

### H2: Benefits of Multi-Task Learning - NOT SUPPORTED

**Hypothesis:**  
A multi-task transformer with shared representations will perform comparably to or better than single-task transformers, particularly for the overall safety task (Q_overall).

**Rationale:**
- Safety dimensions such as harmful content, bias, and policy violations are correlated.
- Multi-task learning enables shared representations that capture common safety signals.
- Joint training acts as an implicit form of regularization.

**Results:**
- **Contrary to hypothesis, multi-task models underperformed single-task:**
  - Single-Task F1 (Q_overall): 0.543 
  - Multi-Task-2 F1 (Q_overall): 0.469 (-14%) 
  - Multi-Task-4 F1 (Q_overall): 0.461 (-15%) 
- **However:** Multi-task showed better PR-AUC in some cases (calibration vs discrimination trade-off)
- **Key insight:** Multi-task models use 60% content words vs single-task's 40%, showing more semantically grounded reasoning despite lower F1

**Status: REJECTED for F1 performance, but NUANCED finding about interpretability advantage**

---

### H3: Task Difficulty and Class Imbalance Effects - CONFIRMED

**Hypothesis:**  
Safety dimensions with stronger class imbalance will exhibit lower predictive performance. Additionally, including severely imbalanced tasks in multi-task training will cause negative transfer.

**Rationale:**
- Minority positive classes provide fewer learning signals.
- Certain violations (e.g., policy or bias) are more subtle and context-dependent.
- Imbalanced labels challenge threshold-based classifiers.
- Shared encoders struggle to balance optimization across tasks with vastly different class distributions.

**Results:**

**Part A: Imbalance Impact on Performance** 
| Task | Positive Rate | Avg F1 | Performance Drop |
|------|---------------|--------|------------------|
| Q_overall | 32-35% | 0.474 | Baseline |
| Q2_harmful | 17.7% | 0.420 | -11% |
| Q3_bias | 9.8% | 0.322 | -32% |
| Q6_policy | 10.0% | 0.282 | -41% |

**Conclusion:** Tasks with <10% positive examples show 30-40% lower F1 scores 

**Part B: Negative Transfer** 
- 2-head model consistently outperforms 4-head on shared tasks:
  - Q_overall: Δ F1 = +0.008
  - Q2_harmful: Δ F1 = +0.011
- Adding severely imbalanced tasks (Q3_bias, Q6_policy at ~10%) degrades performance on balanced tasks
- Shared encoder struggles to balance strong signals (Q_overall at 32%) with sparse signals (Q3/Q6 at 10%)

**Status: STRONGLY CONFIRMED for both parts**

---

## Explainability Hypotheses (Integrated Gradients)

These hypotheses analyze **token-level attribution patterns** produced by Integrated Gradients.

---

### H4: High-Confidence → Concentrated Attribution - CONFIRMED (Strong for Single-Task)

**Hypothesis:**  
For confidently predicted unsafe examples (prob > 0.98), Integrated Gradients will assign high attribution scores concentrated in a few explicit harmful or toxic tokens.

**Rationale:**
- Obvious unsafe content often contains strong lexical triggers.
- Transformer models rely heavily on these tokens for prediction.
- Attribution methods should reflect this reliance.

**Results:**

**Single-Task Model:**  STRONG CONFIRMATION
- High-confidence: 48.1% attribution in top-10 tokens
- Borderline: 18.0% attribution in top-10 tokens
- **Δ = +30.1 percentage points (167% increase)** 
- High-confidence focuses on explicit keywords ("bomb", "kill", "explosive") in short texts (avg 60 tokens)

**Multi-Task Model:**  WEAK CONFIRMATION
- High-confidence: 37.8% attribution in top-10 tokens
- Borderline: 30.3% attribution in top-10 tokens
- **Δ = +7.5 percentage points (25% increase)** 
- More distributed reasoning across confidence levels

**Status: CONFIRMED for single-task (strong effect), WEAK for multi-task (modest effect)**

---

### H5: Borderline → Diffuse Attribution - CONFIRMED (Strong for Single-Task)

**Hypothesis:**  
For lower-confidence or borderline unsafe predictions (prob 0.45-0.55), attribution will be more diffuse across contextual tokens rather than concentrated on single keywords.

**Rationale:**
- Subtle safety violations depend on broader context.
- Model uncertainty leads to distributed reliance on multiple cues.
- Integrated Gradients should reflect this diffuse reasoning.

**Results:**

**Single-Task Model:**  STRONG CONFIRMATION
- High-confidence entropy: 3.659 (concentrated)
- Borderline entropy: 5.143 (diffuse)
- **Δ = -1.485 (41% increase in diffuseness)** 
- Borderline spreads attribution across broader context in longer texts (avg 254 tokens vs 60)

**Multi-Task Model:**  WEAK CONFIRMATION
- High-confidence entropy: 4.054
- Borderline entropy: 4.518
- **Δ = -0.464 (11% increase in diffuseness)** 
- More consistent attribution patterns across confidence levels

**Status: CONFIRMED for single-task (strong effect), WEAK for multi-task (modest effect)**

---

## Key Unexpected Findings

### 1. Multi-Task Interpretability Advantage
Despite lower F1 scores, multi-task models show more semantically grounded reasoning:
- Multi-task: 60% content words
- Single-task: 40% content words
- LogReg: 88% content words

**Implication:** Multi-task bridges gap between keyword-focused baseline and syntactic single-task, potentially more robust to adversarial attacks.

### 2. Model Architecture Affects Attribution Patterns
- **Single-task:** Clear bifurcation between high-confidence (keyword-focused) and borderline (context-dependent)
- **Multi-task:** More consistent, distributed reasoning across confidence levels
- **Implication:** Multi-task may be less vulnerable to keyword-based adversarial attacks despite lower F1

### 3. F1 vs PR-AUC Trade-off
- 2-head better at threshold-based classification (F1)
- 4-head better at ranking/calibration (PR-AUC)
- Suggests different optimization objectives favor different model configurations

---

## Measurement Criteria

**Performance Metrics:**
- F1 Score (positive class) - emphasizes unsafe content identification
- PR-AUC - discrimination quality across all thresholds, appropriate for imbalanced data

**Explainability Metrics:**
- Top-10 Attribution Mass: Proportion of total |attribution| in top-10 tokens (higher = more concentrated)
- Shannon Entropy: Entropy over |attribution| distribution (lower = more concentrated)
- Average Text Length: Tokens per example (proxy for decision complexity)

**Sample Sizes:**
- Performance: Full test set (3,329 ratings)
- Explainability: 30 examples per model (15 high-confidence, 15 borderline)

---