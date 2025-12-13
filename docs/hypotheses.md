# Research Hypotheses

This project evaluates transformer-based models for **binary safety detection** (Safe = 0, Not Safe = 1) across multiple safety dimensions derived from the DICES dataset. We study both **model performance** and **model explainability** using Integrated Gradients.

---

## Performance Hypotheses

### H1: Transformer Superiority over Linear Baselines

**Hypothesis:**  
Transformer-based models will significantly outperform a logistic regression baseline across all evaluated safety dimensions.

**Rationale:**
- Logistic regression relies on shallow lexical features and lacks contextual understanding.
- Transformer models capture semantic nuance, negation, and long-range dependencies.
- Pre-trained language models encode prior knowledge relevant to safety and harmful content.

**Expected Results:**
- Logistic Regression: lower PR-AUC and F1 (positive class).
- Single-task Transformer: substantial improvement over the linear baseline.
- Multi-task Transformer: comparable or improved performance relative to single-task models.

**Measurement:**
- Compare **PR-AUC** and **F1 (positive class)** across all models and safety tasks.

---

### H2: Benefits of Multi-Task Learning

**Hypothesis:**  
A multi-task transformer with shared representations will perform comparably to or better than single-task transformers, particularly for the overall safety task (`Q_overall`).

**Rationale:**
- Safety dimensions such as harmful content, bias, and policy violations are correlated.
- Multi-task learning enables shared representations that capture common safety signals.
- Joint training acts as an implicit form of regularization.

**Expected Results:**
- Multi-task performance on `Q_overall` ≥ single-task performance.
- Improved stability on imbalanced tasks.
- Reduced training cost compared to training multiple independent models.

**Measurement:**
- Compare per-task **PR-AUC** and **F1 (positive class)** between single-task and multi-task models.
- Report training efficiency (single multi-task model vs multiple single-task models).

---

### H3: Task Difficulty and Class Imbalance Effects

**Hypothesis:**  
Safety dimensions with stronger class imbalance will exhibit lower predictive performance.

**Rationale:**
- Minority positive classes provide fewer learning signals.
- Certain violations (e.g., policy or bias) are more subtle and context-dependent.
- Imbalanced labels challenge threshold-based classifiers.

**Expected Results:**
- Higher PR-AUC and F1 for `Q_overall` and `Q2_harmful`.
- Lower PR-AUC and F1 for `Q3_bias` and `Q6_policy`.
- Greater variance in predictions for rarer tasks.

**Measurement:**
- Compare per-task PR-AUC, F1, and predicted positive rates on the test set.

---

## Explainability Hypotheses (Integrated Gradients)

These hypotheses analyze **token-level attribution patterns** produced by Integrated Gradients.

---

### H4: Salient Token Attribution for Unsafe Content

**Hypothesis:**  
For confidently predicted unsafe examples, Integrated Gradients will assign high attribution scores to explicit harmful or toxic tokens.

**Rationale:**
- Obvious unsafe content often contains strong lexical triggers.
- Transformer models rely heavily on these tokens for prediction.
- Attribution methods should reflect this reliance.

**Expected Results:**
- High-magnitude attributions for profanity, slurs, or explicit harmful language.
- A small number of tokens capture a large proportion of total attribution.
- Clear, human-interpretable explanations for unsafe predictions.

**Measurement:**
- Qualitative inspection of top-attributed tokens.
- Proportion of total attribution mass captured by top-k tokens.

---

### H5: Distributed Attribution for Subtle or Borderline Cases

**Hypothesis:**  
For lower-confidence or borderline unsafe predictions, attribution will be more diffuse across contextual tokens rather than concentrated on single keywords.

**Rationale:**
- Subtle safety violations depend on broader context.
- Model uncertainty leads to distributed reliance on multiple cues.
- Integrated Gradients should reflect this diffuse reasoning.

**Expected Results:**
- Lower peak attribution scores compared to highly unsafe cases.
- Attribution spread across phrases and contextual words.
- Greater diversity among top-attributed tokens.

**Measurement:**
- Compare attribution distributions between high-confidence and low-confidence predictions.
- Analyze attribution dispersion or entropy across tokens.

---

## Summary

Together, these hypotheses evaluate:
- The **predictive effectiveness** of single-task and multi-task transformer models for safety detection.
- The extent to which **Integrated Gradients** provides meaningful, human-interpretable explanations of model behavior.
