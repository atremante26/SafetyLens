# Research Hypotheses

## Performance Hypotheses (3)

These hypotheses concern model performance on 3-class safety detection (Safe=0, Ambiguous=1, Unsafe=2).

### H1: Transformer Superiority
**Hypothesis:** Both single-task and multi-task transformers will significantly outperform the logistic regression baseline across all four target variables.

**Rationale:** 
- Transformers can capture contextual nuances and long-range dependencies
- Pre-trained language models already understand semantic safety concepts
- TF-IDF features are limited to word frequencies without context

**Expected Results:**
- Baseline LogReg F1 (macro): 0.65-0.70
- Single-Task Transformer F1 (macro): 0.75-0.80
- Multi-Task Transformer F1 (macro): 0.75-0.82

**Measurement:** Compare macro-averaged F1 scores across all models and targets.

---

### H2: Multi-Task Learning Benefits
**Hypothesis:** The multi-task transformer will perform comparably to (or slightly better than) single-task transformers, especially on the overall safety prediction task (Q_overall).

**Rationale:**
- Shared representations can capture correlations between safety dimensions
- Harmful content, bias, and policy violations often co-occur
- Multi-task learning acts as implicit regularization

**Expected Results:**
- Multi-task F1 on Q_overall: ≥ Single-task F1
- Multi-task may show 1-3% improvement on minority classes
- Training efficiency: 4x faster than training 4 separate models

**Measurement:** Compare per-target F1 scores and training time.

---

### H3: Ambiguous Class Difficulty
**Hypothesis:** All models will struggle most with the Ambiguous class (label=1), showing lower precision and recall compared to Safe and Unsafe classes.

**Rationale:**
- Ambiguous class is inherently subjective (raters unsure)
- Only ~6% of dataset (class imbalance)
- Ambiguous cases likely fall on decision boundaries

**Expected Results:**
- Safe class F1: 0.75-0.85
- Unsafe class F1: 0.70-0.80
- Ambiguous class F1: 0.40-0.60 (significantly lower)

**Measurement:** Per-class F1 scores from confusion matrices.

---

## Explainability Hypotheses (2)

These hypotheses concern token attribution patterns from explainability methods.

### H4: Toxic Keyword Detection
**Hypothesis:** For clearly unsafe content (label=2), explainability methods will consistently highlight explicit toxic keywords and slurs.

**Rationale:**
- Models rely on surface-level features for obvious cases
- Toxic terms have strong negative associations in training data
- Method should show clear attribution to problematic tokens

**Expected Results:**
- Profanity, slurs, violent language receive high attribution scores
- Top-5 tokens capture >60% of total attribution
- Clear, interpretable explanations for unsafe predictions

**Measurement:** Token importance rankings and qualitative analysis.

---

### H5: Contextual Reasoning for Ambiguous Cases
**Hypothesis:** For ambiguous content (label=1), explainability methods will show **diffuse attribution** across multiple context words rather than concentrated on single keywords.

**Rationale:**
- Ambiguous cases require nuanced interpretation
- Model uncertainty leads to distributed attention
- No single "smoking gun" token determines classification

**Expected Results:**
- Lower peak attribution scores compared to unsafe cases
- Top-5 tokens capture <40% of total attribution
- Emphasis on broader context rather than individual triggers
- Higher variance in attribution patterns

**Measurement:** Token attribution score distributions and entropy analysis.

---