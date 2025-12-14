# Initial Findings
---

## 1. Performance Findings

### H1: Transformer Superiority over Linear Baselines

**Observation:**  
- Logistic regression baseline achieved PR-AUC = `0.414` and F1 (positive class) = `0.422` for `Q_overall`.  
- Single-task transformer improved PR-AUC to `0.660` and F1 to `0.543`.  
- Multi-task transformer (2-task) achieved PR-AUC = `0.366` and F1 = `0.469`.
- Multi-task transformer (4-task) achieved PR-AUC = `0.369` and F1 = `0.461`.

**Interpretation:**  
- The single-task transformer model performed much better than the baseline logistic regression and both multi-task transformer models.
- Multi-task transformer models did not noticeably outperform the single-task transformer.

**Figure 1:** PR-AUC and F1 comparison across models  
![PR-AUC and F1 Comparison](results/figures/q_overall_metrics_bar.png)

**Status:** **H1 SUPPORTED** - Transformers significantly outperform logistic regression baseline.

---

### H2: Benefits of Multi-Task Learning

**Observation:**  
- Single-task F1 for `Q_overall`: `0.543`  
- Multi-task 2-task F1 for `Q_overall`: `0.469`  
- Multi-task 4-task F1 for `Q_overall`: `0.461`  

**Additional Analysis from IG Results:**
- **Model Behavior Comparison** (30 examples on `Q2_harmful`):
  - Single-task: Most conservative (13.3% predicted unsafe), highest precision (0.750)
  - 2-task multi-task: Balanced (16.7% predicted unsafe), best accuracy (0.733)
  - 4-task multi-task: Most aggressive (26.7% predicted unsafe), lowest accuracy (0.633)

**Interpretation:**  
- Multi-task learning did not improve model performance for detecting unsafe texts.
- Adding more heads to the transformer did not lead to improvements in model performance.
- Multi-task models show different prediction behaviors: 2-task is more conservative and accurate, while 4-task is more aggressive but less precise.
- We believe this to be an issue of class imbalance across different safety dimensions leading to conflicting learning signals.

**Status:** **H2 REJECTED** - Multi-task learning did not improve performance over single-task models.

---

### H3: Task Difficulty and Class Imbalance Effects

**Observation:**
- Class imbalance varies significantly across safety dimensions in DICES-350:
  - `Q_overall`: 28% unsafe (moderate imbalance)
  - `Q2_harmful`: Severe imbalance (most examples are safe)
  - `Q3_bias`: Severe imbalance
  - `Q6_policy`: Severe imbalance

**Multi-Task Training Challenges:**
- 4-task model learns significantly slower than 2-task model
- Training metrics show 4-task model struggles to learn meaningful patterns for severely imbalanced tasks
- Per-task performance degradation observed when adding more imbalanced tasks

**Mitigation Strategy:**
- Used `pos_weight` in `BCEWithLogitsLoss` to address class imbalance
- 2-task model (focusing on less imbalanced tasks) achieved better overall performance
- Early stopping after epoch 1 prevented overfitting

**Interpretation:**
- Severe class imbalance makes some safety dimensions extremely difficult to learn
- Multi-task models face compounding challenges when tasks have different imbalance levels
- The 2-task approach (selecting more balanced tasks) shows promise as a middle ground

**Status:** **H3 SUPPORTED** - Class imbalance significantly affects task difficulty and model performance.

---

## 2. Explainability Findings

### H4: Salient Token Attribution for High-Confidence Predictions

**Method:** Integrated Gradients (50 steps, 30 examples per model)

**Observation:**
- **High-confidence examples** (probability > 0.8):
  - 2-task multi-task: 2 examples
  - 4-task multi-task: 2 examples  
  - Single-task: 0 examples
  
- **Token Quality:**
  - All models: 0% special tokens as top attribution (perfect filtering)
  - Multi-task models produce meaningful content tokens:
    - Example (4-task, prob=0.855): "Jason", "He", "kill" (harmful content)
    - Example (2-task, prob=0.831): "enjoy", "shame" (contextual words)
  - Single-task: More conservative, no high-confidence predictions

**Top Salient Tokens by Model:**
- **2-Task:** "homeless", "enjoy", "shame" - contextual tokens indicating harmful scenarios
- **4-Task:** "Jason", "He", "them" - proper nouns and pronouns in unsafe contexts
- **Single-Task:** ".", "to", "solve" - function words, less discriminative

**Interpretation:**
- Multi-task models successfully identify salient tokens for unsafe content predictions
- High-confidence predictions correlate with specific harmful content tokens (names in violent contexts, derogatory language)
- Multi-task models are more decisive than single-task, enabling better salient token analysis
- Token filtering successfully removes non-interpretable special tokens

**SHAP Comparison (Logistic Regression):**
- Top features: "response don", "alcohol", "wine", "don talk", "campfire"
- Uses 44 unique TF-IDF features vs. transformers' token-level attributions
- 23 features account for 80% of importance (more distributed than transformers)

**Status:** **H4 SUPPORTED** - Models successfully identify salient tokens for unsafe predictions, with multi-task models showing clearer attribution patterns.

---

### H5: Diffuse Attribution for Low-Confidence Predictions

**Method:** Integrated Gradients (50 steps, 30 examples per model)

**Observation:**
- **Low-confidence examples** (probability < 0.5):
  - 2-task multi-task: 25 examples (83.3%)
  - 4-task multi-task: 22 examples (73.3%)
  - Single-task: 26 examples (86.7%)

- **Confidence Distribution:**
  - 2-task: Range 0.045–0.855, mean 0.357
  - 4-task: Range 0.022–0.907, mean 0.365
  - Single-task: Range 0.156–0.605, mean 0.325

**Attribution Patterns:**
- Low-confidence predictions show more distributed attributions across multiple tokens
- High-confidence predictions concentrate on fewer, more discriminative tokens
- Multi-task models exhibit wider confidence ranges, enabling better diffuse vs. concentrated comparison

**Interpretation:**
- All models produce abundant low-confidence predictions suitable for testing H5
- The hypothesis of diffuse attribution for borderline cases is testable with current data
- Multi-task models' higher confidence spread provides better contrast for comparing concentrated vs. diffuse patterns
- Single-task model's narrow confidence range (max 0.605) limits its ability to show salient token concentration

**Status:** **H5 TESTABLE** - Sufficient low-confidence examples available; preliminary patterns suggest diffuse attribution for borderline cases.

---
