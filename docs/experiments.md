Performance (H1–H3)
Experiment 1 — Performance table across tasks (minimal extension of what you have) ✅/⚠️

Purpose: Directly tests H1 + H3, and supports H2.

You already have Q_overall for 4 models. Extend to the other tasks without adding lots of new model types:

Run and report for each task: Q_overall, Q2_harmful, Q3_bias, Q6_policy

Models to include (keep it small):

LogReg (per task)

Single-task RoBERTa (per task)

Multi-task RoBERTa 4-head (all tasks)

Optional (only if you already have it trained / cheap):
4) Multi-task 2-head (report only Q_overall + Q2_harmful)

Deliverable: one table = rows(model), columns(PR-AUC/F1_pos per task)

This covers:

H1: transformer > logreg (per task)

H3: imbalance tasks (Q3/Q6) lower performance

Partial H2: multi-task vs single-task comparisons

If time is tight: do not retrain anything. Just compute metrics from prediction CSVs you already generate.

Experiment 2 — 2-head vs 4-head ablation on shared tasks (H2 + H3) ✅/⚠️

Purpose: Your clean story for “does adding imbalanced tasks hurt?”

Compare 2-head vs 4-head on shared tasks only:

Q_overall, Q2_harmful

Report PR-AUC + F1_pos, plus pred_pos_rate.

Deliverable: tiny 2×2 table (2-head vs 4-head × two tasks)

This is the strongest evidence for “negative transfer / imbalance impact” without extra experiments.

Explainability (H4–H5)
Experiment 3 — Integrated Gradients: high-confidence vs borderline (H4 + H5) ⚠️ (critical)

Your current IG summary is mostly “low confidence.” That’s fine, but H4/H5 need an explicit contrast.

Pick one primary task for IG to keep it tight:

Recommend Q2_harmful (usually more “keyword-y” and easier to interpret)

For each model you care about (I’d do 2 models max):

Single-task transformer (Q2)

Multi-task 4-head (Q2 head)

Select examples from the test set:

High-confidence unsafe: top K by predicted prob (e.g., K=15–25)

Borderline: closest to 0.5 (e.g., K=15–25 around [0.45, 0.55])

Compute for each example:

Top-10 tokens by |IG|

Top-k attribution mass (concentration)

Attribution entropy (diffuseness)

Deliverables:

One small table: mean(top-k mass), mean(entropy) for high-conf vs borderline (per model)

2–3 qualitative examples showing token lists + short snippets

Covers:

H4: high-confidence unsafe → concentrated on explicit tokens

H5: borderline → more diffuse / contextual

This replaces your “IG token overlap between heads” as the core IG experiment, because it maps directly to H4/H5.

Experiment 4 — Multi-task head overlap (optional “nice to have”) ⚠️ (only if quick)

This is your Experiment A. It’s interesting, but it’s not required to satisfy H4/H5.

If you do it, do it only on the same K examples from Experiment 3 to avoid more sampling complexity.

Compute Jaccard overlap of top-10 IG tokens between:

Q_overall head vs Q2_harmful head within the same multi-task model

Deliverable: mean/median overlap + 1 example.