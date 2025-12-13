import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from models import MultiTaskRoBERTa  # add other models

from explainability import compute_integrated_gradients, top_k_tokens

# CONFIG
MAX_LEN = 128          
N_EXAMPLES = 25        # number of rows to run IG on
N_STEPS = 25           # IG integration steps
TOP_K = 12             # number of top tokens to save
SEED = 42
THRESH = 0.5           # for pred in predict_binary()

TASK_TO_LABEL_COL = {
    "Q_overall": "Q_overall_binary",
    "Q2_harmful": "Q2_harmful_binary",
    "Q3_bias": "Q3_bias_binary",
    "Q6_policy": "Q6_policy_binary",
}

# ARGUMENT PARSING
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--data_csv", required=True, help="Processed dataset CSV")
    p.add_argument("--model_type", required=True, choices=["multitask", "single_task"], help="Type of model checkpoint")
    p.add_argument("--task", required=True, choices=list(TASK_TO_LABEL_COL.keys()))
    p.add_argument("--out_csv", required=True, help="Output CSV for IG results")
    return p.parse_args()

# UTILITIES
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(ckpt_path, model_type, device):
    # Load checkpoint directory
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract metadata from checkpoint
    model_name = ckpt.get("model_name", "roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tasks = ckpt.get("tasks", None)

    # Initialize model based on type
    if model_type == "multitask":
        model = MultiTaskRoBERTa(
            model_name=model_name,
            tasks=ckpt["tasks"]
        ).to(device)
    else:
        model = AutoModelForSequenceClassification( # TODO: FIX WITH ACTUAL MODEL
            model_name,
            num_labels=1
        ).to(device)

    # Load trained weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer, tasks, model_name

# MAIN
def main():
    args = parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load Model
    model, tokenizer, model_type, tasks, model_name = load_model(args.ckpt, args.model_type, device)
    print("Model:", model_name)
    print("Model type:", model_type)
    print("Tasks in ckpt:", tasks)

    # Validate task exists for multitask models
    if model_type == "multitask" and args.task not in tasks:
        raise ValueError(f"Task {args.task} not found in checkpoint tasks {tasks}")

    # Load and Filter Data
    df = pd.read_csv(args.data_csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a `text` column")

    label_col = TASK_TO_LABEL_COL[args.task]
    if label_col not in df.columns:
        raise ValueError(f"Input CSV missing label column: {label_col}")

    # Clean Data
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df[df[label_col].isin([0, 1])].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows after filtering (need label in {0,1} and non-null text).")

    # Sample Examples
    df_sample = df.sample(n=min(N_EXAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"Running IG on {len(df_sample)} examples (task={args.task})")

    # Run Integrated Gradients
    rows = []
    for i in tqdm(range(len(df_sample)), desc="Integrated Gradients"):
        # Extract text and ground truth label
        text = str(df_sample.loc[i, "text"])
        y_true = int(df_sample.loc[i, label_col])

        # Compute IG attributions for this example
        res = compute_integrated_gradients(
            text=text,
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            task=(args.task if model_type == "multitask" else None),
            device=str(device),
            max_length=MAX_LEN,
            n_steps=N_STEPS,
        )

        # Extract top-k most influential tokens
        top = top_k_tokens(res["tokens"], res["attributions"], k=TOP_K)

        # Build output row with prediction info
        out = {
            "task": args.task,
            "true_label": y_true,
            "prob_pos": res["prob"],
            "pred": res["pred"],
            "logit": res["logit"],
            "convergence_delta": res["convergence_delta"],
            "text": text,
        }

        # Add top-k tokens and their attributions as separate columns
        for j, (tok, val) in enumerate(top, start=1):
            out[f"tok_{j}"] = tok
            out[f"attr_{j}"] = val

        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Save Results
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()

"""
Example:
python scripts/run_integrated_gradients.py \
  --ckpt results/models/best_multitask_4.pt \
  --data_csv data/processed/dices_350_binary.csv \
  --model_type multitask \
  --task Q2_harmful \
  --out_csv results/ig/ig_q2_harmful.csv
"""
