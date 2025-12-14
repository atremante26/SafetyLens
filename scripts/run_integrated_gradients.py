import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import MultiTaskRoBERTa, load_model
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
    p.add_argument("--model_type", required=True, choices=["multitask", "singletask"], help="Type of model checkpoint")
    p.add_argument("--task", required=True, choices=list(TASK_TO_LABEL_COL.keys()))
    p.add_argument("--out_csv", required=True, help="Output CSV for IG results")
    return p.parse_args()

# UTILITIES
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(ckpt_path, model_type, device):
    """
    Load model and tokenizer from checkpoint
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Detect checkpoint format by checking if it has model parameter keys
    is_direct_state_dict = isinstance(ckpt, dict) and (
        'roberta.embeddings.word_embeddings.weight' in ckpt or
        'classifier.dense.weight' in ckpt
    )
    
    if is_direct_state_dict:
        # Partner's format: checkpoint IS the state_dict
        state_dict = ckpt
        model_name = "roberta-base"
        tasks = None
    else:
        # Our format: checkpoint is a wrapper dict with metadata
        model_name = ckpt.get("model_name", "roberta-base")
        tasks = ckpt.get("tasks", None)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize model architecture
    if model_type == "multitask":
        model = MultiTaskRoBERTa(
            model_name=model_name,
            tasks=tasks
        ).to(device)
    else:  # single_task
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "safe", 1: "unsafe"},
            label2id={"safe": 0, "unsafe": 1}
        ).to(device)

    # Load trained weights (suppress HuggingFace warnings during load)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, model_name, tasks

# MAIN
def main():
    args = parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load Model
    model, tokenizer, model_name, tasks = load_model(args.ckpt, args.model_type, device)
    print("Model:", model_name)
    print("Model type:", args.model_type)
    print("Tasks in ckpt:", tasks)

    # Validate task exists for multitask models
    if args.model_type == "multitask" and args.task not in (tasks or []):
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
            model_type=args.model_type,
            task=(args.task if args.model_type == "multitask" else None),
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
Example (Multi Task):
python -m scripts.run_integrated_gradients \
  --ckpt results/models/best_multitask_4.pt \
  --data_csv data/processed/dices_350_binary.csv \
  --model_type multitask \
  --task Q2_harmful \
  --out_csv results/ig/ig_q2_harmful.csv

Example (Single Task):
python -m scripts.run_integrated_gradients \
  --ckpt results/models/best_singletask.pt \
  --data_csv data/processed/dices_350_binary.csv \
  --model_type singletask \
  --task Q_overall \
  --out_csv results/ig/ig_results_single_q_overall_sample.csv
"""
