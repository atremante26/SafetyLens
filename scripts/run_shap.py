import argparse
import pandas as pd
from explainability.SHAP import (
    run_shap_logistic_regression,
    run_shap_single_task_transformer,
    run_shap_multi_task_transformer
)

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
    p.add_argument("--vectorizer", required=False, help="Path to logistic regression vectorizer")
    p.add_argument("--data_csv", required=True, help="Processed dataset CSV")
    p.add_argument("--model_type", required=True, choices=["logreg", "multitask", "single_task"], help="Type of model checkpoint")
    p.add_argument("--task", required=True, choices=list(TASK_TO_LABEL_COL.keys()))
    p.add_argument("--out_csv", required=True, help="Output CSV for IG results")
    return p.parse_args()


def main():
    args = parse_args()

    # Load test dataset
    df = pd.read_csv(args.data_csv)

    if args.model == "logreg":
        run_shap_logistic_regression(
            df=df,
            text_col="text",
            model_path=args.ckpt,
            vectorizer_path=args.vectorizer,
            out_path=f"{args.out_csv} /logreg",
        )

    elif args.model == "single_task":
        run_shap_single_task_transformer(
            df=df,
            text_col="text",
            model_path=args.ckpt,
            out_path=f"{args.out_csv}/single_task",
            device="cuda"  # or "cpu"
        )

    elif args.model == "multi_task":
        run_shap_multi_task_transformer(
            df=df,
            text_col="text",
            model_path=args.ckpt,
            task_idx=0,
            out_path=f"{args.out_csv}/multi_task",
            device="cuda"  # or "cpu"
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")
    

"""
Example:
python -m scripts.run_shap \
  --ckpt results/models/best_multitask_4.pt \
  --data_csv data/processed/dices_350_binary.csv \
  --model_type multitask \
  --task Q2_harmful \
  --out_csv results/shap/shap_q2_harmful.csv
"""