import sys
import argparse 
from pathlib import Path 
import pandas as pd 

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from explainability.SHAP import( 
    run_shap_logistic_regression, 
    run_shap_single_task_transformer, 
    run_shap_multi_task_transformer 
) 

#CONFIG 
MAX_LEN = 128 
N_EXAMPLES = 25 
N_STEPS = 25 
TOP_K = 12 
SEED = 42 
THRESH = 0.5 
TASK_TO_LABEL_COL = {
    "Q_overall": "Q_overall_binary",
    "Q2_harmful": "Q2_harmful_binary",
    "Q3_bias": "Q3_bias_binary",
    "Q6_policy": "Q6_policy_binary"
    } 

# ARGUMENT PARSING 
def parse_args(): 
    p = argparse.ArgumentParser() 
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)") 
    p.add_argument("--vectorizer", required=False, help="Path to logistic regression vectorizer") 
    p.add_argument("--data_csv", required=True, help="Processed dataset CSV") 
    p.add_argument("--model_type", required=True, choices=["logreg", "multitask", "singletask"], help="Type of model checkpoint") 
    p.add_argument("--task", required=True, choices=list(TASK_TO_LABEL_COL.keys())) 
    p.add_argument("--out_csv", required=True, help="Output CSV for IG results") 
    return p.parse_args() 


def main():
    # Parse arguments
    args = parse_args()

    # Read in data
    df = pd.read_csv(args.data_csv)

    # Sample examples
    N_SHAP_EXAMPLES = 5
    df_sample = df.sample(n=min(N_SHAP_EXAMPLES, len(df)), random_state=42)
    if args.model_type == "logreg":
        run_shap_logistic_regression(
            df=df_sample,
            text_col="text",
            model_path=args.ckpt,
            vectorizer_path=args.vectorizer,
            out_path=args.out_csv
        )
        print(f"SHAP results saved to {args.out_csv}")

    elif args.model_type == "singletask":
        run_shap_single_task_transformer(
            df=df_sample,
            text_col="text",
            model_path=args.ckpt,
            out_path=args.out_csv,
            device="cpu"
        )
        print(f"SHAP results saved to {args.out_csv}")

    elif args.model_type == "multitask":
        TASK_IDX = {
            "Q_overall": 0,
            "Q2_harmful": 1,
            "Q3_bias": 2,
            "Q6_policy": 3
        }
        task_idx = TASK_IDX[args.task]

        run_shap_multi_task_transformer(
            df=df_sample,
            text_col="text",
            model_path=args.ckpt,
            task_idx=task_idx,
            out_path=args.out_csv,
            device="cpu"
        )
        print(f"SHAP results saved to {args.out_csv}")

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    

if __name__ == "__main__":
    main()