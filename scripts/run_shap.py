import argparse 
from pathlib import Path 
import pandas as pd 
from explainability.SHAP import( 
    run_shap_logistic_regression, 
    run_shap_single_task_transformer, 
    run_shap_multi_task_transformer 
) 

#CONFIG 
MAX_LEN = 128 
N_EXAMPLES = 25 
# number of rows to run IG on 
N_STEPS = 25 
# IG integration steps 
TOP_K = 12 
# number of top tokens to save 
SEED = 42 
THRESH = 0.5 
# for pred in predict_binary() 
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
    p.add_argument("--model_type", required=True, choices=["logreg", "multitask", "single_task"], help="Type of model checkpoint") 
    p.add_argument("--task", required=True, choices=list(TASK_TO_LABEL_COL.keys())) 
    p.add_argument("--out_csv", required=True, help="Output CSV for IG results") 
    return p.parse_args() 


def main():
    args = parse_args()
    df = pd.read_csv(args.data_csv)

    if args.model_type == "logreg":
        out_path = Path(args.out_csv) / "logreg_shap.csv"
        run_shap_logistic_regression(
            df=df,
            text_col="text",
            model_path=args.ckpt,
            vectorizer_path=args.vectorizer,
            out_path=out_path
        )
        print(f"SHAP results saved to {out_path}")

    elif args.model_type == "single_task":
        out_path = Path(args.out_csv) / "single_task_shap.csv"
        run_shap_single_task_transformer(
            df=df,
            text_col="text",
            model_path=args.ckpt,
            out_path=out_path,
            device="cuda"
        )
        print(f"SHAP results saved to {out_path}")

    elif args.model_type == "multitask":
        TASK_IDX = {
            "Q_overall": 0,
            "Q2_harmful": 1,
            "Q3_bias": 2,
            "Q6_policy": 3
        }
        task_idx = TASK_IDX[args.task]

        out_path = Path(args.out_csv) / f"multi_task_shap_{args.task}.csv"
        run_shap_multi_task_transformer(
            df=df,
            text_col="text",
            model_path=args.ckpt,
            task_idx=task_idx,
            out_path=out_path,
            device="cuda"
        )
        print(f"SHAP results saved to {out_path}")

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    

if __name__ == "__main__":
    main()

    
""" 
Example: 
python -m scripts.run_shap \
    --ckpt results/models/best_multitask_4.pt \
    --data_csv data/processed/dices_350_binary.csv \
    --model_type multitask \
    --task Q2_harmful \
    --out_csv results/shap/shap_q2_harmful.csv 
    
Example (logistic regression):
python -m scripts.run_shap \
    --ckpt results/models/logistic_regression_model.pkl \
    --data_csv data/processed/dices_350_binary.csv \
    --vectorizer results/models/tfidf_vectorizer.pkl \
    --model_type logreg \
    --task Q_overall \
    --out_csv results/shap/shap_q_overall.csv 
    
"""