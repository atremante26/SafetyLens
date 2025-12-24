import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.logistic_regression import fit_logistic_regression, evaluate
from src.data_preprocessing import load_data_sklearn

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mod_out", required=True, help="Output model .pkl path")
    p.add_argument("--preds_out", required=True, help="Output test predictions .csv path")
    return p.parse_args()

def main():
    """
    Training and evaluating a logistic regression model.
    """

    args = parse_args()
    PATH_MOD = args.mod_out
    PATH_PRED = args.preds_out

    # Retrieving the appropriate data formatting for logistic regression
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()

    # Fit and save the trained logistic regression model
    model = fit_logistic_regression(X_train, y_train, PATH_MOD)

    # Evaluate model on validation set and save predictions
    evaluate(model, X_test, y_test, PATH_PRED)

if __name__ == "__main__":
    main()
