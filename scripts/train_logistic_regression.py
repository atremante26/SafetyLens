import argparse

from models import fit_logistic_regression, evaluate

from src.data_preprocessing import load_data_sklearn

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mod_out", required=True, help="Output model .pkl path")
    p.add_argument("--preds_out", required=True, help="Output test predictions .csv path")
    return p.parse_args()

def main():

    args = parse_args()

    PATH_MOD = args.mod_out
    PATH_PRED = args.preds_out

    # Retrieving the appropriate data formatting for logistic regression
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()

    model = fit_logistic_regression(X_train, y_train, PATH_MOD)

    evaluate(model, X_val, y_val, PATH_PRED)


if __name__ == "__main__":
    main()


'''
Example:
python -m scripts.train_logistic_regression \
  --mod_out results/models/logistic_regression_model.pkl \
  --preds_out results/logistic_regression/test_preds.csv
'''