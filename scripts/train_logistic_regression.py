import argparse

from models import fit_logistic_regression, evaluate

from src.data_preprocessing import load_data_sklearn

def parse_args():
    """
    Parse command-line arguments for logistic regression training and evaluation.

    Returns:
        Parsed command-line arguments with the following attributes:
            - mod_out : File path where the trained model (.pkl) will be saved.
            - preds_out : File path where validation predictions (.csv) will be saved.
    """

    p = argparse.ArgumentParser()
    p.add_argument("--mod_out", required=True, help="Output model .pkl path")
    p.add_argument("--preds_out", required=True, help="Output test predictions .csv path")
    return p.parse_args()

def main():
    """
    Training and evaluating a logistic regression model.

    Workflow: 
        1. Parse command-line arguments.
        2. Load preprocessed data in scikit-learn compatible format.
        3. Train a logistic regression model on the training set.
        4. Evaluate the model on the validation set and save predictions.
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


'''
Example:
python -m scripts.train_logistic_regression \
  --mod_out results/models/logistic_regression_model.pkl \
  --preds_out results/logistic_regression/test_preds.csv
'''