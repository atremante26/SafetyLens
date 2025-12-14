import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def fit_logistic_regression(X_train, y_train, PATH_MOD):
    """
    Fit a logistic regression classifier and persist the trained model to disk.

    Args:
        X_train : Training feature matrix.
        y_train : Binary target labels corresponding to X_train.
        PATH_MOD : File path where the trained model will be saved as a pickle (.pkl) file.

    Returns:
        model : Fitted logistic regression model.

    Hyperparameter Notes:
        - Uses the 'liblinear' solver, which is well-suited for smaller datasets
        and binary classification.
        - `max_iter` is increased to 2000 to reduce the risk of non-convergence.
        - A fixed `random_state` is used for reproducibility.
    """

    # Initialize logistic regression model with specified hyperparameters
    model = LogisticRegression(
        max_iter = 2000,
        solver="liblinear",
        random_state=42
        )
    
    model.fit(X_train, y_train)

    # Saving trained model as .pkl file
    with open(PATH_MOD, 'wb') as file:
        pickle.dump(model, file)

    print(f"Logistic regression model successfully saved to {PATH_MOD}")

    return model

def evaluate(model, X_test, y_test, PATH_PRED):
    """
    Evaluate a trained logistic regression model on validation data,
    save predictions, and report the F1 score.

    Args:
        model : Trained logistic regression model.
        X_val : Validation feature matrix.
        y_val : True binary labels for the validation set.
        PATH_PRED : File path where validation predictions will be saved as a CSV file.

    Outputs:
        - Writes a CSV file containing logits, predicted probabilities,
        and class predictions.
        - Prints the binary F1-score to standard output.
    """

    # Get predictions
    logits = model.decision_function(X_test)
    probs = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    # Save predictions
    df_preds = pd.DataFrame({
        "Q_overall_true": y_test,
        "Q_overall_prob": probs,
        "Q_overall_pred": y_pred
    })

    df_preds.to_csv(PATH_PRED, index=False)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average="binary")

    print("\nValidation Results")
    print("------------------")
    print(f"F1-score: {f1:.4f}\n")

