import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def fit_logistic_regression(X_train, y_train, PATH_MOD):
    """
    Fit a logistic regression classifier and save the trained model.
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

