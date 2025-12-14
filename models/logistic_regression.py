import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from src.data_preprocessing import load_data_sklearn

def fit_logistic_regression(X_train, y_train, PATH_MOD):

    model = LogisticRegression(
        max_iter = 2000,
        solver="liblinear",
        random_state=42
        )
    model.fit(X_train, y_train)
    # Saving model as .pkl file
    with open(PATH_MOD, 'wb') as file:
        pickle.dump(model, file)

    print(f"Logistic regression model successfully saved to {PATH_MOD}")

    return model

def evaluate(model, X_val, y_val, PATH_PRED):

    # Get predictions
    logits = model.decision_function(X_val)
    probs = model.predict_proba(X_val)[:,1]
    y_pred = model.predict(X_val)

    # Save predictions
    df_preds = pd.DataFrame({
        "logit": logits,
        "probability": probs,
        "prediction": y_pred
    })

    df_preds.to_csv(PATH_PRED, index=False)

    # Calculate f1 score
    f1 = f1_score(y_val, y_pred, average="binary")

    print("\nValidation Results")
    print("------------------")
    print(f"F1-score: {f1:.4f}\n")

