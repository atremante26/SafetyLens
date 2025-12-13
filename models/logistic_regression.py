import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from src.data_preprocessing import load_data_sklearn

def fit_logistic_regression(X_train, y_train):

    model = LogisticRegression(
        max_iter = 2000,
        solver="liblinear",
        random_state=42
        )
    model.fit(X_train, y_train)

    filepath = 'results/models/logistic_regression_model.pkl'
    # Saving model as .pkl file
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

    print(f"Logistic regression model successfully saved to {filepath}")

    return model

def evaluate(model, X_val, y_val):

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="binary")

    print("\nValidation Results")
    print("------------------")
    print(f"F1-score: {f1:.4f}\n")


def main():

    # Retrieving the appropriate data formatting for logistic regression
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()

    model = fit_logistic_regression(X_train, y_train)

    evaluate(model, X_val, y_val)



if __name__ == "__main__":
    main()

