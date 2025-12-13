import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from src.data_preprocessing import load_data_sklearn

def fit_logistic_regression():

    # Retrieving the appropriate data formatting for logistic regression
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()

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

if __name__ == "__main__":
    fit_logistic_regression()

