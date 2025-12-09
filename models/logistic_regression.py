import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import data_preprocessing as data_prep

def fit_logistic_regression():

    # Retrieving the appropriate data formatting for logistic regression
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.load_data_sklearn()

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    filepath = '../results/models/logistic_regression_model.pkl'
    # Saving model as .pkl file
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

    print(f"Logistic regression model successfully saved to {filepath}")


    

