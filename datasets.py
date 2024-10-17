import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


def load_benchmarkdata(data_desc):
    if data_desc == "diabetes":
        return load_diabetes_dataset()
    elif data_desc == "german":
        return load_german_dataset()
    else:
        raise ValueError(f"Unknown data set '{data_desc}'")



def load_diabetes_dataset():
    X, y = load_diabetes(return_X_y=True)
    poisoned_samples_ratio = .1

    y_sensitive = (X[:, 1] == X[0, 1]).astype(int)   # Use 'sex' as the sensitive attribute (Note: All variables have been mean centered and scaled by the dataset provider)
    X = np.delete(X, [1], 1) # Remove sensitive attribute from data
    y = (y >= 150).astype(int)  # Convert into binary classification problem

    return X, y, y_sensitive, poisoned_samples_ratio


# Note .csv files were downloaded from https://github.com/algofairness/fairness-comparison/tree/master/fairness/data/preprocessed
# Paper: https://arxiv.org/abs/1802.04422

def load_german_dataset():
    # Load data
    df = pd.read_csv("data/german_numerical_binsensitive.csv")

    # Extract label and sensitive attribute
    y = df["credit"].to_numpy().flatten().astype(int) - 1
    y_sensitive = df["sex"].to_numpy().flatten()

    # Remove other columns and create final data set
    del df["credit"]
    del df["sex"]

    X = df.to_numpy()
    poisoned_samples_ratio = .4

    return X, y, y_sensitive, poisoned_samples_ratio

