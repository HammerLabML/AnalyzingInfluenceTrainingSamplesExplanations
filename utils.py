import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample


def downsample(X, y, n_samples=None):
    sampling = RandomUnderSampler(random_state=42)
    X, y = sampling.fit_resample(X, y)

    if n_samples is not None:
        if X.shape[0] > n_samples:
            X, y = resample(X, y, replace=False, n_samples=n_samples, random_state=42)

    return X, y


class SvcWrapper():
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X): # Not provided by LinearSVC
        y_pred_proba = np.zeros((X.shape[0], 2))
        y_pred = self.model.predict(X)
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred_proba[i, 1] = 1
            else:
                y_pred_proba[i, 0] = 1

        return y_pred_proba
