import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import tensorflow as tf

from .scoring import Scoring
from .dnn import DeepNeuralNetwork


class AccuracyScoring(Scoring):
    def __init__(self, clf, X_test: np.ndarray, y_test: np.ndarray, **kwds):
        self.clf = clf
        self.X_test = X_test
        self.y_test = y_test

        super().__init__(**kwds)

    def reset(self):
        pass

    def compute_score(self, X_train: np.ndarray, y_train: np.ndarray, n_itr: int = 1) -> float:
        # Repeat multiple times to mitigate randomness
        total_scores = []
        for _ in range(n_itr):
            # Fit model
            if not isinstance(self.clf, MLPClassifier) and not isinstance(self.clf, LogisticRegression):
                self.clf.model = tf.keras.models.clone_model(self.clf.model)    # Reset weights!
                self.clf.compile()
            self.clf.fit(X_train, y_train)
            y_test_pred = self.clf.predict(self.X_test)

            # Compute accuracy
            accuracy = f1_score(self.y_test, y_test_pred)
            total_scores.append(accuracy)

        return np.median(total_scores)



class AccuracyScoringApprox():
    def __init__(self, model: DeepNeuralNetwork,
                 X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def compute_score(self) -> float:
        y_test_pred = self.model.predict(self.X_test)
        accuracy = f1_score(self.y_test, y_test_pred)
        return accuracy
