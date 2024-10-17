from abc import abstractmethod
import numpy as np


class Scoring():
    def __init__(self):
        pass

    @abstractmethod
    def compute_score(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        raise NotImplementedError()
