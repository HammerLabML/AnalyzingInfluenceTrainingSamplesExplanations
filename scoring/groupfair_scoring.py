import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from .scoring import Scoring
from .counterfactuals.cf_memory import MemoryExplainer
from .counterfactuals.cf_proto import ProtoExplainer
from .counterfactuals.cf_wachter import WachterExplainer
from .dnn import DeepNeuralNetwork


class GroupFairCfScoringApprox():
    def __init__(self, model: DeepNeuralNetwork,
                 X_test: np.ndarray, y_sensitive_test: np.ndarray,
                 y_neg_label: int = 0,
                 cf_approx: str = "logits"):
        self.cf_approx = cf_approx
        self.model = model
        self.X_test = X_test
        self.y_sensitive_test = y_sensitive_test
        self.y_neg_label = y_neg_label

    def estimate_cf(self, X, y_target: int):
        input_tensor = tf.Variable(X)
        labels = np.array([y_target] * X.shape[0]).reshape(-1, 1)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(input_tensor)
            logits = self.model.model(input_tensor,)
            loss_value = self.model.loss_fn(labels, logits)

            grads = tape.gradient(loss_value, input_tensor).numpy()
            return grads

    def compute_score(self) -> float:
        y_test_pred = self.model.predict(self.X_test)
        y_test_logits = self.model.model(self.X_test).numpy()

        idx = np.argwhere(y_test_pred == self.y_neg_label).flatten()
        y_test_logits = y_test_logits[idx, :]
        y_sensitive_test = self.y_sensitive_test[idx]

        if self.cf_approx == "grad":
            delta_cf = self.estimate_cf(self.X_test[idx, :], 1 - self.y_neg_label)
            delta_cf_cost = np.linalg.norm(delta_cf, ord=2, axis=1)
        elif self.cf_approx == "logits":
            delta_cf_cost = np.abs(y_test_logits[:, 0] - y_test_logits[:, 1])
        else:
            raise ValueError(self.cf_approx)

        idx_sensitive_0 = np.argwhere(y_sensitive_test == 0).flatten()
        idx_sensitive_1 = np.argwhere(y_sensitive_test == 1).flatten()

        delta_cf_cost_group_0 = delta_cf_cost[idx_sensitive_0]
        delta_cf_cost_group_1 = delta_cf_cost[idx_sensitive_1]

        if len(delta_cf_cost_group_1) == 0 or len(delta_cf_cost_group_0) == 0:
            return 0.

        score_0 = np.max(delta_cf_cost_group_0)
        score_1 = np.max(delta_cf_cost_group_1)
        total_score = np.abs(score_0 - score_1)

        if np.isnan(total_score):
            return 0.
        else:
            return total_score



class GroupFairCfScoring(Scoring):
    def __init__(self, clf, X_test: np.ndarray, y_sensitive_test: np.ndarray,
                 y_neg_label: int = 0, **kwds):
        self.clf = clf
        self.X_test = X_test
        self.y_sensitive_test = y_sensitive_test
        self.y_neg_label = y_neg_label

        super().__init__(**kwds)

    def reset(self):
        pass

    def compute_cf_batch(self, clf, X_test, y_target_label, X_train, y_train):
        raise NotImplementedError()

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

            # Compute CFs for negative predictions
            idx = np.argwhere(y_test_pred == self.y_neg_label).flatten()
            Delta_CF = self.compute_cf_batch(self.clf, self.X_test[idx, :],
                                             1 - self.y_neg_label, X_train, y_train)
            if Delta_CF is None:
                continue

            # Filter out cases where the computation of CF failed!
            idx_new = []
            Delta_CF_new = []
            for i, delta_cf in enumerate(Delta_CF):
                if delta_cf is not None:
                    idx_new.append(idx[i])
                    Delta_CF_new.append(delta_cf)

            idx = np.array(idx_new)
            Delta_CF = np.array(Delta_CF_new)

            # Compare cost of recourse between sensitive groups
            idx_sensitive_0 = np.argwhere(self.y_sensitive_test[idx] == 0).flatten()
            idx_sensitive_1 = np.argwhere(self.y_sensitive_test[idx] == 1).flatten()

            def __cost(delta_cf):
                return np.linalg.norm(delta_cf, ord=1)

            score_0 = np.max([__cost(delta_cf)
                                for delta_cf in Delta_CF[idx_sensitive_0, :]])
            score_1 = np.max([__cost(delta_cf)
                                for delta_cf in Delta_CF[idx_sensitive_1, :]])

            total_score = np.abs(score_0 - score_1)
            total_scores.append(total_score)

        return np.median(total_scores)


class MemoryGroupFairCfScoring(GroupFairCfScoring):
    def compute_cf_batch(self, clf, X_test, y_target_label, X_train, y_train):
        expl = MemoryExplainer(clf, X_train, y_train)
        return [expl.compute_delta_cf(X_test[i, :], y_target_label)
                for i in range(X_test.shape[0])]


class ProtoGroupFairCfScoring(GroupFairCfScoring):
    def compute_cf_batch(self, clf, X_test, y_target_label, X_train, y_train):
        expl = ProtoExplainer(clf, X_train, y_train)
        return [expl.compute_delta_cf(X_test[i, :], y_target_label)
                for i in range(X_test.shape[0])]


class WachterGroupFairCfScoring(GroupFairCfScoring):
    def compute_cf_batch(self, clf, X_test, y_target_label, X_train, y_train):
        expl = WachterExplainer(clf, X_train)
        return [expl.compute_delta_cf(X_test[i, :])
                for i in range(X_test.shape[0])]
