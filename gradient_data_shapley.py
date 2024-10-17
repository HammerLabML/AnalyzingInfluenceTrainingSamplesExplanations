from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf

from scoring.dnn import get_model
from scoring.groupfair_scoring import GroupFairCfScoringApprox
from scoring.globalcost_scoring import CostRecourseScoringApprox
from scoring.accuracy_scoring import AccuracyScoringApprox


class GradientDataShapleyInfluenceScore():
    def __init__(self, X_train, y_train, y_sensitive_train,
                 X_test, y_test, y_test_sensitive,
                 n_iter: int = 100, n_train_itr: int = 50,
                 n_jobs: int = 1, cf_approx: str = "logits",
                 scoring_desc: str = "globalrecourse"):
        self.n_jobs = n_jobs
        self.scoring_desc = scoring_desc
        self.cf_approx = cf_approx
        self.X_train = X_train
        self.y_train = y_train
        self.y_sensitive_train = y_sensitive_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_sensitive = y_test_sensitive
        self.n_iter = n_iter
        self.n_train_itr = n_train_itr

    def compute_influence_scores(self, use_log_reg=False, eval_on_train_set=False) -> np.ndarray:
        n_samples = self.X_train.shape[0]

        def __run_step(n_iter):
            my_model = get_model((self.X_train.shape[1],), logreg=use_log_reg)

            X_eval, y_eval, y_sensitive_eval = self.X_test, self.y_test, self.y_test_sensitive
            if eval_on_train_set is True:
                X_eval, y_eval, y_sensitive_eval = self.X_train, self.y_train, self.y_sensitive_train

            if self.scoring_desc == "groupfaircf":
                scoring = GroupFairCfScoringApprox(my_model, X_eval, y_sensitive_eval,
                                                   cf_approx=self.cf_approx)
            elif self.scoring_desc == "accuracy":
                scoring = AccuracyScoringApprox(my_model, X_eval, y_eval)
            elif self.scoring_desc == "globalrecourse":
                scoring = CostRecourseScoringApprox(my_model, X_eval, cf_approx=self.cf_approx)
            else:
                raise ValueError()

            phi = np.zeros(n_samples)
            for _ in range(n_iter):
                cur_score = scoring.compute_score()

                perm = np.random.permutation(n_samples)
                for idx in perm:  # Each sample individually
                    # Compute gradients
                    with tf.GradientTape() as tape:
                        logits = my_model.model(self.X_train[idx, :].reshape(1, -1), training=True)

                        loss_value = my_model.loss_fn(self.y_train[idx].reshape(1, -1), logits)
                        grads = tape.gradient(loss_value, my_model.model.trainable_variables)

                    # Apply gradients
                    my_model.optimizer.apply_gradients(zip(grads,
                                                           my_model.model.trainable_variables))

                    # Evaluate influence of current sample
                    new_score = scoring.compute_score()
                    phi[idx] += new_score - cur_score

                    cur_score = new_score

            return phi

        if self.n_jobs == -1 or self.n_jobs > 1:
            v = Parallel(n_jobs=self.n_jobs)(delayed(__run_step)(self.n_train_itr)
                                             for _ in range(self.n_iter))
        else:
            v = []
            for _ in range(self.n_iter):
                v_t = __run_step(self.n_train_itr)
                v.append(v_t)

        rolling_var = [np.var(v[:n], axis=0) for n in range(2, len(v))]
        return np.mean(v, axis=0), rolling_var
