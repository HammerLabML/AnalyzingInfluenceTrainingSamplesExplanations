import numpy as np
import tensorflow as tf

from alibi.explainers import Counterfactual


class WachterExplainer():
    def __init__(self, model, X_train):
        tf.compat.v1.disable_v2_behavior()
        tf.keras.backend.clear_session()    # Reset!

        self.cf = Counterfactual(model.predict_proba, (1, X_train.shape[1]),
                                 distance_fn='l1', target_proba=.9,
                                 target_class='other', max_iter=20, early_stop=10,
                                 lam_init=1e-1, max_lam_steps=2, tol=0.05, learning_rate_init=0.1,
                                 feature_range=(np.min(X_train), np.max(X_train)), eps=0.01,
                                 init='identity', decay=True, write_dir=None, debug=False)

    def compute_delta_cf(self, x):
        try:
            expl = self.cf.explain(x.reshape(1, -1))
            if expl.cf["class"] == expl.orig_class:
                return None
            else:
                return x - expl.cf["X"].flatten()
        except Exception:
            return None
