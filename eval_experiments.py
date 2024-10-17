import os
import sys
import random
import math
import numpy as np
from sklearn.metrics import f1_score

from scoring.dnn import get_model
import scoring
import scoring.globalcost_scoring
import scoring.groupfair_scoring
from utils import downsample


if __name__ == "__main__":
    folder_out = "exp-eval-results"

    f_in = sys.argv[1]
    scoring_desc = sys.argv[2]
    cf_desc = sys.argv[3]
    use_log_reg = sys.argv[4] == "True"
    random_removal = True if sys.argv[5] == "True" else False

    use_scikit_learn = False
    if scoring_desc == "groupfaircf":
        if cf_desc == "mem":
            cf_method = scoring.groupfair_scoring.MemoryGroupFairCfScoring
        elif cf_desc == "wachter":
            cf_method = scoring.groupfair_scoring.WachterGroupFairCfScoring
            use_scikit_learn = True
        elif cf_desc == "proto":
            cf_method = scoring.groupfair_scoring.ProtoGroupFairCfScoring
            use_scikit_learn = True
        else:
            raise ValueError("Unknown CF method!")
    elif scoring_desc == "globalrecourse":
        if cf_desc == "mem":
            cf_method = scoring.globalcost_scoring.MemoryCostResourceScoring
        elif cf_desc == "wachter":
            cf_method = scoring.globalcost_scoring.WachterCostResourceScoring
            use_scikit_learn = True
        elif cf_desc == "proto":
            cf_method = scoring.globalcost_scoring.ProtoCostResourceScoring
            use_scikit_learn = True
        else:
            raise ValueError("Unknown CF method!")

    n_scoring_itr = 20
    n_fit_itr = 2
    random_removal_n_itr = 5
    use_train_eval = False

    print(folder_out, f_in, n_scoring_itr, n_fit_itr, cf_desc, scoring_desc, f"random_revmoal={random_removal}", f"logreg={use_log_reg}")

    data = np.load(os.path.join(f_in), allow_pickle=True)
    if len(data["X_train_results"].shape) > 1:
        X_train_results = data["X_train_results"].astype('float32')
        y_train_results = data["y_train_results"].astype('int32')
        y_train_sensitive_results = data["y_train_sensitive_results"]
    else:
        X_train_results = [a.astype('float32') for a in data["X_train_results"].tolist()]
        y_train_results = [a.astype('int32') for a in data["y_train_results"].tolist()]
        y_train_sensitive_results = data["y_train_sensitive_results"].tolist()

    t = data["X_test_results"]
    if len(data["X_test_results"].shape) > 1:
        X_test_results = data["X_test_results"].astype('float32')
        y_test_results = data["y_test_results"].astype('int32')
        y_test_sensitive_results = data["y_test_sensitive_results"]
    else:
        X_test_results = [a.astype('float32') for a in data["X_test_results"].tolist()]
        y_test_results = [a.astype('int32') for a in data["y_test_results"].tolist()]
        y_test_sensitive_results = data["y_test_sensitive_results"].tolist()

    infl_scores_results = data["infl_scores_results"].tolist()

    results = []
    for X_train, y_train, y_train_sensitive, X_test, y_test, y_test_sensitive, infl_scores in \
            zip(X_train_results, y_train_results, y_train_sensitive_results, X_test_results,
                y_test_results, y_test_sensitive_results, infl_scores_results):
        if not isinstance(infl_scores, np.ndarray):
            infl_scores = np.array(infl_scores)

        # Compute initial predictive accuracy!
        pred_scores = []
        for _ in range(n_fit_itr):
            clf = get_model((X_train.shape[1],), use_scikit_learn=use_scikit_learn,
                            logreg=use_log_reg)
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            pred_score = f1_score(y_test, y_test_pred)
            pred_scores.append(pred_score)

        orig_pred_score = np.mean(pred_scores)
        print(f"Test data: {X_test.shape}")
        print(f"F1-score: {np.mean(pred_scores), np.var(pred_scores)}")

        # Compute initial score using all training sampes
        X_eval, y_eval_sensitive = X_test, y_test_sensitive
        if use_train_eval is True:
            X_eval, y_eval_sensitive = X_train, y_train_sensitive

        s = cf_method(get_model((X_test.shape[1],), use_scikit_learn=use_scikit_learn,
                                logreg=use_log_reg), X_eval, y_eval_sensitive)
        s_orig = s.compute_score(X_train, y_train, n_itr=n_scoring_itr)

        # Evaluation
        idx = np.argsort(infl_scores)
        infl_scores_sorted = infl_scores[idx]
        max_n_samples_drop = np.sum(infl_scores_sorted > 0)
        for k in [0.01, 0.02, 0.03, 0.04, .05, .1, .15, .2, .25, .3]:
            n_sampels_drop = min([math.ceil(len(idx) * k), max_n_samples_drop])   # Convert to percentage but do not remove points with a negative impact on the value function

            if random_removal is False:
                # Remove most influential samples
                idx_ = idx[: len(idx) - n_sampels_drop]
                X_train_ = X_train[idx_, :]
                y_train_ = y_train[idx_]

                # Recompute predictive accuracy and fairness scores
                pred_scores = []
                for _ in range(n_fit_itr):
                    clf = get_model((X_train_.shape[1],), use_scikit_learn=use_scikit_learn,
                                    logreg=use_log_reg)
                    clf.fit(X_train_, y_train_)
                    y_test_pred = clf.predict(X_test)
                    pred_score = f1_score(y_test, y_test_pred)
                    pred_scores.append(pred_score)
                new_pred_score = np.mean(pred_scores)

                s_new = s.compute_score(X_train_, y_train_, n_itr=n_scoring_itr)
                print(f"{k} ({n_sampels_drop}): Value: {s_orig} -> {s_new} " +
                    f"F1-Score: {orig_pred_score} -> {new_pred_score}")
                results.append((k, (s_orig, s_new), (orig_pred_score, new_pred_score)))
            else:
                # Remove random samples and recompute metrics -- repeat this multiple times!
                pred_scores = []
                scores_new = []
                for _ in range(random_removal_n_itr):
                    idx_ = random.sample(list(idx), k=len(idx) - n_sampels_drop)
                    X_train_ = X_train[idx_, :]
                    y_train_ = y_train[idx_]

                    pred_scores = []
                    for _ in range(n_fit_itr):
                        clf = get_model((X_train_.shape[1],), use_scikit_learn=use_scikit_learn)
                        clf.fit(X_train_, y_train_)
                        y_test_pred = clf.predict(X_test)
                        pred_score = f1_score(y_test, y_test_pred)
                        pred_scores.append(pred_score)

                    s_new = s.compute_score(X_train_, y_train_, n_itr=n_scoring_itr)
                    scores_new.append(s_new)

                # Aggregate results
                new_pred_score = np.mean(pred_scores)

                s_new = np.mean(scores_new)
                print(s_new, np.var(scores_new))
                print(f"{k} ({n_sampels_drop}): Value: {s_orig} -> {s_new} " +
                    f"F1-Score: {orig_pred_score} -> {new_pred_score}")
                results.append((k, (s_orig, s_new), (orig_pred_score, new_pred_score)))

    print(results)

    f_ext = ""
    if f_in.endswith("accuracy.npz"):
        f_ext = "_accuracy"
    np.savez(os.path.join(folder_out, f"{os.path.basename(f_in)}_{cf_desc}_{scoring_desc}_logreg={use_log_reg}_randomremoval={random_removal}{f_ext}"),
             results=np.array(results, dtype=object),
             infl_scores_results=np.array(infl_scores_results, dtype=object))
