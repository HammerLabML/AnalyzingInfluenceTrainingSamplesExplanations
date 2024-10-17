import os
import sys
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

from datasets import load_benchmarkdata
from scoring.dnn import get_model
from gradient_data_shapley import GradientDataShapleyInfluenceScore


if __name__ == "__main__":
    data_desc = sys.argv[1]
    cf_approx = sys.argv[2]
    use_log_reg = sys.argv[3] == "True"
    scoring_desc = sys.argv[4]
    folder_out = sys.argv[5]

    n_reps_per_fold = 5
    n_folds = 5
    n_iter = 40
    n_train_itr = 1
    n_jobs = 12
    eval_on_train_set = False

    print(data_desc, cf_approx, use_log_reg, folder_out)

    # Load data
    X, y, y_sensitive, _ = load_benchmarkdata(data_desc)

    # Run k-fold cross-validation
    X_train_results = []
    y_train_results = []
    y_train_sensitive_results = []
    X_test_results = []
    y_test_results = []
    y_test_sensitive_results = []
    infl_scores_results = []
    infl_scores_rolling_var_results = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        try:
            # Split into train and test set
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_sensitive_train, y_sensitive_test = y_sensitive[train_index], y_sensitive[test_index]

            # Deal with imbalanced data
            sampling = RandomUnderSampler()     # Undersample majority class
            data = np.concatenate((X_train, y_sensitive_train.reshape(-1, 1)), axis=1)
            X_train, y_train = sampling.fit_resample(data, y_train)
            y_sensitive_train = X_train[:, -1].flatten()
            X_train = X_train[:, :-1]

            print(f"Training samples: {X_train.shape}")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            all_infl_scores = []
            for _ in range(n_reps_per_fold):
                # Fit and evaluate model
                clf = get_model((X_train.shape[1],), logreg=use_log_reg)
                clf.fit(X_train, y_train)

                y_train_pred = clf.predict(X_train)
                y_test_pred = clf.predict(X_test)

                print(f"Train: {f1_score(y_train, clf.predict(X_train))}   " +
                      f"Test: {f1_score(y_test, y_test_pred)}")
                print(confusion_matrix(y_test, y_test_pred))

                # Compute influence scores for each training sample with respect to a scoring function (i.e. value function)
                alg = GradientDataShapleyInfluenceScore(X_train, y_train, y_sensitive_train,
                                                        X_test, y_test, y_sensitive_test,
                                                        n_iter=n_iter, n_train_itr=n_train_itr,
                                                        n_jobs=n_jobs, cf_approx=cf_approx,
                                                        scoring_desc=scoring_desc)
                infl_scores, infl_scores_rolling_var = alg.compute_influence_scores(use_log_reg=use_log_reg,
                                                                                    eval_on_train_set=eval_on_train_set)
                all_infl_scores.append(infl_scores)

            all_infl_scores_avg, all_infl_scores_var = np.mean(all_infl_scores, axis=0), \
                np.var(all_infl_scores, axis=0)
            print(f"Var: {np.var(all_infl_scores, axis=0)}")

            infl_scores_results.append(all_infl_scores_avg)
            X_train_results.append(X_train)
            y_train_results.append(y_train)
            X_test_results.append(X_test)
            y_test_results.append(y_test)
            y_train_sensitive_results.append(y_sensitive_train)
            y_test_sensitive_results.append(y_sensitive_test)

            print()
        except Exception as ex:
            print(ex)

    # Store results
    np.savez(os.path.join(folder_out, f"{data_desc}_{cf_approx}_{use_log_reg}_{scoring_desc}"),
             infl_scores_results=np.array(infl_scores_results, dtype=object),
             infl_scores_rolling_var=np.array(infl_scores_rolling_var, dtype=object),
             X_train_results=np.array(X_train_results, dtype=object),
             X_test_results=np.array(X_test_results, dtype=object),
             y_train_results=np.array(y_train_results, dtype=object),
             y_train_sensitive_results=np.array(y_train_sensitive_results, dtype=object),
             y_test_results=np.array(y_test_results, dtype=object),
             y_test_sensitive_results=np.array(y_test_sensitive_results, dtype=object))
