import os
import sys

job_id = int(sys.argv[1]) - 1
print(f"Starting job: {job_id}")

jobs = [
"python eval_experiments.py exp-results/diabetes_logits_False_globalrecourse.npz globalrecourse wachter False False > exp-eval-results/diabetes_logits_False_wachter_globalrecourse.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_globalrecourse.npz globalrecourse wachter True False > exp-eval-results/diabetes_logits_True_wachter_globalrecourse.txt",
"python eval_experiments.py exp-results/german_logits_False_globalrecourse.npz globalrecourse wachter False False > exp-eval-results/german_logits_False_wachter_globalrecourse.txt",
"python eval_experiments.py exp-results/german_logits_True_globalrecourse.npz globalrecourse wachter True False > exp-eval-results/german_logits_True_wachter_globalrecourse.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_accuracy.npz globalrecourse wachter False False > exp-eval-results/diabetes_logits_False_wachter_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_accuracy.npz globalrecourse wachter True False > exp-eval-results/diabetes_logits_True_wachter_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_False_accuracy.npz globalrecourse wachter False False > exp-eval-results/german_logits_False_wachter_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_True_accuracy.npz globalrecourse wachter True False > exp-eval-results/german_logits_True_wachter_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_globalrecourse.npz globalrecourse wachter False True > exp-eval-results/diabetes_logits_False_wachter_globalrecourse_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_globalrecourse.npz globalrecourse wachter True True > exp-eval-results/diabetes_logits_True_wachter_globalrecourse_random.txt",
"python eval_experiments.py exp-results/german_logits_False_globalrecourse.npz globalrecourse wachter False True > exp-eval-results/german_logits_False_wachter_globalrecourse_random.txt",
"python eval_experiments.py exp-results/german_logits_True_globalrecourse.npz globalrecourse wachter True True > exp-eval-results/german_logits_True_wachter_globalrecourse_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_groupfaircf.npz groupfaircf wachter False False > exp-eval-results/diabetes_logits_False_wachter_groupfaircf.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_groupfaircf.npz groupfaircf wachter True False > exp-eval-results/diabetes_logits_True_wachter_groupfaircf.txt",
"python eval_experiments.py exp-results/german_logits_False_groupfaircf.npz groupfaircf wachter False False > exp-eval-results/german_logits_False_wachter_groupfaircf.txt",
"python eval_experiments.py exp-results/german_logits_True_groupfaircf.npz groupfaircf wachter True False > exp-eval-results/german_logits_True_wachter_groupfaircf.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_accuracy.npz groupfaircf wachter False False > exp-eval-results/diabetes_logits_False_wachter_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_accuracy.npz groupfaircf wachter True False > exp-eval-results/diabetes_logits_True_wachter_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_False_accuracy.npz groupfaircf wachter False False > exp-eval-results/german_logits_False_wachter_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_True_accuracy.npz groupfaircf wachter True False > exp-eval-results/german_logits_True_wachter_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_groupfaircf.npz groupfaircf wachter False True > exp-eval-results/diabetes_logits_False_wachter_groupfaircf_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_groupfaircf.npz groupfaircf wachter True True > exp-eval-results/diabetes_logits_True_wachter_groupfaircf_random.txt",
"python eval_experiments.py exp-results/german_logits_False_groupfaircf.npz groupfaircf wachter False True > exp-eval-results/german_logits_False_wachter_groupfaircf_random.txt",
"python eval_experiments.py exp-results/german_logits_True_groupfaircf.npz groupfaircf wachter True True > exp-eval-results/german_logits_True_wachter_groupfaircf_random.txt"
]

os.system(jobs[job_id])
