import os
import sys

job_id = int(sys.argv[1]) - 1
print(f"Starting job: {job_id}")

jobs = [
"python eval_experiments.py exp-results/diabetes_logits_False_globalrecourse.npz globalrecourse proto False False > exp-eval-results/diabetes_logits_False_proto_globalrecourse.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_globalrecourse.npz globalrecourse proto True False > exp-eval-results/diabetes_logits_True_proto_globalrecourse.txt",
"python eval_experiments.py exp-results/german_logits_False_globalrecourse.npz globalrecourse proto False False > exp-eval-results/german_logits_False_proto_globalrecourse.txt",
"python eval_experiments.py exp-results/german_logits_True_globalrecourse.npz globalrecourse proto True False > exp-eval-results/german_logits_True_proto_globalrecourse.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_accuracy.npz globalrecourse proto False False > exp-eval-results/diabetes_logits_False_proto_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_accuracy.npz globalrecourse proto True False > exp-eval-results/diabetes_logits_True_proto_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_False_accuracy.npz globalrecourse proto False False > exp-eval-results/german_logits_False_proto_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_True_accuracy.npz globalrecourse proto True False > exp-eval-results/german_logits_True_proto_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_globalrecourse.npz globalrecourse proto False True > exp-eval-results/diabetes_logits_False_proto_globalrecourse_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_globalrecourse.npz globalrecourse proto True True > exp-eval-results/diabetes_logits_True_proto_globalrecourse_random.txt",
"python eval_experiments.py exp-results/german_logits_False_globalrecourse.npz globalrecourse proto False True > exp-eval-results/german_logits_False_proto_globalrecourse_random.txt",
"python eval_experiments.py exp-results/german_logits_True_globalrecourse.npz globalrecourse proto True True > exp-eval-results/german_logits_True_proto_globalrecourse_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_groupfaircf.npz groupfaircf proto False False > exp-eval-results/diabetes_logits_False_proto_groupfaircf.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_groupfaircf.npz groupfaircf proto True False > exp-eval-results/diabetes_logits_True_proto_groupfaircf.txt",
"python eval_experiments.py exp-results/german_logits_False_groupfaircf.npz groupfaircf proto False False > exp-eval-results/german_logits_False_proto_groupfaircf.txt",
"python eval_experiments.py exp-results/german_logits_True_groupfaircf.npz groupfaircf proto True False > exp-eval-results/german_logits_True_proto_groupfaircf.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_accuracy.npz groupfaircf proto False False > exp-eval-results/diabetes_logits_False_proto_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_accuracy.npz groupfaircf proto True False > exp-eval-results/diabetes_logits_True_proto_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_False_accuracy.npz groupfaircf proto False False > exp-eval-results/german_logits_False_proto_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_True_accuracy.npz groupfaircf proto True False > exp-eval-results/german_logits_True_proto_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_groupfaircf.npz groupfaircf proto False True > exp-eval-results/diabetes_logits_False_proto_groupfaircf_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_groupfaircf.npz groupfaircf proto True True > exp-eval-results/diabetes_logits_True_proto_groupfaircf_random.txt",
"python eval_experiments.py exp-results/german_logits_False_groupfaircf.npz groupfaircf proto False True > exp-eval-results/german_logits_False_proto_groupfaircf_random.txt",
"python eval_experiments.py exp-results/german_logits_True_groupfaircf.npz groupfaircf proto True True > exp-eval-results/german_logits_True_proto_groupfaircf_random.txt"
]

os.system(jobs[job_id])
