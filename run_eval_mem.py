import os
import sys

job_id = int(sys.argv[1]) - 1
print(f"Starting job: {job_id}")

jobs = [
"python eval_experiments.py exp-results/diabetes_logits_False_globalrecourse.npz globalrecourse mem False False > exp-eval-results/diabetes_logits_False_mem_globalrecourse.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_globalrecourse.npz globalrecourse mem True False > exp-eval-results/diabetes_logits_True_mem_globalrecourse.txt",
"python eval_experiments.py exp-results/german_logits_False_globalrecourse.npz globalrecourse mem False False > exp-eval-results/german_logits_False_mem_globalrecourse.txt",
"python eval_experiments.py exp-results/german_logits_True_globalrecourse.npz globalrecourse mem True False > exp-eval-results/german_logits_True_mem_globalrecourse.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_accuracy.npz globalrecourse mem False False > exp-eval-results/diabetes_logits_False_mem_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_accuracy.npz globalrecourse mem True False > exp-eval-results/diabetes_logits_True_mem_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_False_accuracy.npz globalrecourse mem False False > exp-eval-results/german_logits_False_mem_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_True_accuracy.npz globalrecourse mem True False > exp-eval-results/german_logits_True_mem_globalrecourse_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_globalrecourse.npz globalrecourse mem False True > exp-eval-results/diabetes_logits_False_mem_globalrecourse_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_globalrecourse.npz globalrecourse mem True True > exp-eval-results/diabetes_logits_True_mem_globalrecourse_random.txt",
"python eval_experiments.py exp-results/german_logits_False_globalrecourse.npz globalrecourse mem False True > exp-eval-results/german_logits_False_mem_globalrecourse_random.txt",
"python eval_experiments.py exp-results/german_logits_True_globalrecourse.npz globalrecourse mem True True > exp-eval-results/german_logits_True_mem_globalrecourse_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_groupfaircf.npz groupfaircf mem False False > exp-eval-results/diabetes_logits_False_mem_groupfaircf.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_groupfaircf.npz groupfaircf mem True False > exp-eval-results/diabetes_logits_True_mem_groupfaircf.txt",
"python eval_experiments.py exp-results/german_logits_False_groupfaircf.npz groupfaircf mem False False > exp-eval-results/german_logits_False_mem_groupfaircf.txt",
"python eval_experiments.py exp-results/german_logits_True_groupfaircf.npz groupfaircf mem True False > exp-eval-results/german_logits_True_mem_groupfaircf.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_accuracy.npz groupfaircf mem False False > exp-eval-results/diabetes_logits_False_mem_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_accuracy.npz groupfaircf mem True False > exp-eval-results/diabetes_logits_True_mem_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_False_accuracy.npz groupfaircf mem False False > exp-eval-results/german_logits_False_mem_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/german_logits_True_accuracy.npz groupfaircf mem True False > exp-eval-results/german_logits_True_mem_groupfaircf_accuracy.txt",
"python eval_experiments.py exp-results/diabetes_logits_False_groupfaircf.npz groupfaircf mem False True > exp-eval-results/diabetes_logits_False_mem_groupfaircf_random.txt",
"python eval_experiments.py exp-results/diabetes_logits_True_groupfaircf.npz groupfaircf mem True True > exp-eval-results/diabetes_logits_True_mem_groupfaircf_random.txt",
"python eval_experiments.py exp-results/german_logits_False_groupfaircf.npz groupfaircf mem False True > exp-eval-results/german_logits_False_mem_groupfaircf_random.txt",
"python eval_experiments.py exp-results/german_logits_True_groupfaircf.npz groupfaircf mem True True > exp-eval-results/german_logits_True_mem_groupfaircf_random.txt"]

os.system(jobs[job_id])
