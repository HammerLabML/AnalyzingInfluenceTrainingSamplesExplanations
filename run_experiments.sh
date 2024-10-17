#!/bin/bash
python experiments.py diabetes logits True accuracy exp-results > exp-results/accuracy_diabetes_logits_logreg.txt
python exp-decrease_global.py german logits True accuracy exp-results > exp-results/accuracy_german_logits_logreg.txt
python exp-decrease_global.py diabetes logits False accuracy exp-results > exp-results/accuracy_diabetes_logits_dnn.txt
python exp-decrease_global.py german logits False accuracy exp-results > exp-results/accuracy_german_logits_dnn.txt

python experiments diabetes logits True groupfaircf exp-results > exp-results/groupfaircf_diabetes_logits_logreg.txt
python experiments german logits True groupfaircf exp-results > exp-results/groupfaircf_german_logits_logreg.txt
python experiments diabetes logits False groupfaircf exp-results > exp-results/groupfaircf_diabetes_logits_dnn.txt
python experiments german logits False groupfaircf exp-results > exp-results/groupfaircf_german_logits_dnn.txt

python experiments diabetes logits True globalrecourse exp-results > exp-results/globalrecourse_diabetes_logits_logreg.txt
python experiments german logits True globalrecourse exp-results > exp-results/globalrecourse_german_logits_logreg.txt
python experiments diabetes logits False globalrecourse exp-results > exp-results/globalrecourse_diabetes_logits_dnn.txt
python experiments german logits False globalrecourse exp-results > exp-results/globalrecourse_german_logits_dnn.txt