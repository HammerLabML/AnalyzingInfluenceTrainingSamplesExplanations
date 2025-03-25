# Analyzing the Influence of Training Samples on Explanations
This repository contains the implementation of the experiments as proposed in the paper [Towards Understanding the Influence of Training Samples on Explanations](paper.pdf) by André Artelt and Barbara Hammer.

## Abstract

Explainable AI (XAI) is widely used to analyze AI systems' decision-making, such as providing counterfactual explanations for recourse. When unexpected explanations occur, users may want to understand the training data properties shaping them.
Under the umbrella of data valuation, first approaches have been proposed that estimate the influence of data samples on a given model. This process not only helps determine the data's value, 
but also offers insights into how individual, potentially noisy, or misleading examples affect a model, which is crucial for interpretable AI. In this work, we apply the concept of data valuation to the significant area of model evaluations, focusing on how individual training samples impact a model's internal reasoning rather than the predictive performance only.
Hence, we introduce the novel problem of identifying training samples shaping a given explanation or related quantity, and investigate the particular case of the cost of computational recourse. We propose an algorithm to identify such influential samples and conduct extensive empirical evaluations in two case studies.

## Experiments

All experiments are implemented in Python and were tested with Python 3.8 -- required dependencies are listed in [REQUIREMENTS.txt](REQUIREMENTS.txt).
The experiments themself are implemented in [experiments.py](experiments.py) and [eval_experiments.py](eval_experiments.py).

The proposed algorithm (Algorithm 1 in the paper) is implemented in [gradient_data_shapley.py](gradient_data_shapley.py).

## License

MIT license - See [LICENSE](LICENSE).

## How to cite

The version in this repository constitutes an extended version of the [IJCAI 2024 XAI workshop version](https://drive.google.com/file/d/1qafiVLvNXPTql9rawyqYJEfGdBFbDQa-/view?usp=sharing).

You can cite either the workshop version or this version on [arXiv](https://arxiv.org/abs/2406.03012).
