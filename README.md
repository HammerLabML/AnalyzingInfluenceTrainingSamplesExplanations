# Analyzing the Influence of Training Samples on Explanations
This repository contains the implementation of the experiments as proposed in the paper "Analyzing the Influence of Training Samples on Explanations" by André Artelt and Barbara Hammer.

## Abstract

EXplainable AI (XAI) constitutes a popular method to analyze the reasoning of AI systems by explaining their decision-making, e.g. providing a counterfactual explanation of how to achieve recourse. However, in cases such as unexpected explanations, the user might be interested in learning about the cause of this explanation -- e.g. properties of the utilized training data that are responsible for the observed explanation.
    
Under the umbrella of data valuation, first approaches have been proposed that estimate the influence of data samples on a given model. In this work, we take a slightly different stance, as we are interested in the influence of single training samples on a model explanation rather than the model itself. Hence, we propose the novel problem of identifying training data samples that have a high influence on a given explanation (or related quantity) and investigate the particular case of the cost of computational recourse. For this, we propose an algorithm that identifies such influential training samples.

## Experiments

All experiments are implemented in Python and were tested with Python 3.8 -- required dependencies are listed in [REQUIREMENTS.txt](REQUIREMENTS.txt).
The experiments themself are implemented in [experiments.py](experiments.py) and [eval_experiments.py](eval_experiments.py).

The proposed algorithm (Algorithm 1 in the paper) is implemented in [gradient_data_shapley.py](gradient_data_shapley.py).

## License

MIT license - See [LICENSE](LICENSE).

## How to cite

The version in this repository constitutes an extended version of the [IJCAI 2024 XAI workshop version](https://drive.google.com/file/d/1qafiVLvNXPTql9rawyqYJEfGdBFbDQa-/view?usp=sharing).

You can cite either the workshop version or this version on [arXiv](https://arxiv.org/abs/2406.03012).
