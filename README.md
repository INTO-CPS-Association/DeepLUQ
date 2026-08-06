# DeepLUQ: A Python Library for Deep Learning Uncertainty Quantification

## Introduction

DeepLUQ provides methods and metrics for capturing and quantifying uncertainty in deep learning models. It currently implements Monte Carlo Dropout (MC-Dropout) for capturing uncertainty in model predictions, and integrates a range of UQ metrics for the two most common deep learning tasks: regression and classification.

## Installation

DeepLUQ can be installed either from source or from PyPI.

### Option 1: Install from source

Download the latest release from the [releases page](https://github.com/INTO-CPS-Association/DeepLUQ/releases/tag/v0.1.5), then install the downloaded archive:

```bash
pip install deepluq-0.1.5.tar.gz
```

### Option 2: Install from PyPI

Install the latest published version:

```bash
pip install deepluq
```

Or install a specific version:

```bash
pip install deepluq==0.1.5
```

## Case Study Applications

### Assessing the Uncertainty and Robustness of the Laptop Refurbishing Software

Github repository: https://github.com/chengjie-lu/sticker-detection-uncertainty-quantification

Paper: Chengjie Lu, Jiahui Wu, Shaukat Ali, and Mikkel Labori Olsen. "Assessing the Uncertainty and Robustness of the Laptop Refurbishing Software". In 18th IEEE International Conference on Software Testing, Verification and Validation (ICST) 2025. [Preprint](https://arxiv.org/pdf/2409.03782)

### Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots

Github repository: https://github.com/pablovalle/VLA_UQ

Paper: Valle, Pablo, Chengjie Lu, Shaukat Ali, and Aitor Arrieta. "Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots." arXiv preprint arXiv:2507.17049 (2025). [Preprint](https://arxiv.org/pdf/2507.17049)



## References

Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.

Gal, Yarin. "Uncertainty in deep learning." (2016).

Catak, Ferhat Ozgur, Tao Yue, and Shaukat Ali. "Prediction surface uncertainty quantification in object detection models for autonomous driving." 2021 IEEE International Conference on Artificial Intelligence Testing (AITest). IEEE, 2021.
