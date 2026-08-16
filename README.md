# DeepLUQ: A Python Library for Deep Learning Uncertainty Quantification

[![PyPI version](https://img.shields.io/pypi/v/deepluq.svg)](https://pypi.org/project/deepluq/)
[![Python versions](https://img.shields.io/pypi/pyversions/deepluq.svg)](https://pypi.org/project/deepluq/)
[![License](https://img.shields.io/badge/license-INTO--CPS%20Association%20Public%20License-blue.svg)](LICENSE)

## Introduction

DeepLUQ provides methods and metrics for capturing and quantifying uncertainty in deep learning models, as summarized in the figure below. It implements Monte Carlo Dropout and Monte Carlo DropBlock for capturing uncertainty in model predictions, alongside a dedicated UQ method for Vision-Language-Action (VLA) models. On top of these, DeepLUQ integrates:

- Classification metrics (Variation Ratio, Shannon Entropy, Mutual Information) and regression metrics (Total Variance, Prediction Surface) for general deep learning tasks
- UQ metrics tailored to VLA-enabled robotic systems
- Uncertainty-aware mutation scores (Image-level, Object-level, IoU-based, and Uncertainty-Aware Mutation Score) for mutation testing of deep learning models

```{figure} /figs/deepluq.png
:alt: DeepLUQ overview
:width: 100%
:align: center

Overview of the DeepLUQ Library.
```

## Acknowledgement

DeepLUQ is supported by the RoboSAPIENS project funded by the European Commission's Horizon Europe programme under grant agreement number 101133807.

