# Usage: Uncertainty Quantification for Perception Models




## Case Study 1: Laptop Refurbishing Software from DTI

This work employs DeepLUQ to quantify uncertainties in the sticker detection software and provides guidelines for sticker detection software selection based on uncertainty and robustness scores. More details can be found in the accompanying paper (Lu et al., 2025).

### System Requirements

- A machine with an 8-core processor and 16 GB of memory (minimum)
- A machine with a GPU installed is strongly recommended
- Ubuntu 18.04, 20.04, and 22.04 are supported

### Setup


Clone the project (source code available at <https://github.com/chengjie-lu/sticker-detection-uncertainty-quantification>).

Follow the installation and setup instructions in the [Installation Page](https://into-cps-association.github.io/DeepLUQ/docs/install.html).

Install Anaconda:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
source ~/.bashrc
```

Build `torchvision` from source:

```bash
cd vision
# use install instead of develop if you don't care about development.
python setup.py develop
# or, for OSX
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py develop
# for C++ debugging, use DEBUG=1
# DEBUG=1 python setup.py develop
```

### Execute the Sticker Detection Model and Quantify Uncertainties

```bash
python run_cluster.py --uq_method="mc_dropout" \
--model_n="fasterrcnn_resnet50_fpn" --dataset_p="origimg/org" --drop_rate="0.1" \
--save_folder="experiment_results_mc_dropout"
```

The key arguments include:

- `--uq_method`: Set the uncertainty quantification method.
- `--model_n`: Set the model to be evaluated.
- `--dataset_p`: Set the dataset to be used for the evaluation.
- `--drop_rate`: Set the dropout rate.
- `--save_folder`: Set the folder for saving the experiment results.

### Reference

Chengjie Lu, Jiahui Wu, Shaukat Ali, and Mikkel Labori Olsen. "Assessing the Uncertainty and Robustness of the Laptop Refurbishing Software". In 18th IEEE International Conference on Software Testing, Verification and Validation (ICST) 2025. [Preprint](https://arxiv.org/pdf/2409.03782)

## Case Study 2: Anomaly Detector for Turtlebot4

This work quantifies the uncertainty of a Machine Learning (ML)-based anomaly detector for Turtlebot4. The anomaly detector takes lidar readings from the Turtlebot4 and outputs 0 for normal readings and 1 if an anomaly is detected.

### System Requirements

- Windows, Linux, and macOS are all supported

### Setup


Clone the project (source code available at <https://github.com/chengjie-lu/anomoly-detector-uq.git>).

Follow the installation and setup instructions in the [Installation Page](https://into-cps-association.github.io/DeepLUQ/docs/install.html).

Install required libraries:

```bash
cd anomaly_detector
pip install -r requirements.txt
```

### Execute the Anomaly Detector and Quantify Uncertainties

```bash
python anomaly_inference.py --model_path="./anomaly_detector_50.pth" \
--dataset_p="./normal.pickle" \
--drop_rate="0.5"
```

The key arguments include:

- `--model_path`: Set the model to be evaluated.
- `--dataset_p`: Set the dataset to be used for the evaluation.
- `--drop_rate`: Set the dropout rate.

### Acknowledgement

The Anomaly Detector is adopted from [CIDL Group](https://cidl.csd.auth.gr/) from Aristotle University of Thessaloniki.