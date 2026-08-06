# Usage: Uncertainty Quantification for VLA-enabled Robots

Follow the installation and setup instructions in the [Installation Page](https://into-cps-association.github.io/DeepLUQ/docs/install.html).

## Case Study: VLA-enabled Robotic Systems

### System Requirements

- A machine with at least an 8-core processor and 16 GB of RAM
- A dedicated GPU with a minimum of 16 GB of memory
- Ubuntu 20.04 or Ubuntu 22.04

### Setup

Clone this repo:

```bash
git clone https://github.com/pablovalle/VLA_UQ.git
```

Create a virtual environment and install the necessary dependencies:

```bash
# Create an Anaconda environment (Python 3.10 or above):
conda create -n <env_name> python=3.10
conda activate <env_name>

# Install ManiSkill2 real-to-sim environments and their dependencies:
pip install numpy==1.24.4
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
cd {this_repo}
pip install -e .
sudo apt install ffmpeg
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1  # tensorflow gpu support
pip install git+https://github.com/nathanrooy/simulated-annealing
pip install torch==2.3.1 torchvision==0.18.1 timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
```

### Configure VLA (OpenVLA as Example)

Refer to <https://github.com/pablovalle/VLA_UQ> for configuration guidelines covering more VLA models.

```bash
conda activate <env_name>
cd {this_repo}/transformers-4.40.1
pip install -e .
cd {this_repo}/checkpoints
python download_model.py openvla/openvla-7b
cd {this_repo}/checkpoints
cp modeling_prismatic openvla-7b
```

### Execute VLAs

```bash
cd experiments
# available models: openvla-7b, pi0, spatialvla-4b
./run_UQ_exp.sh <env_name> <model_name>
```