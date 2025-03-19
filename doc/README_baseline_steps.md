## Getting started

### Prerequisites

This project is founded on cloud-based infrastructure, specifically Google Cloud, to handle the extensive computational requirements associated with the large dataset used. Due to the substantial size of the dataset and the complexity of model training, the project utilizes Google Cloud's Virtual Machines (VMs) with specialized GPU support for efficient processing.

#### Hardware
The machine configuration is as follows:

- **Machine Type:** g2-standard-4  
This machine type is equipped with 4 vCPUs and 16 GB of memory, offering an appropriate balance of resources for handling data preprocessing and model training tasks.

- **GPU:** 1 x NVIDIA L4 GPU
The NVIDIA L4 GPU was chosen for its optimized performance in deep learning tasks, ensuring fast training and inference times for large models and datasets.

- **Architecture:** x86-64
The x86-64 architecture ensures compatibility with most modern computational frameworks and libraries used in machine learning and deep learning tasks.


#### Software setup

For software, the project uses Google Cloud's environment alongside Python-based frameworks and libraries. Below are the steps to set up the project environment:

1. **Java Installation**

```
sudo apt update
sudo apt install default-jdk
```
2. **DSCASE 2024 code modification**
A slight modification in the codebase is needed to avoid potential issues with multiprocessing in PyTorch. Specifically, add the following line after line 39 in the `cnext.py` script:

```
torch.multiprocessing.set_start_method('spawn')
```

3. **Creating the virtual environment**
The project utilizes a conda environment for managing dependencies. To create the environment, run:
```
conda create -n env_dcase24 python=3.11
conda activate env_dcase24
```

4. **Downgrading pip**
After activating the virtual environment, downgrade `pip` to a specific version for compatibility:

```
python -m pip install --force-reinstall pip==24.0
```

5. **Install project dependencies**
The required Python packages are listed in `requirements.txt`, which should be installed by running:
```
pip install -r requirements.txt
```

The ```requirements.txt``` file includes essential dependencies for AAC model training and evaluation. These include:
```
aac-datasets==0.5.0
aac-metrics==0.5.3
black==24.2.0
codecarbon==2.3.4
deepspeed==0.13.1
flake8==7.0.0
hydra-colorlog==1.2.0
hydra-core==1.3.2
ipykernel==6.29.3
ipython==8.22.1
lightning==2.2.0
nltk==3.8.1
numpy==1.26.4
pre-commit==3.6.2
pytest==8.0.2
PyYAML==6.0.1
tensorboard==2.16.2
tokenizers==0.15.2
torch==2.2.1
torchlibrosa==0.1.0
torchoutil[extras]==0.3.0
```
