#!/bin/bash

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/baseline/baseline_dcase24.config"
. $CONFIG_FILE # to load CONDA_ENV and PY_VERSION

# STEPS
conda activate base
conda remove -n $CONDA_ENV --all -y
conda create --name $CONDA_ENV python=$PY_VERSION -y

# PIP VERSION
conda activate $CONDA_ENV
pip install --force-reinstall pip==23.0

# HELPER COMMANDS TO CONNECT TO THE ENV
conda env list
echo "conda activate $CONDA_ENV"
