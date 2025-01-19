#!/bin/bash

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/clap/baseline_clap.config"
. $CONFIG_FILE # to load CONDA_ENV
cd $INSTALL_DIR

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# activate env
conda activate $CONDA_ENV

# INSTALL pypi pacakge
pip install msclap