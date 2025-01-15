#!/bin/bash

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/baseline/baseline_dcase24.config"
. $CONFIG_FILE # to load CONDA_ENV
conda activate $CONDA_ENV
cd $BASELINE_INSTALL_DIR/dcase2024-task6-baseline

# LAUNCH
dcase24t6-prepare