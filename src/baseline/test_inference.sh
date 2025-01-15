#!/bin/basg

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/baseline/baseline_dcase24.config"
. $CONFIG_FILE # to load CONDA_ENV
conda activate $CONDA_ENV
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# LAUNCH
python test_inference.py
