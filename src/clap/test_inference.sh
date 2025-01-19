#!/bin/bash

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/clap/baseline_clap.config"
. $CONFIG_FILE # to load CONDA_ENV
conda activate $CONDA_ENV
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# LAUNCH
audio_file="../../samples/freesound/784190__klankbeeld__village-bokhoven-nl-0944-am-150405_0586.wav"
python test_inference.py $audio_file
