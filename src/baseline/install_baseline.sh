#!/bin/bash

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/baseline/baseline_dcase24.config"
. $CONFIG_FILE # to load CONDA_ENV
cd $BASELINE_INSTALL_DIR

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# activate env
conda activate $CONDA_ENV

# CLONE BASELINE VIA HTTPS, ssh not working
[ -d "dcase2024-task6-baseline" ] && rm -fr dcase2024-task6-baseline
git clone https://github.com/Labbeti/dcase2024-task6-baseline.git

# EDIT SOME STUFF BEFORE INSTALLING
echo "numpy==1.26.4" >> dcase2024-task6-baseline/requirements.txt
CNEXT="dcase2024-task6-baseline/src/dcase24t6/pre_processes/cnext.py"
awk 'NR==40{print "        torch.multiprocessing.set_start_method(\047spawn\047)"}1' $CNEXT > tmp
mv tmp $CNEXT

# INSTALL
cd dcase2024-task6-baseline
pip install -e .
pre-commit install
