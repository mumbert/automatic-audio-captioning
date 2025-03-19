#!/bin/bash

# LOAD CONFIG AND ENV
CONFIG_FILE="$HOME/automatic-audio-captioning/config/clap/baseline_clap.config"
. $CONFIG_FILE # to load CONDA_ENV
cd $INSTALL_DIR

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# activate env
conda activate $CONDA_ENV

# INSTALL pypi pacakges
pip install msclap
pip install aac-datasets
pip install aac-metrics

# for some reason this is also necessary to avoi errors
aac-metrics-download

# install Java 11 in Debian 11: https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-debian-11
sudo apt update
sudo apt install default-jre
java -version

# Validate version, should be >= 8 and <= 13
link="https://github.com/Labbeti/aac-metrics/blob/b2e4ace787bef36577935605faeef74bbeffcf15/src/aac_metrics/utils/checks.py#L16"
java -version
JAVA_VER=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | awk -F '.' '{sub("^$", "0", $2); print $1}')
if [[ "$JAVA_VER" -ge 8 ]] && [[ "$JAVA_VER" -le 13 ]]
then
    echo "Java version [$JAVA_VER] is within the expected range [8,13]"
else 
    echo -e "\nERROR: Java version [$JAVA_VER] is outside the expected range [8,13]. This affects aac-metrics, check $link"
fi
