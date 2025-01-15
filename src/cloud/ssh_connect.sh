#!/bin/bash

# PARAMETERS
KEY_PATH=$1
USER=$2
IP_ADDRESS=$3

# SSH COMMAND
cmd="ssh -i ${KEY_PATH} ${USER}@${IP_ADDRESS}"
echo -e "\n$cmd\n"
eval $cmd