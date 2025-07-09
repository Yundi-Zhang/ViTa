#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_model/pretraining_reconstruction_mae.yaml"

python3 main.py train -c "$config_file" -m imaging -g imaging_recon -n pretraining_imaging
