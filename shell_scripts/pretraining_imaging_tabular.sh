#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/cl_pretraining_imaging_vita.yaml"

python3 main.py train -c "$config_file" -m imaging_tabular -g imaging-tabular -n pretraining_imaging_tabular
