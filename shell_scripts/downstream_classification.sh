#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/classification_vita.yaml"

python3 main.py train -c "$config_file" -m imaging_tabular -g clas -n vita_classification
