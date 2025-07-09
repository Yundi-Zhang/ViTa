#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_model/segmentation_vita.yaml"

python3 main.py train -c "$config_file" -m imaging -g seg -n vita_segmentation
