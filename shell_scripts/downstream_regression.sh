#!/bin/bash

# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/regression_vita.yaml"

python3 main.py train -c "$config_file" -m imaging_tabular -g regr -n vita_regression
