#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

# Generate a json file containing selected feature names
python3 datasets/preprocessing_tabular/selected_feaure_names.py 

# Process (normalize + impute) input tabular data
python3 datasets/preprocessing_tabular/process_input_tabular.py

# Process classification labels based on ICD10
python3 datasets/preprocessing_tabular/process_classification_labels.py
