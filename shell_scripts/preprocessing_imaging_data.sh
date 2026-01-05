#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

# Process images
python3 datasets/preprocessing_imaging/process_images.py
