#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_model/segmentation_vita.yaml"

python3 run_evaluation.py \
    --base_config "$config_file" \
    --updates '{"general.resume_training": true, "general.resume_ckpt_path": "checkpoints_imaging/downstream_seg_mae_allax.ckpt"}' \
    --mode "imaging" \
    --wandb_group "seg" \
    --wandb_name "vita_segmentation"
