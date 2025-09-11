#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/cl_pretraining_imaging_vita.yaml"

python3 run_evaluation.py \
    --base_config "$config_file" \
    --updates '{"general.resume_training": true, "general.resume_ckpt_path": "checkpoints_imaging_tabular/pretrain_vita.ckpt", "data.validation_data.selected_cols": ["LVM (g)"], "data.validation_data.labels": "raw_tab.csv"}' \
    --mode "imaging_tabular" \
    --wandb_group "imaging-tabular" \
    --wandb_name "pretraining_imaging_tabular"
