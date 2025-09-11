#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/regression_vita.yaml"

python3 run_evaluation.py \
    --base_config "$config_file" \
    --updates '{"general.resume_training": true, "general.resume_ckpt_path": "checkpoints_imaging_tabular/downstream_pred_vita_allindicators.ckpt", "module.tabular_hparams.selected_features": ["Systolic blood pressure-2.mean","Diastolic blood pressure-2.mean","Pulse rate-2.mean","Body fat percentage-2.0","Whole body water mass-2.0","Body mass index (BMI)-2.0","Waist circumference-2.0","Height-2.0","Weight-2.0","Cardiac index-2.0","Average heart rate-2.0","Systolic brachial blood pressure during PWA-2.0","End systolic pressure during PWA-2.0","Stroke volume during PWA-2.0","Mean arterial pressure during PWA-2.0","Sleep duration-2.0"]}' \
    --mode "imaging_tabular" \
    --wandb_group "regr" \
    --wandb_name "vita_regression_indicators"
