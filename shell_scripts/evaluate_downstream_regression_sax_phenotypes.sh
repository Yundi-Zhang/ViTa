#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/regression_vita.yaml"

python3 run_evaluation.py \
    --base_config "$config_file" \
    --updates '{"general.resume_training": true, "general.resume_ckpt_path": "checkpoints_imaging_tabular/downstream_pred_vita_allphenotypes_sax.ckpt", "module.tabular_hparams.selected_features": ["LVEDV (mL)", "LVESV (mL)", "LVSV (mL)", "LVEF (%)", "LVCO (L/min)", "LVM (g)", "RVEDV (mL)", "RVESV (mL)", "RVSV (mL)", "RVEF (%)"]}' \
    --mode "imaging_tabular" \
    --wandb_group "regr" \
    --wandb_name "vita_regression_sax_phenotypes"
