#!/bin/bash


# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/regression_vita.yaml"

python3 run_evaluation.py \
    --base_config "$config_file" \
    --updates '{"general.resume_training": true, "general.resume_ckpt_path": "checkpoints_imaging_tabular/downstream_pred_vita_allphenotypes_lax.ckpt", "module.tabular_hparams.selected_features": ["LAV max (mL)", "LAV min (mL)", "LASV (mL)", "LAEF (%)", "RAV max (mL)", "RAV min (mL)", "RASV (mL)", "RAEF (%)"]}' \
    --mode "imaging_tabular" \
    --wandb_group "regr" \
    --wandb_name "vita_regression_lax_phenotypes"
