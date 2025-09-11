#!/bin/bash

export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

# Disease name
disease_name="CAD" # ["CAD", "Stroke", "Infarct", "diabetes", "High_blood_pressure", "Hypertension"]

# Derived paths
ckpt_name="downstream_clas_vita_${disease_name,,}.ckpt"
cmr_pickle="recon_cmr_subject_paths_50k_${disease_name}.pkl"
raw_tab="labels_${disease_name}.csv"

# Selected features as JSON array
selected_features="[\"Diagnosed_${disease_name}\"]"

config_file="configs/imaging_tabular_model/classification_vita.yaml"

# Build JSON manually
updates_json=$(cat <<EOF
{
  "general.resume_training": true,
  "general.resume_ckpt_path": "checkpoints_imaging_tabular/$ckpt_name",
  "data.imaging_data.cmr_path_pickle_name": "$cmr_pickle",
  "data.tabular_data.raw_tabular_data": "$raw_tab",
  "module.tabular_hparams.selected_features": $selected_features
}
EOF
)

# Run evaluation
python3 run_evaluation.py \
  --base_config "$config_file" \
  --updates "$updates_json" \
  --mode "imaging_tabular" \
  --wandb_group "clas" \
  --wandb_name "vita_classification"
