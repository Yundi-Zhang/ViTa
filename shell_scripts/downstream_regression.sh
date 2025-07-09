#!/bin/bash

#SBATCH --qos=mcml
#SBATCH --job-name=test
#SBATCH --output=./logs/%x-%A.out
#SBATCH --error=./logs/%x-%A.err
#SBATCH --partition=mcml-hgx-h100-94x4
### mcml-hgx-h100-94x4, lrz-hgx-h100-94x4, mcml-hgx-a100-80x4, mcml-dgx-a100-40x8
#SBATCH --nodes=1                     # Request 2 nodes
#SBATCH --ntasks-per-node=1           # 4 tasks per node (1 task per GPU)
#SBATCH --cpus-per-task=16            # Number of CPUs per task
#SBATCH --gres=gpu:1                  # Request 4 GPUs per node (total 8 GPUs)
#SBATCH --time=2-00:00:00             # Time limit
#SBATCH --mem-per-cpu=8G              # Memory per CPU

####### Setup the environment ########
source ~/miniconda3/bin/activate
conda activate kmae_39
cd /dss/dsshome1/0B/ge82pez2/Projects/VisionLanguageLatent

# Setup the environment
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$(pwd)

config_file="configs/imaging_tabular_model/regression_vita.yaml"

python3 main.py train -c "$config_file" -m imaging_tabular -g regr -n vita_regression
