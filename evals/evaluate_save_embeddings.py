import os
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description="Run main.py for multiple segmentation checkpoints.")
parser.add_argument("--ckpt_file", type=str, required=True, help="Path to checkpoint JSON file.")
parser.add_argument("--task_name", type=str, required=True, help="Name of the task to pass to main.py.")
parser.add_argument("--save_path", type=str, required=True, help="Path to saved embeddings")
parser.add_argument("--embedding_type", type=str, required=True, help="The type of embeddings to save")

args = parser.parse_args()

ckpt_file_path = args.ckpt_file
task_name = args.task_name
save_path = args.save_path
embedding_type = args.embedding_type

output_config_dir = os.getenv("LOG_FOLDER")

main_py_command = ["python3", "evals/save_embeddings.py", "test", "-m", "imaging_tabular", "-g", "save_emb"]

# Ensure output dirs exist
Path(output_config_dir).mkdir(parents=True, exist_ok=True)

# Load checkpoint mapping
with open(ckpt_file_path, "r") as f:
    cl_ckpts = json.load(f)

ckpt_path = cl_ckpts["cl"]
config_file = Path("configs/imaging_tabular_model/cl_pretraining_imaging_vita.yaml")

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Modify the config (adjust the field to your actual structure)
config["general"]["wandb_run_id"] = None
config["general"]["resume_training"] = True
config["general"]["resume_ckpt_path"] = ckpt_path
config["module"]["imaging_hparams"]["mask_ratio"] = 0.0
# config["data"]["num_test"] = 30

# Save new config
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_filename = f"save_embeddings_{timestamp}.yaml"
config_path = os.path.join(output_config_dir, config_filename)

with open(config_path, "w") as f:
    yaml.safe_dump(config, f)

# Run main.py with this config
cmd = main_py_command + ["-c", config_path] + ["-n", f"{task_name}_{embedding_type}"] + ["--save_path", f"{save_path}"] + ["--embedding_type", f"{embedding_type}"]
result = subprocess.run(cmd)

print(f"[{save_path} - {embedding_type}] saving embeddings completed")

