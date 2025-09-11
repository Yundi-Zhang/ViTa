import os
import yaml
import subprocess
import tempfile
import argparse
import json

def set_nested(d, key_path, value):
    """Set a value in a nested dict using dot-separated keys."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_eval(base_config,
             updates,
             mode="imaging",
             wandb_group="imaging_recon",
             wandb_name="pretraining_imaging"):
    """
    Run evaluation with a modified copy of the config file.
    """
    # Load original config
    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    # Apply updates (dot-path support)
    for k, v in updates.items():
        print(f"✅ Updating {k} = {v}")
        set_nested(config, k, v)

    # Save to a temporary config file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as tmp:
        yaml.safe_dump(config, tmp)
        tmp_config = tmp.name

    try:
        cmd = [
            "python3", "main.py", "test",
            "-c", tmp_config,
            "-m", mode,
            "-g", wandb_group,
            "-n", wandb_name
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        os.remove(tmp_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True, help="Path to base config YAML")
    parser.add_argument("--updates", type=str, required=True,
                        help=(
                            'JSON string with updates, e.g. '
                            '\'{"general.resume_training": true, '
                            '"general.resume_ckpt_path": "path", '
                            '"module.tabular_hparams.selected_features": ["x", "y"]}\''
                        ))
    parser.add_argument("--mode", default="imaging")
    parser.add_argument("--wandb_group", default="imaging_recon")
    parser.add_argument("--wandb_name", default="pretraining_imaging")

    args = parser.parse_args()
    updates = json.loads(args.updates)  # parse JSON string into dict

    run_eval(
        base_config=args.base_config,
        updates=updates,
        mode=args.mode,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
    )
