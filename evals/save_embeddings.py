import argparse
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from lightning import seed_everything

from utils.general import get_data_paths
from initializers.initialize_dataloader import initialize_dataloader
from initializers.initialize_model import initialize_model
from initializers.initialize_parameters import initialize_parameters


def parser_command_line():
    "Define the arguments required for the script"
    parser = argparse.ArgumentParser(description="Masked Autoencoder Downstream Tasks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest="pipeline", help="pipeline to run")
    
    # Arguments for testing
    parser_test = subparser.add_parser("test", help="test  the model")
    parser_test.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_test.add_argument("-m", "--module", default="imaging", help="imaging or imaging_tabular.")
    parser_test.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_test.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    parser_test.add_argument("--save_path", type=str, required=True, help="Path to saved embeddings")
    parser_test.add_argument("--embedding_type", type=str, required=True, help="The type of embeddings to save")
    
    return parser.parse_args()


@torch.no_grad()
def save_embeddings():
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("medium")
    
    args = parser_command_line() # Load the arguments from the command line
    paths = get_data_paths() # Get the file path from the .env file
    params = initialize_parameters(args)
    seed_everything(params.general.seed, workers=True) # Sets seeds for numpy, torch and python.random.

    data_module = initialize_dataloader(args, params, paths)
    data_loader=data_module.test_dataloader()
    
    model = initialize_model(args, params, paths, data_module)
    if torch.cuda.is_available(): model = model.to("cuda")

    resume_ckpt_path = Path(paths.log_folder) / params.general.resume_ckpt_path
    checkpoint = torch.load(resume_ckpt_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    if args.embedding_type == "cls":
        generate_class_latents(model, data_loader, args.save_path)
    elif args.embedding_type == "temporal":
        generate_temp_latents(model, data_loader, args.save_path)
    elif args.embedding_type == "all":
        generate_all_latents(model, data_loader, args.save_path)
    else:
        raise ValueError


def generate_class_latents(model, data_loader, save_path):
    cls_tokens, subj_ids = [], []
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating latent cls tokens..."):
        if i == 50: break
        imgs, tabs, y, subject_ids = batch
        if torch.cuda.is_available():
            imgs = imgs.to("cuda")
        _, img_embeddings, all_img_embeddings = model.forward_imaging(imgs) 
        cls_tokens.append(img_embeddings)
        subj_ids.append(subject_ids)
    cls_tokens = torch.concat(cls_tokens, dim=0).detach().cpu()
    subj_ids = torch.concat(subj_ids, dim=0).detach().cpu()
    np.savez(save_path, cls_tokens=cls_tokens, subj_ids=subj_ids, )

    print(f"Saved the class embeddings to {save_path}")


def generate_temp_latents(model, data_loader, save_path):
    temp_embeddings, subj_ids = [], []
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating latent temporal tokens..."):
        if i == 50: break
        imgs, tabs, y, subject_ids = batch
        if torch.cuda.is_available():
            imgs = imgs.to("cuda")
        _, _, all_img_embeddings = model.forward_imaging(imgs) 
        patch_t = model.encoder_imaging.patch_size[0]
        B, S, T, H, W = imgs.shape
        dim = all_img_embeddings.shape[-1]
        t = T // patch_t
        t_embeddings = all_img_embeddings[:, 1:, :].reshape(B, S, t, -1, dim)
        t_embeddings = t_embeddings.moveaxis(2, 1)
        t_embeddings = t_embeddings.reshape(B, t, -1, dim).mean(dim=2)
        temp_embeddings.append(t_embeddings)
        subj_ids.append(subject_ids)
    temp_embeddings = torch.concat(temp_embeddings, dim=0).detach().cpu()
    subj_ids = torch.concat(subj_ids, dim=0).detach().cpu()
    np.savez(save_path, temp_embeddings=temp_embeddings, subj_ids=subj_ids, )

    print(f"Saved the temporal embeddings to {save_path}")


def generate_all_latents(model, data_loader, save_path):
    all_tokens, subj_ids = [], []
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating all latent tokens..."):
        if i == 50: break
        imgs, tabs, y, subject_ids = batch
        if torch.cuda.is_available():
            imgs = imgs.to("cuda")
        _, img_embeddings, all_img_embeddings = model.forward_imaging(imgs) 
        all_tokens.append(all_img_embeddings)
        subj_ids.append(subject_ids)
    all_tokens = torch.concat(all_tokens, dim=0).detach().cpu()
    subj_ids = torch.concat(subj_ids, dim=0).detach().cpu()
    np.savez(save_path, all_tokens=all_tokens, subj_ids=subj_ids, )

    print(f"Saved the temporal embeddings to {save_path}")


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    save_embeddings()
