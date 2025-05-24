from typing import Dict
from collections import OrderedDict

from termcolor import colored
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block

from utils.imaging_model_related import get_1d_sincos_pos_embed_from_grid


__all__ = ["TabularViT"]


class TabularViT(nn.Module):
    def __init__(self, args, all_feature_names: dict, **kwargs) -> None:
        super().__init__()
        self.grad_checkpointing = args.grad_checkpointing
        self.embedding_dim = args.embedding_dim
        self.all_feature_names = all_feature_names
        self.num_numerical = len(all_feature_names["numerical"])
        self.num_single_c = len(all_feature_names["single_categorical"])
        self.num_multiple_c = len(all_feature_names["multi_categorical"])
        self.num_features = self.num_numerical + self.num_single_c + self.num_multiple_c
        
        # Initialize feature embedding functions
        feauture_embeddings = self.tabular_feature_embeddings(out_dim=self.embedding_dim, all_feature_names=all_feature_names)
        self.numerical_fe, self.single_c_fe, self.multi_c_fe = feauture_embeddings
        assert (len(self.numerical_fe) + len(self.single_c_fe) + len(self.multi_c_fe)) == self.num_features
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_features, self.embedding_dim), requires_grad=False)
        self.network = nn.ModuleList([
            Block(self.embedding_dim, 
                  args.encoder_num_heads, 
                  args.encoder_mlp_ratio, 
                  qkv_bias=True, 
                  norm_layer=nn.LayerNorm)
            for i in range(args.encoder_num_layers)])
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.initialize_parameters()
        
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding
        grid_1d = torch.arange(self.num_features, dtype=torch.float32)
        enc_pos_embed = get_1d_sincos_pos_embed_from_grid(self.embedding_dim + 1, grid_1d)
        enc_pos_embed = torch.concatenate([torch.zeros([1, self.embedding_dim]), enc_pos_embed[:, :-1]], dim=0)  # cls
        self.enc_pos_embed.data.copy_(enc_pos_embed.unsqueeze(0))
        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def tabular_feature_embeddings(self, out_dim, all_feature_names):
        numerical_feature_names = all_feature_names["numerical"]
        single_c_feature_names = all_feature_names["single_categorical"]
        multiple_c_feature_names = all_feature_names["multi_categorical"]
        numerical_feature_embeddings = nn.ModuleList([nn.Linear(1, out_dim) for _ in range(len(numerical_feature_names))])
        single_c_feature_embeddings = nn.ModuleList([nn.Embedding(2, out_dim) for _ in range(len(single_c_feature_names))])
        multiple_c_feature_embeddings = nn.ModuleList()
        for _, v in multiple_c_feature_names.items():
            num_classes = v[0]
            multiple_c_feature_embeddings.extend([nn.Embedding(num_classes, out_dim)])

        return numerical_feature_embeddings, single_c_feature_embeddings, multiple_c_feature_embeddings
    
    def forward_embeddings(self, x):
        numerical_x = x[:, :self.num_numerical]
        single_c_x = x[:, self.num_numerical:(self.num_numerical + self.num_single_c)].to(torch.long)
        multiple_c_x = x[:, (self.num_numerical + self.num_single_c):].to(torch.long)
        out = []
        for i, layer in enumerate(self.numerical_fe):
            numerical_emb = layer(numerical_x[:, i:(i+1)])
            out.append(numerical_emb[:, None])
        for i, layer in enumerate(self.single_c_fe):
            out.append(layer(single_c_x[:, i:(i+1)]))
        for i, layer in enumerate(self.multi_c_fe):
            out.append(layer(multiple_c_x[:, i:(i+1)]))
        out = torch.concat(out, dim=1)
        return out
    
    def forward(self, x: torch.Tensor, num_paddings: int = None) -> torch.Tensor:
        x = self.forward_embeddings(x) # Generate feature embeddings
        enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.concat([cls_token, x], dim=1) # Concatenate class token
        x = x + enc_pos_embed # Add positional encoding
        if num_paddings is not None:
            paddings = torch.zeros((x.shape[0], num_paddings, x.shape[2]), device=x.device)
            x = torch.concat([x, paddings], dim=1)
        # Apply transformer encoder
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.network:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.network:
                x = blk(x)
        x = self.norm(x)
        return x