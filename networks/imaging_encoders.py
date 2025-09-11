import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block

from networks.tokenizers import *
from utils.imaging_model_related import Masker, sincos_pos_embed, patchify, unpatchify


class ImagingMaskedEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.enc_embed_dim = kwargs.get("enc_embed_dim")
        self.patch_size = kwargs.get("patch_size")
        self.circular_pe = kwargs.get("circular_pe")
        self.use_enc_pe = kwargs.get("use_enc_pe")
        self.mask_loss = kwargs.get("mask_loss")
        self.shift_size = kwargs.get("shift_size")
        self.patch_embed_cls = globals()[kwargs.get("patch_embed_cls")]
        self.img_shape = kwargs.get("img_shape")
        self.patch_in_channels = kwargs.get("patch_in_channels")
        self.use_both_axes = kwargs.get("use_both_axes")
        self.patch_p_num = np.prod(kwargs.get("patch_size")) * self.patch_in_channels
        self.grad_checkpointing = kwargs.get("grad_checkpointing", False)
        # --------------------------------------------------------------------------
        # MAE encoder
        self.pixel_unshuffle_scale = kwargs.get("pixel_unshuffle_scale")
        self.patch_embed = self.patch_embed_cls(self.img_shape, 
                                                in_channels=self.patch_in_channels, 
                                                patch_size=kwargs.get("patch_size"), 
                                                out_channels=kwargs.get("enc_embed_dim"), )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.out_channels))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, self.patch_embed.out_channels), 
                                      requires_grad=False)
        self.encoder = nn.ModuleList([Block(dim=self.patch_embed.out_channels, 
                                            num_heads=kwargs.get("enc_num_heads"), 
                                            mlp_ratio=kwargs.get("mlp_ratio"), 
                                            qkv_bias=True,)
                                      for i in range(kwargs.get("enc_depth"))])
        self.encoder_norm = nn.LayerNorm(self.patch_embed.out_channels)
        # --------------------------------------------------------------------------
        # MAE Masker
        self.using_masker = False if kwargs.get("mask_ratio") == 0.0 else True
        if self.using_masker:
            self.masker = Masker(mask_type=kwargs.get("mask_type"), 
                                 mask_ratio=kwargs.get("mask_ratio"), 
                                 grid_size=self.patch_embed.grid_size)
        self.initialize_parameters()
    
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding
        enc_pos_embed = sincos_pos_embed(self.enc_embed_dim, self.patch_embed.grid_size, cls_token=True,
                                        use_both_axes=self.use_both_axes, circular_pe=self.circular_pe)
        if not self.use_enc_pe:
            enc_pos_embed[:, 0] = 0
        self.enc_pos_embed.data.copy_(enc_pos_embed.unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        """Forward pass of encoder
        input: [B, S, T, H, W] torch.Tensor
        output:
            latent: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            mask: [B, 1 + length * mask_ratio] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        """
        # Embed patches: (B, S, T, H, W) -> (B, S * T * num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add positional embedding: (B, S * T * num_patches, embed_dim)
        if self.use_enc_pe:
            enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
            x = x + enc_pos_embed[:, 1:, :]
            cls_token = self.cls_token + enc_pos_embed[:, :1, :] # (1, 1, embed_dim)
        else:
            cls_token = self.cls_token
        
        # Mask patches: length -> length * mask_ratio
        if self.using_masker:
            x, mask, ids_restore = self.masker(x)
        else:
            mask, ids_restore = None, None
            
        # Append cls token: (B, 1 + length * mask_ratio, em bed_dim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.encoder:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.encoder:
                x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_with_skip_connection(self, x):
        # Embed patches: (B, S, T, H, W) -> (B, S * num_patches, embed_dim)
        x = self.patch_embed(x)

        # Add positional embedding: (B, S * num_patches, embed_dim)
        if self.use_enc_pe:
            enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
            x = x + enc_pos_embed[:, 1:, :]
            cls_token = self.cls_token + enc_pos_embed[:, :1, :] # (1, 1, embed_dim)
        else:
            cls_token = self.cls_token
        
        # Append cls token: (B, 1 + length * mask_ratio, embed_dim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply MAE encoder
        hidden_latents = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.encoder:
                hidden_latents.append(x[:, 1:, :])
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.encoder:
                hidden_latents.append(x[:, 1:, :])
                x = blk(x)
        encoder_output = self.encoder_norm(x)
        return encoder_output, hidden_latents