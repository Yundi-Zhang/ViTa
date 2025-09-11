import torch
from torch import nn
from typing import List, Sequence, Tuple, Union
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block

from networks.unetr_blocks import UnetOutBlock, UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from utils.imaging_model_related import sincos_pos_embed


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Relu(Layer):
    """A linear layer followed by a ReLU activation function."""
    def __init__(self, in_size, out_size, dropout=False, **kwargs):
        super(Relu, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio, norm_layer=nn.LayerNorm, grad_checkpointing: bool = False):
        super(ViTDecoder, self).__init__()
        self.network = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim)
        self.grad_checkpointing = grad_checkpointing
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply transformer decoder
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.network:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.network:
                x = blk(x)
        x = self.norm(x)
        return x
        

class LinearDecoder(nn.Module):
    def __init__(self, in_size, dim, depth, layer_type:Layer=Relu):
        super(LinearDecoder, self).__init__()
        self.network = nn.ModuleList([layer_type(in_size, dim // 2, dropout=0.1)])
        self.network.extend([
            layer_type(dim // 2 ** i, dim // 2 ** (i + 1), dropout=0.1) for i in range(1, depth)])
        self.fc = nn.Linear(dim // 2 ** depth, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.flatten(start_dim=1)
        for layer in self.network:
            x = layer(x)
        x = self.fc(x)
        return x


class UNETR_decoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        upsample_kernel_sizes: Union[list, Sequence[int]],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name="batch")

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name="batch", spatial_dims=2)

        """

        super().__init__()

        patch_size = (1, *patch_size)
        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size
        self.slice_num = img_size[0]
        self.output_channel = out_channels * self.slice_num # times slice num
        self.upsample_kernel_sizes = upsample_kernel_sizes
        assert len(self.upsample_kernel_sizes) == 3, "Only support UNETR decoder depth equals 3"
            
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=self.upsample_kernel_sizes[1:],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=self.upsample_kernel_sizes[2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[1],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=self.output_channel)

    def proj_feat(self, x, hidden_size, grid_size):
        new_view = (x.size(0), *grid_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(grid_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor, x: torch.Tensor, hidden_states_out: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of UNETR decoder.

        Args:
            x_in (torch.Tensor): images in the shape of (batch, slice, time, height, width)
            x (torch.Tensor): latent features extracted from the encoder
            hidden_states_out (List[torch.Tensor]): the output of each layer of encoder

        Returns:
            torch.Tensor: segmentation probability in the same shape as x_in
        """
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        proj_x2 = self.proj_feat(x2, self.hidden_size//self.grid_size[0], self.grid_size) # (batch, hidden_size, s, 2, 16, 16)
        proj_x2 = proj_x2.view(proj_x2.shape[0], -1, *self.grid_size[1:])
        enc2 = self.encoder2(proj_x2)
        x3 = hidden_states_out[1]
        proj_x3 = self.proj_feat(x3, self.hidden_size//self.grid_size[0], self.grid_size) # (batch, hidden_size, s, 2, 16, 16)
        proj_x3 = proj_x3.view(proj_x3.shape[0], -1, *self.grid_size[1:])
        enc3 = self.encoder3(proj_x3)
        
        proj_x = self.proj_feat(x, self.hidden_size//self.grid_size[0], self.grid_size)
        dec3 = proj_x.view(proj_x.shape[0], -1, *self.grid_size[1:])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        seg_out = self.out(out)
        seg_pred = seg_out.view(seg_out.shape[0], -1, self.slice_num, *seg_out.shape[2:]) # (B, 4, slice, T, H, W)
        return seg_pred


class ImagingMaskedDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.grid_size = kwargs.get("grid_size")
        self.dec_embed_dim = kwargs.get("dec_embed_dim")
        self.use_both_axes = kwargs.get("use_both_axes")
        self.decoder_num_patches = kwargs.get("num_patches")
        self.use_enc_pe = kwargs.get("use_enc_pe")

        self.decoder_embed = nn.Linear(kwargs.get("enc_embed_dim"), kwargs.get("dec_embed_dim"))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.decoder_num_patches, kwargs.get("dec_embed_dim")), requires_grad=False)
        self.decoder = ViTDecoder(dim=kwargs.get("dec_embed_dim"), 
                                  num_heads=kwargs.get("dec_num_heads"),
                                  depth=kwargs.get("dec_depth"),
                                  mlp_ratio=kwargs.get("mlp_ratio"),)
        self.recon_head = nn.Linear(in_features=kwargs.get("dec_embed_dim"),
                                    out_features=kwargs.get("head_out_dim"),)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim), requires_grad=True)
        self.initialize_parameters()
    
    def initialize_parameters(self):
        dec_pos_embed = sincos_pos_embed(self.dec_embed_dim, self.grid_size, cls_token=True,
                                         use_both_axes=self.use_both_axes)
        if not self.use_enc_pe:
            dec_pos_embed[:, 0] = 0
        self.dec_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

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

    def forward(self, x, ids_restore):
        """Forward pass of reconstruction decoder
        input:
            x: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        output:
            pred: [B, length, embed_dim] torch.Tensor
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        dec_pos_embed = self.dec_pos_embed.repeat(x.shape[0], 1, 1)
        
        # Append mask tokens and add positional embedding in schuffled order
        if ids_restore is not None:
            mask_token_n = ids_restore.shape[1] + 1 - x.shape[1]
            mask_tokens = self.mask_token.repeat(x.shape[0], mask_token_n, 1)
            x_shuffle = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_restore = torch.gather(x_shuffle, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_shuffle.shape[-1]))
            x_restore_ = x_restore + dec_pos_embed[:, 1:, :]
        else:
            x_restore_ = x[:, 1:, :] + dec_pos_embed[:, 1:, :]
        cls_tok_ = x[:, :1, :] + dec_pos_embed[:, :1, :]
        
        # Reconstruction decoder
        x = torch.cat([cls_tok_, x_restore_], dim=1) # add class token
        x = self.decoder(x) # apply transformer decoder
        
        # Reconstruction head
        x = self.recon_head(x)
        x = x[:, 1:, :] # remove cls token
        x = torch.sigmoid(x) # scale x to [0, 1] for reconstruction task
        return x