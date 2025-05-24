from typing import Optional
import numpy as np
from torch import nn
import torch

from utils.imaging_model_related import pixel_unshuffle3d

__all__ = ["PatchEmbed", "PatchEmbed_Spatial", "PatchEmbed_SAX"]


class PatchEmbed(nn.Module):
    def __init__(self, im_shape: list[int], 
                 in_channels: int = 1, 
                 patch_size: list[int] = [1, 16, 16], 
                 out_channels: int = 256, 
                 flatten: bool = True, 
                 bias: bool = True,
                 norm_layer: Optional[nn.Module] = None, **kwargs):
        super().__init__()
        
        assert len(patch_size) == 3, "Patch size should be 3D"
        assert in_channels == 1, "Only supporting input channel size as 1"
        self.im_shape = im_shape
        if len(im_shape) == 3:
            self.im_shape = im_shape.unsqueeze(1) # (S, H, W) -> (S, 1, H, W)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten = flatten
        
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.grid_size = (im_shape[0], 
                          im_shape[1] // patch_size[0], 
                          im_shape[2] // patch_size[1], 
                          im_shape[3] // patch_size[2]) # (S, t, h, w)
        self.num_patches = np.prod(self.grid_size)
    
    def forward(self, x):
        """ 
        input: (B, S, T, H, W)
        output: (B * S, out_channels, t, h, w) or (B, num_patches, out_channels) if flatten is True, 
                where num_patches = S * T * H * W / np.prod(patch_size)
        """
        x_ = x.reshape(-1, *self.im_shape[-3:]) # (B*S, T, H, W)
        x_ = x_.unsqueeze(1) # (B*S, 1, T, H, W)
        x_ = self.proj(x_) # (B*S, out_channels, t, h, w)
        
        if self.flatten:
            x__ = x_.flatten(2) # (B*S, out_channels, t*h*w)
            x__ = x__.moveaxis(1, -1) # (B*S, t*h*w, out_channels)
            x_ = x__.reshape(x.shape[0], -1, x__.shape[-1]) # (B, S*t*h*w, out_channels)
        else:
            x_ = x_.moveaxis(1, -1)
            
        x = self.norm(x_)
        return x
    

class PatchEmbed_Spatial(nn.Module):
    def __init__(self, im_shape: list[int], 
                 in_channels: int, 
                 patch_size: list[int] = [16, 16], 
                 out_channels: int = 256, 
                 flatten: bool = True, 
                 bias: bool = True,
                 norm_layer: Optional[nn.Module] = None, **kwargs):
        super().__init__()
        
        assert len(patch_size) == 2, "Patch size should be 2D"
        self.im_shape = im_shape
        if len(im_shape) == 3:
            self.im_shape = im_shape.unsqueeze(1) # (S, H, W) -> (S, 1, H, W)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten = flatten
        
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.grid_size = (im_shape[0], im_shape[2] // patch_size[0], im_shape[3] // patch_size[1]) # (S, h, w)
        self.num_patches = np.prod(self.grid_size)
    
    def forward(self, x):
        """ 
        input: (B, S, T, H, W)
        output: (B * S, out_channels, t, h, w) or (B, num_patches, out_channels) if flatten is True, 
                where num_patches = S * T * H * W / np.prod(patch_size)
        """
        x_ = x.reshape(-1, *self.im_shape[-3:]) # (B*S, T, H, W)
        x_ = self.proj(x_) # (B*S, out_channels, h, w)
        
        if self.flatten:
            x__ = x_.flatten(2) # (B*S, out_channels, h*w)
            x__ = x__.moveaxis(1, -1) # (B*S, h*w, out_channels)
            x_ = x__.reshape(x.shape[0], -1, x__.shape[-1]) # (B, S*h*w, out_channels)
        else:
            x_ = x_.moveaxis(1, -1)
            
        x = self.norm(x_)
        return x


class PatchEmbed_SAX(nn.Module):
    def __init__(self, 
                 im_shape: list[int], 
                 sax_slice_num: int = 1, 
                 in_channels: int = 1, 
                 patch_size: list[int] = [1, 16, 16], 
                 out_channels: int = 256, 
                 flatten: bool = True, 
                 bias: bool = True,
                 pixel_unshuffle_scale: int = 1, 
                 norm_layer: Optional[nn.Module] = None, 
                 **kwargs):
        super().__init__()
        
        assert len(patch_size) == 3, "Patch size should be 3D"

        self.im_shape = im_shape
        self.S_sax = sax_slice_num
        if len(im_shape) == 3:
            self.im_shape = im_shape.unsqueeze(1) # (S, H, W) -> (S, 1, H, W)
            
        if self.im_shape[0] == sax_slice_num:
            assert in_channels == sax_slice_num
            grid_size_slice = 1
        elif self.im_shape[0] == 3:
            grid_size_slice = 3
        elif self.im_shape[0] > sax_slice_num:
            grid_size_slice = 4
        else:
            raise NotImplementedError
        
        self.out_channels = out_channels
        self.flatten = flatten
        self.pixel_unshuffle_scale = pixel_unshuffle_scale
        self.in_channels = in_channels * (pixel_unshuffle_scale ** 3)
        self.proj = nn.Conv3d(self.in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.grid_size = (grid_size_slice,
                          im_shape[1] // patch_size[0] // pixel_unshuffle_scale, 
                          im_shape[2] // patch_size[1] // pixel_unshuffle_scale, 
                          im_shape[3] // patch_size[2] // pixel_unshuffle_scale) # (s, t, h, w)
        self.num_patches = np.prod(self.grid_size)
    
    def forward(self, x):
        """ 
        input: (B, S, T, H, W)
        output: (B * S, out_channels, t, h, w) or (B, num_patches, out_channels) if flatten is True, 
                where num_patches = S * T * H * W / np.prod(patch_size)
        """
        B, S, T, H, W = x.shape
        if x.shape[1] > self.S_sax: # sax together with lax
            sax = x[:, 3:, ...] # (B, S_sax, T, H, W)
            lax = x[:, :3, ...] # (B, 3, T, H, W)
            sax_ = sax[:, None] # (B, 1, S_sax, T, H, W)
            lax_ = lax[:, :, None] # (B, S_lax, 1, T, H, W)
            lax_rep = torch.tile(lax_, dims=(1, 1, self.S_sax, 1, 1, 1)) # (B, S_lax, S_sax, T, H, W)
            repeated_x = torch.cat([lax_rep, sax_], dim=1) # (B, S_lax + 1, S_sax, T, H, W)
            processed_x = repeated_x.reshape(-1, self.S_sax, T, H, W)
        elif x.shape[1] == self.S_sax:
            processed_x = x # (B, S_sax, T, H, W)
        else:
            repeated_x = torch.tile(x[:, :, None], dims=(1, 1, self.S_sax, 1, 1, 1)) # (B, S_lax, S_sax, T, H, W)
            processed_x = repeated_x.reshape(-1, self.S_sax, T, H, W)
        if self.pixel_unshuffle_scale != 1:
            processed_x = pixel_unshuffle3d(processed_x, self.pixel_unshuffle_scale) # (B*S_new, S_sax*r**3, T//r, H//r, W//r)
        x_ = self.proj(processed_x) # (B*(S_lax+1), out_channels, t, h, w)
        
        if self.flatten:
            x__ = x_.flatten(2) # (B*S, out_channels, t*h*w)
            x__ = x__.moveaxis(1, -1) # (B*S, t*h*w, out_channels)
            x_ = x__.reshape(x.shape[0], -1, x__.shape[-1]) # (B, S*t*h*w, out_channels)
        else:
            x_ = x_.moveaxis(1, -1)
            
        x = self.norm(x_)
        return x
