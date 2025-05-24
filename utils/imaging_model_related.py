from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def sincos_pos_embed(embed_dim, grid_size, cls_token=False, use_both_axes=True, circular_pe=False, shift_patch: Optional[tuple] = (0, 0, 0)):
    """
    Generate multi-scale sine-cosine position embedding for 3D+t images with both long axis and short axis slices.
    
    grid_size: int of the grid (S, T, H/patch_size, W/patch_size) where the first 3 slices should be long axis slices
    embed_dim: output dimension for each position where the first dimension is to distinguish long axis from short axis
    pos_embed: [np.prod(grid_size), embed_dim] or [1+np.prod(grid_size), embed_dim] (w/ or w/o cls_token)
    circular_pe: boolean, indicating whether the positional embedding of temporal dimension setup as circular or not.
    shift_patch: the number of the patches to be slided of shape (t_slide, h_slide, w_slide)
    """
    assert len(shift_patch) == 3
    _shift_patch = (0,) + shift_patch
    if use_both_axes:
        assert len(grid_size) >= 3, "Grid_size should be at least 3D for positional embeding with long axis"
        assert (embed_dim - 1) % (len(grid_size) * 2) == 0, "Each dimension has 2 channels (sin, cos)"
        grid_size_sax = (grid_size[0] - 3, *grid_size[1:])
        grid_size_lax = (3, *grid_size[1:])
        
        if not circular_pe:
            grid_lax = torch.meshgrid(*[torch.arange(s, e+s, dtype=torch.float32) for (s, e) in zip(_shift_patch, grid_size_lax)], indexing='ij')
            grid_sax = torch.meshgrid(*[torch.arange(s, e+s, dtype=torch.float32) for (s, e) in zip(_shift_patch, grid_size_sax)], indexing='ij')
        else:
            grid_lax, grid_sax = [], []
            for i, (s, e_l, e_s) in enumerate(zip(_shift_patch, grid_size_lax, grid_size_sax)):
                if i == 1: # temporal
                    _g_lax = torch.arange(s, e_l+s, dtype=torch.float32)
                    _g_sax = torch.arange(s, e_s+s, dtype=torch.float32)
                    g_lax = torch.sin(_g_lax * 2 * torch.pi / len(_g_lax))
                    g_sax = torch.sin(_g_sax * 2 * torch.pi / len(_g_sax))
                else:
                    g_lax = torch.arange(s, e_l+s, dtype=torch.float32)
                    g_sax = torch.arange(s, e_s+s, dtype=torch.float32)
                grid_lax.append(g_lax)
                grid_sax.append(g_sax)
            grid_lax = torch.meshgrid(grid_lax, indexing='ij')
            grid_sax = torch.meshgrid(grid_sax, indexing='ij')
        _grid_lax = torch.stack(grid_lax, dim=0)
        _grid_sax = torch.stack(grid_sax, dim=0)
        _pe_lax = get_multi_sincos_pos_embed_from_grid(embed_dim - 1, _grid_lax)
        _pe_sax = get_multi_sincos_pos_embed_from_grid(embed_dim - 1, _grid_sax)
        _pe_axes = torch.cat([torch.zeros([_pe_lax.shape[0], 1]), torch.ones([_pe_sax.shape[0], 1])], dim=0)
        _pe = torch.cat([_pe_lax, _pe_sax], dim=0)
        pos_embed = torch.cat([_pe_axes, _pe], dim=1)
    else:
        assert len(grid_size) >= 2, "Grid_size should be at least 2D"
        assert embed_dim % (len(grid_size) * 2) == 0, "Each dimension has 2 channels (sin, cos)"
        if not circular_pe:
            grid = torch.meshgrid(*[torch.arange(s, e-s, dtype=torch.float32) for (s, e) in zip(_shift_patch, grid_size)], indexing='ij')
        else:
            grid = []
            for i, (s, e) in enumerate(zip(_shift_patch, grid_size)):
                if i == 1: # temporal
                    _g = torch.arange(s, e_l+s, dtype=torch.float32)
                    g = torch.sin(_g * 2 * torch.pi / len(_g))
                else:
                    g = torch.arange(s, e+s, dtype=torch.float32)
                grid.append(g)
            grid = torch.meshgrid(grid, indexing='ij')
        grid = torch.stack(grid, dim=0)
        pos_embed = get_multi_sincos_pos_embed_from_grid(embed_dim, grid)
        
    if cls_token:
        pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_multi_sincos_pos_embed_from_grid(embed_dim, grid):
    # use half of dimensions to encode grid
    grid_dim = len(grid.shape) - 1
    emb = [get_1d_sincos_pos_embed_from_grid(embed_dim // grid_dim, grid[i]) for i in range(grid.shape[0])]
    emb = torch.concatenate(emb, dim=1) # [(S*T*H*W, D/4)] -> (S*T*H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

 
def patchify(im: torch.Tensor, patch_size: list[int], **kwargs):
    """Split image into patches of size patch_size.
    
    im: [B, S, T, H, W]
    patch_size: a list of 3
    x: [B, L, np.prod(patch_size)] where L = S * T * H * W / np.prod(patch_size)
    """
    assert len(im.shape) == 5
    B, S, T, H, W = im.shape
    
    if len(patch_size) == 3:
        t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
        x = im.reshape(B, S, t, patch_size[0], h, patch_size[1], w, patch_size[2])
        x = torch.einsum("bstphqwr->bsthwpqr", x)
        x = x.reshape(B, S * t * h * w, np.prod(patch_size))
    elif len(patch_size) == 2:
        h, w = H // patch_size[0], W // patch_size[1]
        x = im.reshape(B, S, T, h, patch_size[0], w, patch_size[1])
        x = torch.einsum("bsthqwr->bshwtqr", x)
        x = x.reshape(B, S * h * w, T * np.prod(patch_size))
    return x


def unpatchify(x: torch.Tensor, im_shape: list[int], patch_size: list[int, int, int]=[5, 16, 16], **kwargs):
    """Combine patches into image.
    
    x: [B, L, np.prod(patch_size) or T * np.prod(patch_size)]
    im_shape: [B, S, T, X, Y]
    im: [B, S, T, X, Y] where X = Y
    """
    assert len(x.shape) == 3
    assert len(im_shape) == 5
    
    B, S, T, H, W = im_shape
    if len(patch_size) == 3:
        t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
        x = x.reshape(B, S, t, h, w, patch_size[0], patch_size[1], patch_size[2])
        x = torch.einsum("bsthwpqr->bstphqwr", x)
        im = x.reshape(im_shape)
    elif len(patch_size) == 2:
        h, w = H // patch_size[0], W // patch_size[1]
        x = x.reshape(B, S, h, w, T, patch_size[0], patch_size[1])
        x = torch.einsum("bshwtqr->bsthqwr", x)
        im = x.reshape(im_shape)
    return im


def patchify_SAX(im: torch.Tensor, patch_size: list[int], S_sax: int, in_channels: int = 6, pixel_unshuffle_scale: int = 1, **kwargs):
    """Split image into patches of size patch_size.
    
    im: [B, S, T, H, W]
    patch_size: a list of 3
    x: [B, L, np.prod(patch_size)] where L = S * T * H * W / np.prod(patch_size)
    """
    assert len(im.shape) == 5
    assert len(patch_size) == 3
    B, S, T, H, W = im.shape
    if S == S_sax:
        x = im[:, None] # (B, 1, S, T, H, W)
        S_new = 1
    elif S > S_sax:
        S_lax = S - S_sax
        S_new = 1 + S_lax
        
        sax = im[:, 3:, ...] # (B, S_lax, T, H, W)
        lax = im[:, :3, ...] # (B, S_lax, T, H, W)
        sax_ = sax[:, None] # (B, 1, S_sax, T, H, W)
        lax_ = lax[:, :, None] # (B, S_lax, 1, T, H, W)
        lax_rep = torch.tile(lax_, dims=(1, 1, S_sax, 1, 1, 1)) # (B, S_lax, S_sax, T, H, W)
        x = torch.cat([lax_rep, sax_], dim=1) # (B, S_lax + 1, S_sax, T, H, W)
    else:
        S_new = im.shape[1]
        x = torch.tile(im[:, :, None], dims=(1, 1, in_channels, 1, 1, 1)) # (B, S_lax, in_channels, T, H, W)
    if pixel_unshuffle_scale != 1:
        x = pixel_unshuffle3d(x.reshape(-1, *x.shape[2:]), pixel_unshuffle_scale) # (B*S_new, S_sax*r**3, T//r, H//r, W//r)
        x = x.reshape(B, -1, *x.shape[1:])
    t = T // patch_size[0] // pixel_unshuffle_scale
    h = H // patch_size[1] // pixel_unshuffle_scale
    w = W // patch_size[2] // pixel_unshuffle_scale
    x = x.reshape(B, S_new, in_channels, t, patch_size[0], h, patch_size[1], w, patch_size[2])
    x = torch.einsum("bsitphqwr->bsthwipqr", x)
    x = x.reshape(B, S_new * t * h * w, in_channels * np.prod(patch_size))

    return x


def unpatchify_SAX(x: torch.Tensor, im_shape: list[int], patch_size: list[int, int, int], 
                   S_sax: int, in_channels: int = 6, pixel_unshuffle_scale: int = 1, **kwargs):
    """Combine patches into image.
    
    x: [B, L, np.prod(patch_size) or T * np.prod(patch_size)]
    im_shape: [B, S, T, X, Y]
    im: [B, S, T, X, Y] where X = Y
    """
    assert len(x.shape) == 3
    assert len(im_shape) == 5

    B, S, T, H, W = im_shape
    t = T // patch_size[0] // pixel_unshuffle_scale
    h = H // patch_size[1] // pixel_unshuffle_scale
    w = W // patch_size[2] // pixel_unshuffle_scale
    if S == S_sax:
        x = x.reshape(B, 1, t, h, w, in_channels, patch_size[0], patch_size[1], patch_size[2])
        x = torch.einsum("bsthwipqr->bsitphqwr", x) # (B, S_new, channel)
        x = x.reshape(B, 1, in_channels, T, H, W)
        im = x[:, 0, ...]
    elif S > S_sax:
        S_lax = S - S_sax
        S_new = 1 + S_lax
        
        x = x.reshape(B, S_new, t, h, w, in_channels, patch_size[0], patch_size[1], patch_size[2])
        x = torch.einsum("bsthwipqr->bsitphqwr", x) # (B, S_new, channel)
        x = x.reshape(B, S_new, in_channels, t*patch_size[0], h*patch_size[1], w*patch_size[2])
        if pixel_unshuffle_scale != 1:
            x = pixel_shuffle3d(x.reshape(-1, *x.shape[2:]), pixel_unshuffle_scale)
            x = x.reshape(B, -1, *x.shape[1:])
        lax = x[:, :3, 0] # (B, S_lax, T, H, W)
        sax = x[:, -1] # (B, S_lax, T, H, W)
        im = torch.cat([lax, sax], dim=1) # (B, S, T, H, W)
    else:
        x = x.reshape(B, im_shape[1], t, h, w, in_channels, patch_size[0], patch_size[1], patch_size[2])
        x = torch.einsum("bsthwipqr->bsitphqwr", x) # (B, S_new, channel)
        x = x.reshape(B, im_shape[1], in_channels, T, H, W)
        im = x[:, :, 0, ...]
    return im

    
class Masker:
    def __init__(self, mask_type, mask_ratio, grid_size, **kwargs):
        self.mask_ratio = mask_ratio
        if mask_type == "random":
            self.masking_strategy = self.random_masking
        else:
            raise NotImplementedError
    
    def __call__(self, x):
        """
        x: [N, L, D], sequence
        x_masked: [N, L * mask_ratio, D], masked sequence
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        mask, ids_restore, ids_keep = self.masking_strategy(input_size=x.shape, device=x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, mask, ids_restore
    
    def call_masking_fctn(self, x, fctn_name, **kwargs):
        fctn = eval(f"self.{fctn_name}")
        mask, ids_restore, ids_keep = fctn(input_size=x.shape, device=x.device, **kwargs)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, mask, ids_restore
        
    def random_masking(self, input_size, device, **kwargs):
        """
        # Reference: https://github.com/facebookresearch/mae.git
        
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        input_size: [N, L, D], sequence
        device: torch.device
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        N, L, D = input_size  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore, ids_keep

    def random_masking_plus_given_index(self, input_size, device, given_index, **kwargs):
        """
        Perform per-sample random masking by per-sample shuffling and also mask out the given index.
        input_size: [N, L, D], sequence
        given_index: [L_g], given index to be masked out
        device: torch.device
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        N, L, D = input_size
        maskout_index_total = np.union1d(given_index, np.random.choice(L, int(L * self.mask_ratio), replace=False))
        len_keep = L - len(maskout_index_total)
        mask = torch.zeros([N, L], device=device)
        mask[:, maskout_index_total] = 1
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return mask, ids_restore, ids_keep


class PixelShuffle3D(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type="linear", scale_factor=0.5):
        super(PixelShuffle3D, self).__init__()
        # Initialize the parameters for the PixelShuffle3D class
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_type = layer_type
        self.patches_ori_tuple = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_ori = np.prod(self.patches_ori_tuple)
        self.scale_factor = scale_factor
        self.num_patches_downsample = int(self.num_patches_ori * (scale_factor ** 3))
        self.shuffle_dim = int(in_dim / (scale_factor ** 3))
        # Linear or MLP projection for the input tokens
        if layer_type == "linear":
            self.proj = nn.Linear(self.shuffle_dim, self.out_dim)
        elif layer_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(self.shuffle_dim, self.out_dim),
                nn.GELU(),
                nn.Linear(self.out_dim, self.out_dim)
            )
            
    def pixel_shuffle_3d(self, x):
        # Assuming the input x has shape (N, W, H, Z, C)
        x = x.contiguous()
        n, w, h, z, c = x.size()
        # Calculate new dimensions based on the scale factor
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        new_z = int(z * self.scale_factor)
        new_c = int(c / (self.scale_factor ** 3))
        # N, W, H, Z, C --> N, W * scale, H * scale, Z * scale, C // (scale ** 3)
        x = x.view(n, new_w, new_h, new_z, new_c)
        # N, W * scale, H * scale, Z * scale, C // (scale ** 3) --> N, H * scale, W * scale, Z * scale, C // (scale ** 3)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x
    
    def forward(self, x):
        x = x.reshape(-1, *self.patches_ori_tuple, self.in_dim)
        x = self.pixel_shuffle_3d(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(-1, self.num_patches_downsample, self.shuffle_dim)
        x = self.proj(x)
        return x
    
    
def pixel_shuffle3d(input, upscale_factor):
    batch_size, channels, time, height, width = input.size()
    channels //= upscale_factor ** 3
    output_time = time * upscale_factor
    output_height = height * upscale_factor
    output_width = width * upscale_factor
    
    input = input.view(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, 
                       time, height, width)
    output = input.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    output = output.view(batch_size, channels, output_time, output_height, output_width)
    
    return output


def pixel_unshuffle3d(input, downscale_factor):
    batch_size, channels, time, height, width = input.size()
    output_channels = channels * (downscale_factor ** 3)
    output_time = time // downscale_factor
    output_height = height // downscale_factor
    output_width = width // downscale_factor
    
    input = input.view(batch_size, channels, output_time, downscale_factor, 
                       output_height, downscale_factor, output_width, downscale_factor)
    output = input.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    output = output.view(batch_size, output_channels, output_time, output_height, output_width)
    
    return output