# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py

import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1)
        x_embed = mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SphericalPositionEmbedding(nn.Module):
    """Project ERP latitude/longitude codes to the feature dimension."""

    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim

        self.projector = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

    def forward(self, x):
        b, _, h, w = x.size()

        base_dtype = torch.float32
        y = (torch.arange(h, device=x.device, dtype=base_dtype) + 0.5) / h
        x_coord = (torch.arange(w, device=x.device, dtype=base_dtype) + 0.5) / w

        latitude = (0.5 - y).unsqueeze(1) * math.pi
        longitude = (x_coord * 2.0 - 1.0).unsqueeze(0) * math.pi

        spherical_code = torch.stack(
            [
                longitude.expand(h, w).sin(),
                longitude.expand(h, w).cos(),
                latitude.expand(h, w).sin(),
                latitude.expand(h, w).cos(),
            ],
            dim=0,
        ).unsqueeze(0).expand(b, -1, -1, -1)

        pos = self.projector(spherical_code)
        return pos.to(dtype=x.dtype)
