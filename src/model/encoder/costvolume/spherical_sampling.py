import math

import torch
from torch import Tensor


def sample_erp_features_spherically(
    feature: Tensor,
    directions: Tensor,
    output_shape: tuple[int, int],
    eps: float = 1e-6,
) -> Tensor:
    """Sample ERP features with seam-aware wrap and spherical interpolation."""

    b, c, h, w = feature.shape
    _, _, d, n = directions.shape

    target_dirs = directions.permute(0, 2, 3, 1).contiguous()
    phi = torch.atan2(target_dirs[..., 0], target_dirs[..., 2])
    theta = torch.asin(target_dirs[..., 1].clamp(-1 + eps, 1 - eps))

    if w > 1:
        x = (phi + math.pi) / (2 * math.pi) * (w - 1)
    else:
        x = torch.zeros_like(phi)
    if h > 1:
        y = (theta + math.pi / 2) / math.pi * (h - 1)
    else:
        y = torch.zeros_like(theta)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x0 = torch.remainder(x0, max(w, 1))
    y0 = y0.clamp(0, max(h - 1, 0))
    x1 = torch.remainder(x0 + 1, max(w, 1))
    y1 = (y0 + 1).clamp(max=max(h - 1, 0))

    f00 = _gather_feature_samples(feature, x0, y0)
    f01 = _gather_feature_samples(feature, x1, y0)
    f10 = _gather_feature_samples(feature, x0, y1)
    f11 = _gather_feature_samples(feature, x1, y1)

    d00 = _erp_index_to_direction(x0, y0, h, w)
    d01 = _erp_index_to_direction(x1, y0, h, w)
    d10 = _erp_index_to_direction(x0, y1, h, w)
    d11 = _erp_index_to_direction(x1, y1, h, w)

    w00 = _inverse_chord_weights(target_dirs, d00, eps)
    w01 = _inverse_chord_weights(target_dirs, d01, eps)
    w10 = _inverse_chord_weights(target_dirs, d10, eps)
    w11 = _inverse_chord_weights(target_dirs, d11, eps)

    weights = torch.stack((w00, w01, w10, w11), dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True)

    sampled = (
        f00 * weights[:, 0:1]
        + f01 * weights[:, 1:2]
        + f10 * weights[:, 2:3]
        + f11 * weights[:, 3:4]
    )
    out_h, out_w = output_shape
    return sampled.reshape(b, c, d, out_h, out_w)


def _gather_feature_samples(feature: Tensor, x_idx: Tensor, y_idx: Tensor) -> Tensor:
    b, c, _, w = feature.shape
    _, d, n = x_idx.shape
    flat = feature.reshape(b, c, -1)
    linear_idx = (y_idx * w + x_idx).reshape(b, 1, d * n).expand(-1, c, -1)
    gathered = flat.gather(2, linear_idx)
    return gathered.reshape(b, c, d, n)


def _erp_index_to_direction(x_idx: Tensor, y_idx: Tensor, h: int, w: int) -> Tensor:
    if w > 1:
        u = x_idx.to(torch.float32) / (w - 1)
    else:
        u = torch.zeros_like(x_idx, dtype=torch.float32)
    if h > 1:
        v = y_idx.to(torch.float32) / (h - 1)
    else:
        v = torch.zeros_like(y_idx, dtype=torch.float32)

    phi = u * (2 * math.pi) - math.pi
    theta = v * math.pi - math.pi / 2
    cos_theta = torch.cos(theta)
    return torch.stack(
        [
            cos_theta * torch.sin(phi),
            torch.sin(theta),
            cos_theta * torch.cos(phi),
        ],
        dim=-1,
    )


def _inverse_chord_weights(target_dirs: Tensor, source_dirs: Tensor, eps: float) -> Tensor:
    dot = (target_dirs * source_dirs.to(target_dirs.dtype)).sum(dim=-1).clamp(-1.0, 1.0)
    chord_sq = (2.0 - 2.0 * dot).clamp_min(eps)
    return chord_sq.reciprocal()
