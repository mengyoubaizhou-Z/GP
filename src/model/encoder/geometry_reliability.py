from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ...geometry.projection import sample_image_grid


@dataclass
class GeometryReliabilityCfg:
    enable: bool = False
    compute_in_train: bool = True
    compute_in_eval: bool = True
    sigma_depth: float = 0.03
    depth_error_type: str = "relative"
    occlusion_margin: float = 0.03
    min_depth: float = 1.0e-3
    far_depth_margin: float = 0.98
    invalid_reliability: float = 0.5
    detach: bool = True
    aggregate: str = "min"
    return_debug: bool = True


def as_bv1hw(
    x: Tensor | None,
    batch_size: int,
    num_views: int,
    name: str,
) -> Tensor | None:
    if x is None:
        return None

    if x.ndim == 5 and x.shape[:3] == (batch_size, num_views, 1):
        return x

    if x.ndim == 4:
        if x.shape[:2] == (batch_size, num_views):
            return x.unsqueeze(2)
        if x.shape[0] == batch_size * num_views and x.shape[1] == 1:
            return rearrange(x, "(v b) 1 h w -> b v 1 h w", v=num_views, b=batch_size)

    raise ValueError(
        f"Could not convert {name} with shape {tuple(x.shape)} to [B,V,1,H,W] "
        f"for B={batch_size}, V={num_views}."
    )


def _world_to_erp_grid(points_cam: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    z_radial = points_cam.norm(dim=-1).clamp_min(eps)
    direction = points_cam / z_radial.unsqueeze(-1)

    lon = torch.atan2(direction[..., 0], direction[..., 2])
    lat = torch.asin(direction[..., 1].clamp(-1.0, 1.0))
    coord_x = lon / (2 * np.pi) + 0.5
    coord_y = lat / np.pi + 0.5

    coord_x = coord_x.remainder(1.0)
    x_grid = coord_x * 2.0 - 1.0
    y_grid = coord_y * 2.0 - 1.0
    return torch.stack([x_grid, y_grid], dim=-1), coord_y


def _sample_erp_depth(depth: Tensor, grid: Tensor) -> Tensor:
    _, _, _, width = depth.shape
    if width > 1:
        coord_x = ((grid[..., 0] + 1.0) * 0.5).remainder(1.0)
        padded_depth = F.pad(depth, (1, 1, 0, 0), mode="circular")
        padded_x = 2.0 * (coord_x * width + 1.0) / (width + 2) - 1.0
        grid = torch.stack([padded_x, grid[..., 1]], dim=-1)
        depth = padded_depth

    return F.grid_sample(
        depth,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def compute_geometry_reliability(
    depth: Tensor,
    photometric_confidence: Tensor | None,
    extrinsics: Tensor,
    near: Tensor,
    far: Tensor,
    cfg: GeometryReliabilityCfg,
) -> dict:
    if cfg.depth_error_type not in ("relative", "inverse"):
        raise ValueError(f"Unsupported depth_error_type: {cfg.depth_error_type}")
    if cfg.aggregate not in ("min", "mean"):
        raise ValueError(f"Unsupported aggregate: {cfg.aggregate}")

    if cfg.detach:
        depth = depth.detach()
        if photometric_confidence is not None:
            photometric_confidence = photometric_confidence.detach()
        extrinsics = extrinsics.detach()
        far = far.detach()

    batch_size, num_views, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype
    compute_dtype = torch.float32

    depth = depth.float()
    extrinsics = extrinsics.float()
    far = far.float()
    world_to_cam = extrinsics.inverse()
    confidence = as_bv1hw(
        photometric_confidence,
        batch_size,
        num_views,
        "photometric_confidence",
    )
    if confidence is None:
        confidence = depth.new_zeros((batch_size, num_views, 1, height, width))
    confidence = confidence.float()

    xy, _ = sample_image_grid((height, width), device=device)
    lon = xy[..., 0] * (2 * np.pi) - np.pi
    lat = xy[..., 1] * np.pi - np.pi / 2
    camera_dirs = torch.stack(
        [
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
            torch.cos(lat) * torch.cos(lon),
        ],
        dim=-1,
    ).to(dtype=compute_dtype)

    errors_per_source = []
    valid_per_source = []
    eps = max(cfg.min_depth, 1.0e-6)

    for src_idx in range(num_views):
        src_depth = depth[:, src_idx]
        src_valid = (
            torch.isfinite(src_depth)
            & (src_depth > cfg.min_depth)
            & (src_depth < far[:, src_idx, None, None] * cfg.far_depth_margin)
        )

        src_rotation = extrinsics[:, src_idx, :3, :3]
        src_origin = extrinsics[:, src_idx, :3, 3]
        world_dirs = torch.einsum("bij,hwj->bhwi", src_rotation, camera_dirs)
        world_points = src_origin[:, None, None, :] + world_dirs * src_depth[..., None]

        target_errors = []
        target_valids = []
        for tgt_idx in range(num_views):
            if tgt_idx == src_idx:
                continue

            target_w2c = world_to_cam[:, tgt_idx]
            points_cam = (
                torch.einsum("bij,bhwj->bhwi", target_w2c[:, :3, :3], world_points)
                + target_w2c[:, None, None, :3, 3]
            )
            projected_depth = points_cam.norm(dim=-1)
            grid, coord_y = _world_to_erp_grid(points_cam, eps)

            target_depth = depth[:, tgt_idx : tgt_idx + 1]
            sampled_depth = _sample_erp_depth(target_depth, grid)
            sampled_depth = sampled_depth[:, 0]

            if cfg.depth_error_type == "relative":
                error = (sampled_depth - projected_depth).abs() / projected_depth.clamp_min(eps)
            else:
                inv_sampled = 1.0 / sampled_depth.clamp_min(eps)
                inv_projected = 1.0 / projected_depth.clamp_min(eps)
                error = (inv_sampled - inv_projected).abs() / inv_projected.clamp_min(eps)

            target_valid = (
                src_valid
                & torch.isfinite(sampled_depth)
                & torch.isfinite(projected_depth)
                & (sampled_depth > cfg.min_depth)
                & (projected_depth > cfg.min_depth)
                & (sampled_depth < far[:, tgt_idx, None, None] * cfg.far_depth_margin)
                & (coord_y >= 0.0)
                & (coord_y <= 1.0)
            )
            occluded = sampled_depth < projected_depth * (1.0 - cfg.occlusion_margin)
            target_valid = target_valid & (~occluded)

            target_errors.append(error)
            target_valids.append(target_valid)

        if target_errors:
            pair_errors = torch.stack(target_errors, dim=1)
            pair_valids = torch.stack(target_valids, dim=1)
            valid_count = pair_valids.sum(dim=1)

            if cfg.aggregate == "min":
                inf = torch.full_like(pair_errors, torch.inf)
                aggregated_error = torch.where(pair_valids, pair_errors, inf).min(dim=1).values
            else:
                summed_error = torch.where(pair_valids, pair_errors, torch.zeros_like(pair_errors)).sum(dim=1)
                aggregated_error = summed_error / valid_count.clamp_min(1)

            aggregated_valid = valid_count > 0
        else:
            aggregated_error = torch.full_like(src_depth, torch.inf)
            aggregated_valid = torch.zeros_like(src_depth, dtype=torch.bool)

        errors_per_source.append(aggregated_error)
        valid_per_source.append(aggregated_valid)

    geometry_error = torch.stack(errors_per_source, dim=1).unsqueeze(2)
    geometry_valid = torch.stack(valid_per_source, dim=1).unsqueeze(2).to(depth.dtype)

    reliability = torch.exp(-geometry_error / max(cfg.sigma_depth, 1.0e-8)).clamp(0.0, 1.0)
    invalid_value = torch.full_like(reliability, cfg.invalid_reliability)
    geometry_reliability = torch.where(geometry_valid.bool(), reliability, invalid_value)
    geometry_error = torch.where(
        geometry_valid.bool(),
        geometry_error,
        torch.zeros_like(geometry_error),
    )
    geometry_conflict = confidence * (1.0 - geometry_reliability)

    outputs = {
        "geometry_reliability": geometry_reliability.to(dtype=dtype),
        "geometry_valid": geometry_valid.to(dtype=dtype),
        "geometry_error": geometry_error.to(dtype=dtype),
        "geometry_conflict": geometry_conflict.to(dtype=dtype),
    }
    if cfg.detach:
        outputs = {key: value.detach() for key, value in outputs.items()}

    return outputs
