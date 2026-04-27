from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import numpy as np

from ....geometry.projection import get_world_rays_erp
from ....misc.sh_rotation import rotate_sh
from .gaussians import (
    build_covariance_from_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    use_jacobian_aware_gaussians: bool | None = None
    use_jacobian_aware_scale: bool | None = None
    use_jacobian_aware_basis: bool | None = None
    return_world_space_rotations: bool = True
    jacobian_scale_multiplier: float = 0.1
    jacobian_scale_radial_factor: float = 0.5


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg | dict):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = GaussianAdapterCfg(**cfg)
        legacy_flag = cfg.use_jacobian_aware_gaussians
        if cfg.use_jacobian_aware_scale is None:
            cfg.use_jacobian_aware_scale = True if legacy_flag is None else legacy_flag
        if cfg.use_jacobian_aware_basis is None:
            cfg.use_jacobian_aware_basis = True if legacy_flag is None else legacy_flag
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        sh_base: Float[Tensor, "*#batch 3"] | None = None,
    ) -> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        if self.cfg.use_jacobian_aware_scale:
            scale_multiplier = self.get_jacobian_scale_multiplier(
                coordinates,
                image_shape,
            )
            scales = scales * depths[..., None] * scale_multiplier
        else:
            pixel_size = 1 / torch.tensor((w, h), dtype=scales.dtype, device=device)
            multiplier = self.get_scale_multiplier(pixel_size)
            scales = scales * depths[..., None] * multiplier[..., None]

        rotations_matrix = quaternion_to_matrix(rotations)
        if self.cfg.use_jacobian_aware_basis:
            # Align the canonical Gaussian axes with the local ERP tangent frame so the
            # learned rotation acts as a residual on top of spherical geometry.
            local_rotation = self.get_tangent_frame(coordinates)
            rotations_matrix = local_rotation @ rotations_matrix

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        if sh_base is not None:
            sh_base_full = torch.zeros_like(sh)
            sh_base_full[..., 0] = (sh_base - 0.5) * 2
            sh = sh + sh_base_full
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = build_covariance_from_matrix(scales, rotations_matrix)
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        world_rotations = c2w_rotations @ rotations_matrix

        # Compute Gaussian means.
        origins, directions = get_world_rays_erp(coordinates, extrinsics)
        means = origins + directions * depths[..., None]

        if self.cfg.return_world_space_rotations:
            returned_rotations = matrix_to_quaternion(world_rotations)
        elif self.cfg.use_jacobian_aware_basis:
            returned_rotations = matrix_to_quaternion(rotations_matrix)
        else:
            returned_rotations = rotations.broadcast_to((*scales.shape[:-1], 4))

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            scales=scales,
            rotations=returned_rotations,
        )

    def get_scale_multiplier(
        self,
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * pixel_size.new_tensor([2 * np.pi, np.pi]) * pixel_size
        return xy_multipliers.sum(dim=-1)

    def get_jacobian_scale_multiplier(
        self,
        coordinates: Float[Tensor, "*#batch 2"],
        image_shape: tuple[int, int],
        eps: float = 1e-6,
    ) -> Float[Tensor, "*batch 3"]:
        h, w = image_shape
        phi_theta = coordinates * coordinates.new_tensor([2 * np.pi, np.pi]) - coordinates.new_tensor([np.pi, np.pi / 2])
        lat = phi_theta[..., 1]
        cos_lat = torch.cos(lat).clamp_min(eps)

        lon_scale = cos_lat * (2 * np.pi / w)
        lat_scale = torch.full_like(lon_scale, np.pi / h)
        radial_scale = self.cfg.jacobian_scale_radial_factor * torch.sqrt(lon_scale * lat_scale)

        return self.cfg.jacobian_scale_multiplier * torch.stack(
            (lon_scale, lat_scale, radial_scale),
            dim=-1,
        )

    def get_tangent_frame(
        self,
        coordinates: Float[Tensor, "*#batch 2"],
    ) -> Float[Tensor, "*batch 3 3"]:
        phi_theta = coordinates * coordinates.new_tensor([2 * np.pi, np.pi]) - coordinates.new_tensor([np.pi, np.pi / 2])
        phi = phi_theta[..., 0]
        theta = phi_theta[..., 1]

        east = torch.stack(
            (
                torch.cos(phi),
                torch.zeros_like(phi),
                -torch.sin(phi),
            ),
            dim=-1,
        )
        north = torch.stack(
            (
                -torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
                -torch.sin(theta) * torch.cos(phi),
            ),
            dim=-1,
        )
        ray = torch.stack(
            (
                torch.cos(theta) * torch.sin(phi),
                torch.sin(theta),
                torch.cos(theta) * torch.cos(phi),
            ),
            dim=-1,
        )

        return torch.stack((east, north, ray), dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
