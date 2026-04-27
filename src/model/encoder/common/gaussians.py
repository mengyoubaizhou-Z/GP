import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def matrix_to_quaternion(
    matrices: Float[Tensor, "*batch 3 3"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 4"]:
    """Convert rotation matrices to xyzw quaternions."""
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must have shape (..., 3, 3).")

    m00 = matrices[..., 0, 0]
    m01 = matrices[..., 0, 1]
    m02 = matrices[..., 0, 2]
    m10 = matrices[..., 1, 0]
    m11 = matrices[..., 1, 1]
    m12 = matrices[..., 1, 2]
    m20 = matrices[..., 2, 0]
    m21 = matrices[..., 2, 1]
    m22 = matrices[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )

    quat_candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    quat_candidates = quat_candidates / (2.0 * q_abs[..., None].clamp_min(eps))

    best = q_abs.argmax(dim=-1)
    gather_idx = best[..., None, None].expand(*best.shape, 1, 4)
    quat_wxyz = quat_candidates.gather(-2, gather_idx).squeeze(-2)

    # Convert from wxyz to xyzw to match scipy / existing code.
    return torch.cat(
        (quat_wxyz[..., 1:], quat_wxyz[..., :1]),
        dim=-1,
    )


def build_covariance_from_matrix(
    scale: Float[Tensor, "*#batch 3"],
    rotation: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    rotation = quaternion_to_matrix(rotation_xyzw)
    return build_covariance_from_matrix(scale, rotation)


def _sqrt_positive_part(x: Tensor) -> Tensor:
    result = torch.zeros_like(x)
    positive_mask = x > 0
    result[positive_mask] = torch.sqrt(x[positive_mask])
    return result
