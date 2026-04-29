import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from .loss import Loss


@dataclass
class LossWsMseCfg:
    weight: float
    eps: float = 1e-8
    min_spherical_weight: float = 1e-6


@dataclass
class LossWsMseCfgWrapper:
    ws_mse: LossWsMseCfg


class LossWsMse(Loss[LossWsMseCfg, LossWsMseCfgWrapper]):
    def forward(
        self,
        prediction: dict | None,
        batch: BatchedExample,
        encoder_outputs: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pred = prediction["color"]
        target = batch["target"]["image"]
        batch_size, num_views, channels, height, width = pred.shape

        err2 = (pred - target) ** 2
        theta = (
            torch.arange(height, device=pred.device, dtype=pred.dtype) + 0.5
        ) * math.pi / height
        weight_h = torch.sin(theta).clamp_min(self.cfg.min_spherical_weight)
        weight = weight_h.view(1, 1, 1, height, 1)

        numerator = (err2 * weight).sum()
        denominator = weight.sum() * batch_size * num_views * channels * width
        loss = numerator / denominator.clamp_min(self.cfg.eps)

        return self.cfg.weight * loss
