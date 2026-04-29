from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_mvdepth import LossMVDepth, LossMVDepthCfgWrapper
from .loss_pyimage import LossPyimage, LossPyimageCfgWrapper
from .loss_ws_mse import LossWsMse, LossWsMseCfgWrapper
from ..dataset import DatasetCfg

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossMVDepthCfgWrapper: LossMVDepth,
    LossPyimageCfgWrapper: LossPyimage,
    LossWsMseCfgWrapper: LossWsMse,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper \
    | LossMVDepthCfgWrapper | LossPyimageCfgWrapper | LossWsMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper], dataset_cfg: DatasetCfg) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg, dataset_cfg) for cfg in cfgs]
