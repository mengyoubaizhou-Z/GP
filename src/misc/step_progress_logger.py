from __future__ import annotations

from typing import Any

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


class StepProgressLogger(Callback):
    def __init__(self, every_n_steps: int = 1) -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking or trainer.global_step == 0:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return

        metrics = trainer.callback_metrics
        epoch = int(trainer.current_epoch)
        step = int(trainer.global_step)

        parts = [
            f"[train] epoch={epoch}",
            f"step={step}",
            f"batch_idx={batch_idx}",
        ]

        for key in (
            "loss/total",
            "loss/mse",
            "loss/lpips",
            "loss/pyimage",
            "train/psnr_probabilistic",
            "lr-Adam",
        ):
            if key not in metrics:
                continue
            value = _to_python_scalar(metrics[key])
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")

        print(" | ".join(parts), flush=True)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            print("[val] sanity_check_complete", flush=True)
            return

        metrics = trainer.callback_metrics
        parts = [f"[val] epoch={int(trainer.current_epoch)}", f"step={int(trainer.global_step)}"]
        for key in (
            "val/psnr_val",
            "val/lpips_val",
            "val/ssim_val",
            "val/ws_psnr_val",
        ):
            if key not in metrics:
                continue
            value = _to_python_scalar(metrics[key])
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")
        print(" | ".join(parts), flush=True)
