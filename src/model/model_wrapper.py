from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from collections import defaultdict

import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, ImageMetrics, DepthMetrics
from ..global_cfg import get_cfg
from ..loss import Loss, LossPyimage
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
import torch.nn.functional as F
from .types import Gaussians


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool
    final_div_factor: float | None
    weight_decay: float | None
    pct_start: float | None
    div_factor: float | None


@dataclass
class TestCfg:
    compute_scores: bool
    save_image: bool
    save_video: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int

@dataclass
class ValCfg:
    num_visualize: int

@dataclass
class PredictCfg:
    extended_visualization: bool
    save_image: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder | None
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    val_cfg: ValCfg
    predict_cfg: PredictCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        val_cfg: ValCfg,
        predict_cfg: PredictCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder | None,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        weights_path: Optional[Path] = None,
        wo_defbp2: bool = False,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.predict_cfg = predict_cfg
        self.step_tracker = step_tracker
        self.wo_defbp2 = wo_defbp2

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.image_metrics = ImageMetrics()
        self.depth_metrics = DepthMetrics()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = defaultdict(list)
            self.test_step_inputs = defaultdict(list)

        if weights_path is not None:
            path_exist = weights_path.exists()
            assert get_cfg().mode != 'train' or path_exist, f"==> Weights path {weights_path} does not exist for training"
            if path_exist:
                print(f"==> Finetuning from {weights_path}")
                load_state_dict = torch.load(weights_path)["state_dict"]
                state_dict = self.state_dict()
                new_state_dict = {}
                for k, v in load_state_dict.items():
                    if k in state_dict and state_dict[k].shape == v.shape:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=False)
            else:
                print(f"==> Weights path {weights_path} does not exist for testing, skipping loading")

        self.automatic_optimization = self.encoder.gaussian_head.num_patchs == 1 if hasattr(self.encoder, "gaussian_head") else True

    def _robust_normalize01(self, x, eps=1e-6):
        x = x.detach().float()
        finite = torch.isfinite(x)
        if not finite.any():
            return torch.zeros_like(x)

        valid = x[finite]
        q_low = torch.quantile(valid, 0.02)
        q_high = torch.quantile(valid, 0.98)
        denom = q_high - q_low
        if not torch.isfinite(denom) or denom.abs() < eps:
            return torch.zeros_like(x)

        normalized = (x - q_low) / denom.clamp_min(eps)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return normalized.clamp(0.0, 1.0)

    def _as_bvhw_mvs_tensor(self, x, batch_size, num_views, name):
        if x is None:
            return None

        if not torch.is_tensor(x):
            print(f"[mvs_debug] Warning: {name} is not a tensor: {type(x)}")
            return None

        if x.ndim == 5 and x.shape[:3] == (batch_size, num_views, 1):
            return x[:, :, 0]

        if x.ndim == 4:
            if x.shape[:2] == (batch_size, num_views):
                return x
            if x.shape[0] == batch_size * num_views and x.shape[1] == 1:
                return rearrange(
                    x,
                    "(v b) 1 h w -> b v h w",
                    v=num_views,
                    b=batch_size,
                )

        print(
            "[mvs_debug] Warning: could not convert "
            f"{name} with shape {tuple(x.shape)} to [B,V,H,W] "
            f"for B={batch_size}, V={num_views}"
        )
        return None

    def _resize_bvhw_to_context(self, x, target_hw, mode="bilinear"):
        batch_size, num_views, _, _ = x.shape
        x_flat = rearrange(x, "b v h w -> (b v) 1 h w")
        if mode == "bilinear":
            x_flat = F.interpolate(
                x_flat,
                size=target_hw,
                mode=mode,
                align_corners=False,
            )
        else:
            x_flat = F.interpolate(x_flat, size=target_hw, mode=mode)
        return rearrange(
            x_flat,
            "(b v) 1 h w -> b v h w",
            b=batch_size,
            v=num_views,
        )

    def _as_bvdhw_mvs_tensor(self, x, batch_size, num_views, name):
        if x is None:
            return None

        if not torch.is_tensor(x):
            print(f"[mvs_debug] Warning: {name} is not a tensor: {type(x)}")
            return None

        if x.ndim == 5 and x.shape[:2] == (batch_size, num_views):
            return x

        if x.ndim == 4 and x.shape[0] == batch_size * num_views:
            return rearrange(
                x,
                "(v b) d h w -> b v d h w",
                v=num_views,
                b=batch_size,
            )

        print(
            "[mvs_debug] Warning: could not convert "
            f"{name} with shape {tuple(x.shape)} to [B,V,D,H,W] "
            f"for B={batch_size}, V={num_views}"
        )
        return None

    def _compute_mvs_uncertainty_maps(self, stage, batch_size, num_views, stage_name):
        raw_correlation = self._as_bvdhw_mvs_tensor(
            stage.get("raw_correlation"),
            batch_size,
            num_views,
            f"{stage_name}.raw_correlation",
        )
        disp_candidates = self._as_bvdhw_mvs_tensor(
            stage.get("disp_candi_curr"),
            batch_size,
            num_views,
            f"{stage_name}.disp_candi_curr",
        )
        if raw_correlation is None or disp_candidates is None:
            return None

        logits = raw_correlation.detach().float()
        pdf = F.softmax(logits, dim=2)
        pdf = torch.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

        num_depth_candidates = pdf.shape[2]
        entropy = -(pdf * (pdf.clamp_min(1e-8).log())).sum(dim=2)
        entropy = entropy / max(np.log(num_depth_candidates), 1e-6)

        top2 = torch.topk(pdf, k=min(2, num_depth_candidates), dim=2).values
        if num_depth_candidates == 1:
            top1_top2_margin = top2[:, :, 0]
        else:
            top1_top2_margin = top2[:, :, 0] - top2[:, :, 1]

        depth_candidates = torch.where(
            disp_candidates > 0,
            1.0 / disp_candidates.detach().float().clamp_min(1e-6),
            torch.zeros_like(disp_candidates, dtype=torch.float32),
        )
        mean_depth = (pdf * depth_candidates).sum(dim=2, keepdim=True)
        depth_variance = (pdf * (depth_candidates - mean_depth).square()).sum(dim=2)

        return {
            "entropy": entropy,
            "top1_top2_margin": top1_top2_margin,
            "depth_variance": depth_variance,
        }

    def _save_mvs_debug_maps(self, batch, encoder_outputs, batch_idx):
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        mvs_outputs = encoder_outputs.get("mvs_outputs", None)
        if mvs_outputs is None:
            return
        if "depth" not in mvs_outputs or "photometric_confidence" not in mvs_outputs:
            print(
                "[mvs_debug] Warning: mvs_outputs is missing depth or "
                "photometric_confidence; skipping MVS debug maps."
            )
            return

        with torch.no_grad():
            context_rgb = batch["context"]["image"].detach()
            batch_size, num_views, _, height, width = context_rgb.shape
            target_hw = (height, width)

            depth = self._as_bvhw_mvs_tensor(
                mvs_outputs.get("depth"),
                batch_size,
                num_views,
                "mvs_outputs.depth",
            )
            confidence = self._as_bvhw_mvs_tensor(
                mvs_outputs.get("photometric_confidence"),
                batch_size,
                num_views,
                "mvs_outputs.photometric_confidence",
            )
            if depth is None or confidence is None:
                return

            disp_raw = self._as_bvhw_mvs_tensor(
                mvs_outputs.get("disp"),
                batch_size,
                num_views,
                "mvs_outputs.disp",
            )
            disp_source = "mvs_outputs.disp"
            if disp_raw is None:
                disp = torch.where(
                    depth > 0,
                    1.0 / depth.clamp_min(1e-6),
                    torch.zeros_like(depth),
                )
                disp_source = "inverse_depth"
            else:
                disp = disp_raw

            depth_up = self._resize_bvhw_to_context(depth, target_hw)
            disp_up = self._resize_bvhw_to_context(disp, target_hw)
            confidence_up = self._resize_bvhw_to_context(confidence, target_hw)

            scene = str(batch["scene"][0])
            name = batch.get("name", None)
            name = "none" if name is None else str(name[0])
            dis = batch.get("dis", None)
            if dis is None:
                dis = "none"
            else:
                dis = dis[0].item() if torch.is_tensor(dis[0]) else dis[0]
                dis = str(dis)

            def safe_path_part(value):
                return str(value).replace("/", "_").replace("\\", "_")

            out_dir = (
                Path(self.logger.save_dir)
                / "test"
                / "mvs_debug"
                / f"{safe_path_part(name)}_{safe_path_part(dis)}"
                / safe_path_part(scene)
                / f"batch_{batch_idx:06d}"
            )

            context_indices = batch["context"]["index"].detach()

            def save_debug_set(root, depth_bvhw, disp_bvhw, confidence_bvhw, save_raw_confidence):
                for view_idx in range(num_views):
                    index = context_indices[0, view_idx]
                    index = int(index.item()) if torch.is_tensor(index) else int(index)

                    depth_norm = self._robust_normalize01(depth_bvhw[0, view_idx])
                    disp_norm = self._robust_normalize01(disp_bvhw[0, view_idx])
                    conf_raw = torch.nan_to_num(
                        confidence_bvhw[0, view_idx].detach().float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    conf_norm = self._robust_normalize01(confidence_bvhw[0, view_idx])
                    low_conf = (1.0 - conf_norm).clamp(0.0, 1.0)

                    depth_color = apply_color_map_to_image(depth_norm, "turbo")
                    disp_color = apply_color_map_to_image(disp_norm, "turbo")
                    conf_norm_color = apply_color_map_to_image(conf_norm, "viridis")
                    low_conf_color = apply_color_map_to_image(low_conf, "magma")

                    save_image(depth_color, root / "mvs_depth" / f"{index:06d}.png")
                    save_image(disp_color, root / "mvs_disparity" / f"{index:06d}.png")
                    save_image(conf_norm_color, root / "mvs_confidence_norm" / f"{index:06d}.png")
                    save_image(low_conf_color, root / "mvs_low_confidence" / f"{index:06d}.png")

                    if save_raw_confidence:
                        conf_raw_color = apply_color_map_to_image(conf_raw, "viridis")
                        low_conf_overlay = (
                            0.6 * context_rgb[0, view_idx].detach().float().clamp(0.0, 1.0)
                            + 0.4 * low_conf_color
                        ).clamp(0.0, 1.0)
                        save_image(context_rgb[0, view_idx], root / "context_rgb" / f"{index:06d}.png")
                        save_image(
                            conf_raw_color,
                            root / "mvs_confidence_raw" / f"{index:06d}.png",
                        )
                        save_image(
                            low_conf_overlay,
                            root / "mvs_low_conf_overlay" / f"{index:06d}.png",
                        )

            def save_uncertainty_set(root, uncertainty_maps):
                entropy_up = self._resize_bvhw_to_context(
                    uncertainty_maps["entropy"],
                    target_hw,
                )
                margin_up = self._resize_bvhw_to_context(
                    uncertainty_maps["top1_top2_margin"],
                    target_hw,
                )
                variance_up = self._resize_bvhw_to_context(
                    uncertainty_maps["depth_variance"],
                    target_hw,
                )

                for view_idx in range(num_views):
                    index = context_indices[0, view_idx]
                    index = int(index.item()) if torch.is_tensor(index) else int(index)

                    entropy = torch.nan_to_num(
                        entropy_up[0, view_idx].detach().float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    margin = torch.nan_to_num(
                        margin_up[0, view_idx].detach().float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    variance_norm = self._robust_normalize01(variance_up[0, view_idx])

                    save_image(
                        apply_color_map_to_image(entropy, "magma"),
                        root / "mvs_entropy" / f"{index:06d}.png",
                    )
                    save_image(
                        apply_color_map_to_image(margin, "viridis"),
                        root / "mvs_top1_top2_margin" / f"{index:06d}.png",
                    )
                    save_image(
                        apply_color_map_to_image(variance_norm, "turbo"),
                        root / "mvs_depth_variance" / f"{index:06d}.png",
                    )

            save_debug_set(out_dir, depth_up, disp_up, confidence_up, True)

            raw_tensors = {
                "depth": depth.detach().cpu(),
                "depth_up": depth_up.detach().cpu(),
                "disp_source": disp_source,
                "photometric_confidence": confidence.detach().cpu(),
                "photometric_confidence_up": confidence_up.detach().cpu(),
                "context_indices": context_indices.detach().cpu(),
                "scene": scene,
                "batch_idx": batch_idx,
            }
            if disp_raw is not None:
                raw_tensors["disp"] = disp_raw.detach().cpu()
                raw_tensors["disp_up"] = disp_up.detach().cpu()
            else:
                raw_tensors["disp_visualization"] = disp.detach().cpu()
                raw_tensors["disp_visualization_up"] = disp_up.detach().cpu()
            out_dir.mkdir(exist_ok=True, parents=True)
            torch.save(raw_tensors, out_dir / "raw_tensors.pt")

            for stage_idx, stage in enumerate(mvs_outputs.get("stages", [])):
                if "depth" not in stage or "photometric_confidence" not in stage:
                    continue

                stage_depth = self._as_bvhw_mvs_tensor(
                    stage.get("depth"),
                    batch_size,
                    num_views,
                    f"mvs_outputs.stages[{stage_idx}].depth",
                )
                stage_confidence = self._as_bvhw_mvs_tensor(
                    stage.get("photometric_confidence"),
                    batch_size,
                    num_views,
                    f"mvs_outputs.stages[{stage_idx}].photometric_confidence",
                )
                if stage_depth is None or stage_confidence is None:
                    continue

                stage_disp = self._as_bvhw_mvs_tensor(
                    stage.get("disp"),
                    batch_size,
                    num_views,
                    f"mvs_outputs.stages[{stage_idx}].disp",
                )
                if stage_disp is None:
                    stage_disp = torch.where(
                        stage_depth > 0,
                        1.0 / stage_depth.clamp_min(1e-6),
                        torch.zeros_like(stage_depth),
                    )

                stage_depth_up = self._resize_bvhw_to_context(stage_depth, target_hw)
                stage_disp_up = self._resize_bvhw_to_context(stage_disp, target_hw)
                stage_confidence_up = self._resize_bvhw_to_context(stage_confidence, target_hw)
                save_debug_set(
                    out_dir / f"stage_{stage_idx}",
                    stage_depth_up,
                    stage_disp_up,
                    stage_confidence_up,
                    False,
                )
                if stage_idx in (1, 2):
                    uncertainty_maps = self._compute_mvs_uncertainty_maps(
                        stage,
                        batch_size,
                        num_views,
                        f"mvs_outputs.stages[{stage_idx}]",
                    )
                    if uncertainty_maps is not None:
                        save_uncertainty_set(out_dir / f"stage_{stage_idx}", uncertainty_maps)

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        deferred_backprop = not self.automatic_optimization
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()

        # Run the model.
        if hasattr(self.encoder, "mvs_forward"):
            encoder_outputs = self.encoder.mvs_forward(batch["context"])
        else:
            encoder_outputs = self.encoder(batch["context"], self.global_step)

        with torch.set_grad_enabled(not deferred_backprop):
            if hasattr(self.encoder, "gh_forward"):
                encoder_outputs["gaussians"] = self.encoder.gh_forward(
                    encoder_outputs["mvs_outputs"],
                    batch["context"],
                    self.global_step,
                )

        if self.decoder is None:
            output = None
        else:
            if deferred_backprop and self.wo_defbp2:
                gaussians = encoder_outputs["gaussians"]
                if isinstance(gaussians["gaussians"], list):
                    for g in gaussians["gaussians"]:
                        g.requires_grad_(True)
                else:
                    gaussians["gaussians"].requires_grad_(True)

            with torch.set_grad_enabled(not deferred_backprop or self.wo_defbp2):
                output = self.decoder.render_pano(
                    encoder_outputs["gaussians"]["gaussians"],
                    batch["target"]["extrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    depth_mode=self.train_cfg.depth_mode,
                    context_extrinsics=batch["context"]["extrinsics"],
                )

            # Compute metrics.
            with torch.set_grad_enabled(False):
                target_gt = batch["target"]["image"]
                psnr_probabilistic = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(output["color"], "b v c h w -> (b v) c h w"),
                )
                self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Deferred Back-Propagation
        if deferred_backprop and not self.wo_defbp2:
            output['color'].requires_grad_(True)

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, encoder_outputs, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss, prog_bar=True)

        # Log some information.
        self.log("info/near", batch["context"]["near"].detach().float().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().float().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.automatic_optimization:
            return total_loss

        self.manual_backward(total_loss, retain_graph=deferred_backprop)
        del total_loss, loss

        if deferred_backprop:
            # Backpropagate from color to gaussian parameters
            color_grad = output['color'].grad
            gaussians = encoder_outputs["gaussians"]

            if not self.wo_defbp2:
                if isinstance(gaussians["gaussians"], list):
                    for g in gaussians["gaussians"]:
                        g.requires_grad_(True)
                else:
                    gaussians["gaussians"].requires_grad_(True)

                for face_idx in range(self.decoder.num_faces):
                    output = self.decoder.render_pano(
                        gaussians["gaussians"],
                        batch["target"]["extrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        face_idx=face_idx,
                        context_extrinsics=batch["context"]["extrinsics"],
                    )
                    self.manual_backward(output['color'], gradient=color_grad)

            # Backpropagate from gaussian parameters to input
            keys = [f.name for f in fields(Gaussians)]
            # encoder_outputs = self.encoder.mvs_forward(batch["context"])
            for patch_idx in range(self.encoder.gaussian_head.num_patchs):
                if isinstance(gaussians["gaussians"], list):
                    patch_grad = torch.cat([
                        getattr(g, k).grad[i == patch_idx].flatten()
                        for g, i in zip(gaussians["gaussians"], gaussians["patch_idx"])
                        for k in keys
                    ])
                else:
                    mask = gaussians["patch_idx"] == patch_idx
                    patch_grad = torch.cat([
                        getattr(gaussians["gaussians"], k).grad[mask].flatten()
                        for k in keys
                    ])
                patch = self.encoder.gaussian_head.patch_forward(
                    encoder_outputs["mvs_outputs"], batch["context"], self.global_step, patch_idx)

                if isinstance(gaussians["gaussians"], list):
                    gs_patch = torch.cat([
                        getattr(g, k).flatten()
                        for g in patch["gaussians"]
                        for k in keys
                    ])
                else:
                    gs_patch = torch.cat([
                        getattr(patch["gaussians"], k).flatten()
                        for k in keys
                    ])
                retain_graph = patch_idx < self.encoder.gaussian_head.num_patchs - 1
                self.manual_backward(gs_patch, gradient=patch_grad, retain_graph=retain_graph)
                del patch_grad, patch, gs_patch
            self.encoder.gaussian_head.clean_padded_cache()

        self.clip_gradients(opt, gradient_clip_val=get_cfg().trainer.gradient_clip_val)
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1

        # Render Gaussians.
        encoder_outputs = self.encoder(batch["context"], self.global_step)
        if self.test_cfg.save_image:
            self._save_mvs_debug_maps(batch, encoder_outputs, batch_idx)
        gaussians = encoder_outputs.pop("gaussians", {}).pop("gaussians", None)

        if self.test_cfg.compute_scores:
            self.test_step_inputs["context_extrinsics"].append(
                batch["context"]["extrinsics"].cpu().numpy().tolist())
            self.test_step_inputs["target_extrinsics"].append(
                batch["target"]["extrinsics"].cpu().numpy().tolist())
            self.test_step_inputs["context_index"].append(
                batch["context"]["index"].cpu().numpy().tolist())
            self.test_step_inputs["target_index"].append(
                batch["target"]["index"].cpu().numpy().tolist())

        name = batch.get("name", None)
        name = None if name is None else name[0]
        dis = batch.get("dis", None)
        dis = None if dis is None else dis[0].item()
        if self.test_cfg.compute_scores and 'mvs_outputs' in encoder_outputs \
                and 'depth' in batch["context"] and name == "m3d" and dis == 0.5:
            depth_gt = rearrange(batch["context"]["depth"], "1 v 1 h w -> 1 v h w")
            depth_est = encoder_outputs["mvs_outputs"]["depth"]
            mask = rearrange(batch["context"]["mask"], "1 v 1 h w -> 1 v h w")
            depth_est = F.interpolate(depth_est, depth_gt.shape[-2:], mode="bilinear", align_corners=False)
            depth_metrics = self.depth_metrics(depth_gt, depth_est, mask)
            for k, m in depth_metrics.items():
                self.test_step_outputs[k].append(m)

        if self.decoder is not None:
            output = self.decoder.render_pano(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                depth_mode=None,
                context_extrinsics=batch["context"]["extrinsics"],
            )

            (scene,) = batch["scene"]
            path = Path(self.logger.save_dir) / 'test'
            images_prob = output['color'][0]
            rgb_gt = batch["target"]["image"][0]

            # Save images.
            if self.test_cfg.save_image:
                indices = batch["target"]["index"][0]
                batch_name = f"{indices[0]:0>6}_{indices[-1]:0>6}"
                for index, color, gt in zip(indices, images_prob, batch["target"]["image"][0]):
                    if len(indices) == 1:
                        save_image(color, path / f"{name}_{dis}" / scene / f"color/{index:0>6}.png")
                    else:
                        save_image(color, path / f"{name}_{dis}" / scene / f"color/{batch_name}/{index:0>6}.png")
                    save_image(gt, path / f"{name}_{dis}" / scene / f"gt/{batch_name}/{index:0>6}.png")
                for index, context in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                    save_image(context, path / f"{name}_{dis}" / scene / f"in/{batch_name}/{index:0>6}.png")

            # save video
            if self.test_cfg.save_video:
                frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
                save_video(
                    [a for a in images_prob],
                    path / "video" / f"{scene}_frame_{frame_str}.mp4",
                )

            # compute scores
            if self.test_cfg.compute_scores:
                rgb = images_prob

                image_metrics = self.image_metrics(rgb_gt, rgb, average=False)
                for k, m in image_metrics.items():
                    k = f"{name}_{dis}_{k}" if name is not None else k
                    self.test_step_outputs[k].append(m.tolist())

    def on_test_end(self) -> None:
        out_dir = Path(self.logger.save_dir) / 'test'
        saved_scores = {}
        if not self.test_cfg.compute_scores:
            return

        with (out_dir / "results.json").open("w") as f:
            json.dump(
                {
                    "metrics": self.test_step_outputs,
                    "inputs": self.test_step_inputs,
                },
                f,
                indent=4
            )

        for metric_name, metric_scores in self.test_step_outputs.items():
            scores = np.array(metric_scores)
            avg_scores = scores.mean()
            saved_scores[metric_name] = avg_scores
            print(metric_name, avg_scores)
            metric_scores.clear()

        try:
            wandb.summary.update(saved_scores)
            metrics_table = wandb.Table(columns=list(saved_scores.keys()), data=[list(saved_scores.values())])
            self.logger.experiment.log({'test/metrics': metrics_table})
        except Exception:
            pass

        with (out_dir / f"scores_all_avg.json").open("w") as f:
            json.dump(saved_scores, f)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        encoder_outputs = self.encoder(batch["context"], self.global_step)

        if batch_idx < self.val_cfg.num_visualize:
            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Draw cameras.
            cameras = hcat(*render_cameras(batch, 256))
            self.logger.log_image(
                "cameras", [prep_image(add_border(cameras))], step=self.global_step
            )

        if "mvs_outputs" in encoder_outputs and "depth" in batch["context"]:
            depth_gt = rearrange(batch["context"]["depth"], "1 v 1 h w -> 1 v h w")
            depth_est = encoder_outputs["mvs_outputs"]["depth"]
            mask = rearrange(batch["context"]["mask"], "1 v 1 h w -> 1 v h w")
            depth_est = F.interpolate(depth_est, depth_gt.shape[-2:], mode="bilinear", align_corners=False)
            depth_metrics = self.depth_metrics(depth_gt, depth_est, mask)
            for k, m in depth_metrics.items():
                self.log(f"val/{k}_val", m)

        gaussians_softmax = encoder_outputs.get("gaussians", {}).get("gaussians", None)
        if gaussians_softmax is not None and self.decoder is not None:
            output_softmax = self.decoder.render_pano(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                context_extrinsics=batch["context"]["extrinsics"],
            )
            rgb_softmax = output_softmax['color'][0]

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            image_metrics = self.image_metrics(rgb_gt, rgb_softmax)
            for k, m in image_metrics.items():
                self.log(f"val/{k}_val", m)

            if batch_idx < self.val_cfg.num_visualize:
                # Construct comparison image.
                comparison = hcat(
                    add_label(vcat(*batch["context"]["image"][0]), "Context"),
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_softmax), "Target (Softmax)"),
                )
                self.logger.log_image(
                    "comparison",
                    [prep_image(add_border(comparison))],
                    step=self.global_step,
                    caption=batch["scene"],
                )

                # Render projections and construct projection image.
                # projections = hcat(*render_projections(
                #                         gaussians_softmax,
                #                         256,
                #                         extra_label="(Softmax)",
                #                     )[0])
                # self.logger.log_image(
                #     "projection",
                #     [prep_image(add_border(projections))],
                #     step=self.global_step,
                #     caption=batch["scene"],
                # )

                # # Run video validation step.
                # self.render_video_interpolation(batch, self.global_step)
                # self.render_video_wobble(batch, self.global_step)
                # if self.train_cfg.extended_visualization:
                #     self.render_video_interpolation_exaggerated(batch, self.global_step)
                self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Render pyramid gaussians.
        stages = encoder_outputs.get("gaussians", {}).get("stages", None)
        if stages is not None and batch_idx < self.val_cfg.num_visualize:
            gaussians = [s['gaussians'] for s in stages]
            output_stages = self.decoder.render_pano(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                context_extrinsics=batch["context"]["extrinsics"] \
                if hasattr(self.encoder, 'gaussian_head') and self.encoder.gaussian_head.deferred_blend
                else None,
            )
            if isinstance(output_stages, dict):
                output_stages = [output_stages]
            cols = []
            for i, output_stage in enumerate(output_stages):
                rgb_stage = output_stage['color'][0]
                cols.append(add_label(vcat(*rgb_stage), f"Stage {i}"))
            pgs = hcat(*cols)
            self.logger.log_image(
                "pyramid",
                [prep_image(add_border(pgs))],
                step=self.global_step,
                caption=batch["scene"],
            )

        # Visualize pyramid image loss.
        loss_pyimage = None
        for loss in self.losses:
            if isinstance(loss, LossPyimage):
                loss_pyimage = loss
                break
        if loss_pyimage is not None and batch_idx < self.val_cfg.num_visualize:
            output_stages = loss_pyimage.render(batch, encoder_outputs)
            if isinstance(output_stages, dict):
                output_stages = [output_stages]
            cols = []
            for i, output_stage in enumerate(output_stages):
                rgb_stage = output_stage['color'][0]
                cols.append(add_label(vcat(*rgb_stage), f"Stage {i}"))
            pgs = hcat(*cols)
            self.logger.log_image(
                "pyimage_loss",
                [prep_image(add_border(pgs))],
                step=self.global_step,
                caption=batch["scene"],
            )

    def predict_step(self, batch, batch_idx):
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=batch_idx
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=batch_idx)

        # Run video validation step.
        self.render_video_interpolation(batch, batch_idx)
        self.render_video_wobble(batch, batch_idx)
        if self.predict_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch, batch_idx)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample, step: int) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60, step=step)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, step: int) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][:, 0],
                (
                    batch["context"]["extrinsics"][:, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][:, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][:, 0],
                (
                    batch["context"]["intrinsics"][:, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][:, 0]
                ),
                t,
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "rgb", step=step)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample, step: int) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][:, 0],
                (
                    batch["context"]["extrinsics"][:, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][:, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][:, 0],
                (
                    batch["context"]["intrinsics"][:, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][:, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=100,
            smooth=False,
            loop_reverse=False,
            step=step
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        step: int = 0,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step)["gaussians"]["gaussians"]

        t = torch.linspace(0, 1, num_frames, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)
        # extrinsics = extrinsics.type_as(gaussians_prob.means)
        # intrinsics = intrinsics.type_as(gaussians_prob.means)

        # Color-map the result.
        def depth_map(result):
            result = result.float()
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        context_extrinsics = batch["context"]["extrinsics"]
        del batch
        output_prob = self.decoder.render_pano(
            gaussians_prob, extrinsics, near, far, None, cpu=True, context_extrinsics=context_extrinsics
        )
        video = output_prob['color'][0]

        # output_prob = self.decoder.render_pano(
        #     gaussians_prob, extrinsics, near, far, "depth", cpu=True, context_extrinsics=context_extrinsics
        # )
        # images_prob = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_prob['color'][0].cpu(), depth_map(output_prob['depth'][0].cpu()))
        # ]
        # video = [
        #     add_border(
        #         hcat(
        #             add_label(image_prob, "Softmax"),
        #             # add_label(image_det, "Deterministic"),
        #         )
        #     )
        #     for image_prob, _ in zip(images_prob, images_prob)
        # ]
        # video = torch.stack(video)

        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=24, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                self.logger.log_video(key, value, step)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.estimated_stepping_batches + 10,
                            pct_start=self.optimizer_cfg.pct_start,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                            div_factor=self.optimizer_cfg.div_factor,
                            final_div_factor=self.optimizer_cfg.final_div_factor,
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
