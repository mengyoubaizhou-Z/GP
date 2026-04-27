from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from src.model.encoder.unifuse.datasets.util import Equirec2Cube

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetMixCfg(DatasetCfgCommon):
    name: Literal["mix"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_chunk_interval: int
    train_times_per_scene: int
    train_scenes: list[str] | None = None
    val_scenes: list[str] | None = None
    test_scenes: list[str] | None = None
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = False
    cache_images: bool = True
    include_nested_sequences: bool = False


class DatasetMix(IterableDataset):
    cfg: DatasetMixCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    data: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetMixCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        assert len(cfg.roots) == 1
        self.root = cfg.roots[0]
        all_sequences = self.discover_sequences(self.root)
        self.validate_scene_splits(all_sequences)
        self.data = self.select_sequences_for_stage(all_sequences)

        if self.cfg.overfit_to_scene is not None:
            self.data = [self.root / self.cfg.overfit_to_scene]

        self.e2c_mono = Equirec2Cube(512, 1024, 256)
        self.times_per_scene = (
            self.cfg.train_times_per_scene
            if self.stage == "train"
            else self.view_sampler.cfg.test_times_per_scene
        )
        self.load_images = True

    def discover_sequences(self, root: Path) -> list[Path]:
        sequences = []
        for camera_pose in root.rglob("camera_pose.json"):
            sequence_dir = camera_pose.parent
            if not self.cfg.include_nested_sequences and sequence_dir.parent != root:
                continue
            if (sequence_dir / "images").is_dir() or (sequence_dir / "image").is_dir():
                sequences.append(sequence_dir)
        sequences.sort()
        return sequences

    def validate_scene_splits(self, sequences: list[Path]) -> None:
        available = {sequence.relative_to(self.root).as_posix() for sequence in sequences}

        split_map = {
            "train": set(self.cfg.train_scenes or []),
            "val": set(self.cfg.val_scenes or []),
            "test": set(self.cfg.test_scenes or []),
        }

        for split_name, scenes in split_map.items():
            missing = sorted(scenes - available)
            if missing:
                raise ValueError(
                    f"{split_name}_scenes contains unknown scenes: {missing}"
                )

        overlaps = [
            ("train", "val", split_map["train"] & split_map["val"]),
            ("train", "test", split_map["train"] & split_map["test"]),
            ("val", "test", split_map["val"] & split_map["test"]),
        ]
        for left, right, shared in overlaps:
            if shared:
                raise ValueError(
                    f"{left}_scenes and {right}_scenes overlap: {sorted(shared)}"
                )

    def select_sequences_for_stage(self, sequences: list[Path]) -> list[Path]:
        if self.cfg.overfit_to_scene is not None:
            return [self.root / self.cfg.overfit_to_scene]

        scene_names = {
            sequence.relative_to(self.root).as_posix(): sequence for sequence in sequences
        }

        if self.stage == "train":
            selected = self.cfg.train_scenes
        elif self.stage == "val":
            selected = self.cfg.val_scenes
        else:
            selected = self.cfg.test_scenes

        if not selected:
            return sequences

        return [scene_names[name] for name in selected]

    def get_image_dir(self, example_path: Path) -> Path:
        for candidate in ("images", "image"):
            image_dir = example_path / candidate
            if image_dir.is_dir():
                return image_dir
        raise FileNotFoundError(f"Could not find image directory under {example_path}")

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def load_extrinsics(self, example_path: Path):
        with (example_path / "camera_pose.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        frames = list(payload.keys())
        extrinsics_orig = torch.tensor(list(payload.values()), dtype=torch.float32)
        return frames, extrinsics_orig

    @cached_property
    def total_frames(self):
        extrinsics = [self.load_extrinsics(example)[1] for example in self.data]
        return sum(len(e) for e in extrinsics)

    def __iter__(self):
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.data = self.shuffle(self.data)

        worker_info = torch.utils.data.get_worker_info()
        if self.stage != "train" and worker_info is not None:
            self.data = [
                example
                for data_idx, example in enumerate(self.data)
                if data_idx % worker_info.num_workers == worker_info.id
            ]

        for example_path in self.data:
            frames, extrinsics_orig = self.load_extrinsics(example_path)
            image_dir = self.get_image_dir(example_path)
            scene = str(example_path.relative_to(self.root))

            if self.cfg.cache_images and self.stage == "train" and self.load_images:
                images = [image_dir / frame for frame in frames]
                images = self.convert_images(images)

            for i in range(self.times_per_scene):
                context_indices, target_indices = self.view_sampler.sample(
                    scene,
                    extrinsics_orig,
                    i=i,
                )
                if context_indices is None:
                    break

                load_target = (target_indices >= 0).all()

                context_extrinsics = extrinsics_orig[context_indices]
                if load_target:
                    target_extrinsics = extrinsics_orig[target_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    context_extrinsics[:, :3, 3] /= scale
                    if load_target:
                        target_extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                intrinsics = torch.eye(3, dtype=torch.float32)
                fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
                intrinsics[0, 0] = fx
                intrinsics[1, 1] = fy
                intrinsics[0, 2] = cx
                intrinsics[1, 2] = cy

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                data = {
                    "context": {
                        "extrinsics": context_extrinsics,
                        "intrinsics": repeat(intrinsics, "h w -> b h w", b=len(context_indices)),
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "scene": scene,
                }

                if load_target:
                    data["target"] = {
                        "extrinsics": target_extrinsics,
                        "intrinsics": repeat(intrinsics, "h w -> b h w", b=len(target_indices)),
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    }

                if self.load_images:
                    if self.cfg.cache_images and self.stage == "train":
                        context_images = images[context_indices]
                        if load_target:
                            target_images = images[target_indices]
                    else:
                        context_images = [image_dir / frames[i] for i in context_indices]
                        context_images = self.convert_images(context_images)
                        if load_target:
                            target_images = [image_dir / frames[i] for i in target_indices]
                            target_images = self.convert_images(target_images)

                    mono_images = F.interpolate(context_images, size=(256, 512), mode="bilinear")
                    mono_images = F.interpolate(mono_images, size=(512, 1024), mode="bilinear")

                    cube_image = []
                    for img in mono_images:
                        img = img.numpy()
                        img = rearrange(img, "c h w -> h w c")
                        img = self.e2c_mono.run(img)
                        cube_image.append(img)
                    cube_image = np.stack(cube_image)
                    cube_image = rearrange(cube_image, "v h w c -> v c h w")

                    data["context"]["image"] = context_images
                    data["context"]["mono_image"] = mono_images
                    data["context"]["cube_image"] = cube_image
                    if load_target:
                        data["target"]["image"] = target_images

                yield data

    def convert_images(
        self,
        images: list[str | Path],
    ):
        torch_images = []
        for image in images:
            with Image.open(image) as pil_image:
                pil_image = pil_image.convert("RGB")
                pil_image = pil_image.resize(self.cfg.image_shape[::-1], Image.LANCZOS)
                torch_images.append(self.to_tensor(pil_image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.data) * self.times_per_scene
