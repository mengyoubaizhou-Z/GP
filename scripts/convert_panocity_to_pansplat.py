#!/usr/bin/env python3
"""Convert PanoCity panoramic poses into the GP/PanSplat mix dataset format.

Output per block:
    <output-root>/<prefix><city>_<block>/
        images/                 # symlinks or copies to pano_images/*.png
        camera_pose.json         # {image_name: 4x4 c2w matrix}

The converter is intentionally tolerant to small variations in the pose-json layout:
- list of records with transformation_matrix / transform_matrix
- dict keyed by image name
- dict with a frames / poses list
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
POSE_KEYS = ("transformation_matrix", "transform_matrix", "matrix", "pose", "c2w")
NAME_KEYS = (
    "image_name",
    "file_name",
    "filename",
    "image",
    "image_path",
    "file_path",
    "rgb",
    "rgb_path",
    "pano",
    "pano_path",
)


def as_4x4_matrix(value: Any, *, source: str) -> list[list[float]]:
    """Normalize a nested/list-flat matrix into a 4x4 Python list."""
    if isinstance(value, dict):
        for key in POSE_KEYS:
            if key in value:
                value = value[key]
                break

    if not isinstance(value, list):
        raise ValueError(f"{source}: pose is not a list or dict")

    # Flat 16 values.
    if len(value) == 16 and all(isinstance(x, (int, float)) for x in value):
        value = [value[i : i + 4] for i in range(0, 16, 4)]

    # 3x4 matrix -> append homogeneous bottom row.
    if (
        len(value) == 3
        and all(isinstance(row, list) and len(row) == 4 for row in value)
    ):
        value = [*value, [0.0, 0.0, 0.0, 1.0]]

    if not (
        len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
    ):
        raise ValueError(f"{source}: expected a 4x4 or 3x4 matrix")

    return [[float(x) for x in row] for row in value]


def basename_from_record(record: dict[str, Any], fallback: str | None = None) -> str:
    for key in NAME_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).name
    if fallback is not None:
        return Path(fallback).name
    raise ValueError(f"Cannot infer image name from record keys: {sorted(record.keys())}")


def parse_pose_payload(payload: Any, image_dir: Path | None = None) -> dict[str, list[list[float]]]:
    """Parse common PanoCity pose json forms into {image_name: 4x4}."""
    poses: dict[str, list[list[float]]] = {}

    # Common form: {"frames": [...]} or {"poses": [...]}.
    if isinstance(payload, dict):
        for list_key in ("frames", "poses", "data", "images"):
            if isinstance(payload.get(list_key), list):
                return parse_pose_payload(payload[list_key], image_dir)

        # Dict keyed by image names, or dict containing a single top-level wrapper.
        if all(isinstance(v, (dict, list)) for v in payload.values()):
            parsed_any = False
            for key, value in payload.items():
                # Skip non-frame metadata blocks.
                if key in {"camera_model", "intrinsics", "metadata", "height", "width"}:
                    continue
                try:
                    if isinstance(value, dict):
                        image_name = basename_from_record(value, fallback=key)
                        matrix = as_4x4_matrix(value, source=key)
                    else:
                        image_name = Path(key).name
                        matrix = as_4x4_matrix(value, source=key)
                    poses[image_name] = matrix
                    parsed_any = True
                except ValueError:
                    # Some wrappers may not be frame entries; ignore and fail later if none parsed.
                    continue
            if parsed_any:
                return poses

        # Single record should not happen for a full scene, but support it.
        if any(k in payload for k in POSE_KEYS):
            image_name = basename_from_record(payload)
            poses[image_name] = as_4x4_matrix(payload, source=image_name)
            return poses

    # List of frame records.
    if isinstance(payload, list):
        image_names = sorted_image_names(image_dir) if image_dir is not None else []
        for idx, record in enumerate(payload):
            if isinstance(record, dict):
                image_name = basename_from_record(
                    record,
                    fallback=image_names[idx] if idx < len(image_names) else f"pano_{idx:07d}.png",
                )
                matrix = as_4x4_matrix(record, source=image_name)
            else:
                image_name = image_names[idx] if idx < len(image_names) else f"pano_{idx:07d}.png"
                matrix = as_4x4_matrix(record, source=image_name)
            poses[image_name] = matrix
        return poses

    raise ValueError("Unsupported PanoCity pose JSON layout")


def sorted_image_names(image_dir: Path | None) -> list[str]:
    if image_dir is None or not image_dir.is_dir():
        return []
    return [p.name for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def find_panocity_blocks(source_root: Path, selected_cities: set[str] | None) -> list[Path]:
    blocks: list[Path] = []
    for city_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if selected_cities and city_dir.name not in selected_cities:
            continue
        for block_dir in sorted(p for p in city_dir.iterdir() if p.is_dir()):
            if (block_dir / "pano_images").is_dir() and list(block_dir.glob("*Pano*_poses.json")):
                blocks.append(block_dir)
    return blocks


def choose_pose_file(block_dir: Path) -> Path:
    candidates = sorted(block_dir.glob("*Pano*_poses.json"))
    if not candidates:
        raise FileNotFoundError(f"No panoramic pose json found in {block_dir}")
    if len(candidates) > 1:
        print(f"[WARN] multiple pose files in {block_dir}; using {candidates[0].name}")
    return candidates[0]


def link_or_copy_images(
    image_dir: Path,
    output_image_dir: Path,
    image_names: Iterable[str],
    *,
    copy: bool,
    overwrite: bool,
) -> int:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    missing: list[str] = []
    for image_name in image_names:
        src = image_dir / image_name
        if not src.is_file():
            missing.append(image_name)
            continue
        dst = output_image_dir / image_name
        if dst.exists() or dst.is_symlink():
            if overwrite:
                dst.unlink()
            else:
                raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
        count += 1
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images under {image_dir}; examples: {missing[:5]}"
        )
    return count


def convert_block(
    block_dir: Path,
    output_root: Path,
    *,
    prefix: str,
    copy: bool,
    overwrite: bool,
) -> Path:
    city = block_dir.parent.name
    block = block_dir.name
    image_dir = block_dir / "pano_images"
    pose_file = choose_pose_file(block_dir)

    with pose_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    poses = parse_pose_payload(payload, image_dir=image_dir)

    output_dir = output_root / f"{prefix}{city}_{block}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pose = output_dir / "camera_pose.json"
    if output_pose.exists() and not overwrite:
        raise FileExistsError(f"{output_pose} exists; pass --overwrite to replace it")

    linked = link_or_copy_images(
        image_dir,
        output_dir / "images",
        poses.keys(),
        copy=copy,
        overwrite=overwrite,
    )
    with output_pose.open("w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
        f.write("\n")

    print(f"{block_dir} -> {output_dir}  ({linked} frames)")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PanoCity blocks into GP/PanSplat camera_pose.json + images format."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Path to extracted PanoCity root")
    parser.add_argument("--output-root", type=Path, required=True, help="Output mix dataset root")
    parser.add_argument("--city", action="append", help="Optional city filter: beijing / jinan / ningbo. Repeatable")
    parser.add_argument("--prefix", default="panocity_", help="Prefix for output sequence directories")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing links/files")
    args = parser.parse_args()

    blocks = find_panocity_blocks(args.source_root, set(args.city) if args.city else None)
    if not blocks:
        raise SystemExit(f"No PanoCity blocks found under {args.source_root}")

    for block_dir in blocks:
        convert_block(
            block_dir,
            args.output_root,
            prefix=args.prefix,
            copy=args.copy,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
