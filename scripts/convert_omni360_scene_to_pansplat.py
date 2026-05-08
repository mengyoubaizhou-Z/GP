#!/usr/bin/env python3
"""Convert extracted Omni360-Scene collections into the GP/PanSplat mix dataset format.

Expected output per detected sequence:
    <output-root>/<prefix><relative-sequence-name>/
        images/
        camera_pose.json

Because Omni360-Scene is distributed as large ZIP archives and the inner Raw archive layout may
change across CityPark / DTW / NYC, this converter is deliberately auto-detecting:
- image folders: images, image, pano_images, panoramas, panorama, rgb, raw, Raw
- pose files: camera_pose.json, transform*.json, *poses*.json, pose*.json,
              frame_trajectory.txt, trajectory.txt, airsim_rec.txt, AirSimRecording.txt

Supported pose layouts:
- JSON frames/list/dict with transform_matrix / transformation_matrix / pose / c2w
- TXT: 7-col tx ty tz qx qy qz qw
- TXT: 8-col timestamp tx ty tz qx qy qz qw, or tx ty tz qx qy qz qw image_name
- TXT: 9-col index image_name tx ty tz qx qy qz qw
- TXT: 12/13-col KITTI-style 3x4 matrix, optional leading frame index
- TXT: 16/17-col 4x4 matrix, optional leading frame index
- AirSim recording txt with headers containing POS_X/POS_Y/POS_Z and Q_W/Q_X/Q_Y/Q_Z
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_DIR_NAMES = {"images", "image", "pano_images", "panoramas", "panorama", "rgb", "raw", "Raw"}
POSE_JSON_PATTERNS = (
    "camera_pose.json",
    "transform.json",
    "transforms.json",
    "transforms_*.json",
    "*poses*.json",
    "pose*.json",
)
POSE_TXT_PATTERNS = (
    "frame_trajectory.txt",
    "keyframe_trajectory.txt",
    "trajectory.txt",
    "poses.txt",
    "pose.txt",
    "gt.txt",
    "airsim_rec.txt",
    "AirSimRecording.txt",
    "*_rec.txt",
)
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


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def make_matrix(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    r = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    return [
        [r[0][0], r[0][1], r[0][2], float(tx)],
        [r[1][0], r[1][1], r[1][2], float(ty)],
        [r[2][0], r[2][1], r[2][2], float(tz)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def as_4x4_matrix(value: Any, *, source: str) -> list[list[float]]:
    if isinstance(value, dict):
        for key in POSE_KEYS:
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"{source}: pose is not a list or dict")
    if len(value) == 16 and all(isinstance(x, (int, float)) for x in value):
        value = [value[i : i + 4] for i in range(0, 16, 4)]
    if len(value) == 12 and all(isinstance(x, (int, float)) for x in value):
        value = [value[i : i + 4] for i in range(0, 12, 4)]
    if len(value) == 3 and all(isinstance(row, list) and len(row) == 4 for row in value):
        value = [*value, [0.0, 0.0, 0.0, 1.0]]
    if not (len(value) == 4 and all(isinstance(row, list) and len(row) == 4 for row in value)):
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


def image_files(image_dir: Path) -> list[Path]:
    return [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def image_name_by_index(images: list[Path], idx: int, fallback_ext: str = ".png") -> str:
    if 0 <= idx < len(images):
        return images[idx].name
    return f"{idx:06d}{fallback_ext}"


def parse_json_payload(payload: Any, image_dir: Path) -> dict[str, list[list[float]]]:
    images = image_files(image_dir)
    poses: dict[str, list[list[float]]] = {}

    if isinstance(payload, dict):
        for list_key in ("frames", "poses", "data", "images"):
            if isinstance(payload.get(list_key), list):
                return parse_json_payload(payload[list_key], image_dir)
        if all(isinstance(v, (dict, list)) for v in payload.values()):
            parsed_any = False
            for key, value in payload.items():
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
                    continue
            if parsed_any:
                return poses
        if any(k in payload for k in POSE_KEYS):
            image_name = basename_from_record(payload, fallback=image_name_by_index(images, 0))
            poses[image_name] = as_4x4_matrix(payload, source=image_name)
            return poses

    if isinstance(payload, list):
        for idx, record in enumerate(payload):
            if isinstance(record, dict):
                image_name = basename_from_record(record, fallback=image_name_by_index(images, idx))
                matrix = as_4x4_matrix(record, source=image_name)
            else:
                image_name = image_name_by_index(images, idx)
                matrix = as_4x4_matrix(record, source=image_name)
            poses[image_name] = matrix
        return poses

    raise ValueError("Unsupported JSON pose layout")


def split_pose_line(line: str) -> list[str]:
    return [p for p in re.split(r"[\s,]+", line.strip()) if p]


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_airsim_recording_txt(path: Path) -> dict[str, list[list[float]]] | None:
    """Parse AirSim-style recording files when a recognizable header is present."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if not first:
            return None
        delimiter = "\t" if "\t" in first else None
        header = [h.strip() for h in (first.split("\t") if delimiter == "\t" else first.split())]
        upper = {h.upper(): i for i, h in enumerate(header)}
        required = ["POS_X", "POS_Y", "POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z"]
        if not all(k in upper for k in required):
            return None
        image_key = next((k for k in ("IMAGEFILE", "IMAGE_FILE", "FILENAME", "FILE") if k in upper), None)
        if image_key is None:
            return None

        poses: dict[str, list[list[float]]] = {}
        reader = csv.reader(f, delimiter="\t" if delimiter == "\t" else " ")
        for row_idx, row in enumerate(reader):
            row = [x for x in row if x != ""]
            if not row or len(row) <= max(upper.values()):
                continue
            image_name = Path(row[upper[image_key]]).name
            tx = float(row[upper["POS_X"]])
            ty = float(row[upper["POS_Y"]])
            tz = float(row[upper["POS_Z"]])
            qw = float(row[upper["Q_W"]])
            qx = float(row[upper["Q_X"]])
            qy = float(row[upper["Q_Y"]])
            qz = float(row[upper["Q_Z"]])
            poses[image_name] = make_matrix(tx, ty, tz, qx, qy, qz, qw)
        return poses


def parse_txt_pose_file(path: Path, image_dir: Path) -> dict[str, list[list[float]]]:
    airsim = parse_airsim_recording_txt(path)
    if airsim is not None and airsim:
        return airsim

    images = image_files(image_dir)
    poses: dict[str, list[list[float]]] = {}
    sequential_idx = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = split_pose_line(stripped)
            # Skip textual headers.
            if not all(is_float(p) for p in parts if Path(p).suffix.lower() not in IMAGE_EXTS):
                if line_idx == 1:
                    continue

            # 360VO-like: index image_name tx ty tz qx qy qz qw
            if len(parts) == 9 and not is_float(parts[1]):
                _, image_name, tx, ty, tz, qx, qy, qz, qw = parts
                poses[Path(image_name).name] = make_matrix(
                    float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)
                )
                continue

            # tx ty tz qx qy qz qw image_name
            if len(parts) == 8 and not is_float(parts[-1]):
                tx, ty, tz, qx, qy, qz, qw, image_name = parts
                poses[Path(image_name).name] = make_matrix(
                    float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)
                )
                continue

            values = [float(x) for x in parts]

            # 7-col: tx ty tz qx qy qz qw
            if len(values) == 7:
                tx, ty, tz, qx, qy, qz, qw = values
                image_name = image_name_by_index(images, sequential_idx)
                poses[image_name] = make_matrix(tx, ty, tz, qx, qy, qz, qw)
                sequential_idx += 1
                continue

            # TUM-like 8-col: timestamp tx ty tz qx qy qz qw
            if len(values) == 8:
                _, tx, ty, tz, qx, qy, qz, qw = values
                image_name = image_name_by_index(images, sequential_idx)
                poses[image_name] = make_matrix(tx, ty, tz, qx, qy, qz, qw)
                sequential_idx += 1
                continue

            # KITTI-like 3x4, optional leading frame index.
            if len(values) in {12, 13}:
                if len(values) == 13:
                    frame_idx = int(values[0])
                    matrix_values = values[1:]
                    image_name = image_name_by_index(images, frame_idx)
                else:
                    matrix_values = values
                    image_name = image_name_by_index(images, sequential_idx)
                    sequential_idx += 1
                matrix_3x4 = [matrix_values[i : i + 4] for i in range(0, 12, 4)]
                poses[image_name] = as_4x4_matrix(matrix_3x4, source=f"{path}:{line_idx}")
                continue

            # 4x4, optional leading frame index.
            if len(values) in {16, 17}:
                if len(values) == 17:
                    frame_idx = int(values[0])
                    matrix_values = values[1:]
                    image_name = image_name_by_index(images, frame_idx)
                else:
                    matrix_values = values
                    image_name = image_name_by_index(images, sequential_idx)
                    sequential_idx += 1
                poses[image_name] = as_4x4_matrix(matrix_values, source=f"{path}:{line_idx}")
                continue

            raise ValueError(f"{path} line {line_idx}: unsupported pose format with {len(parts)} columns")

    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return poses


def find_image_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if path.name in IMAGE_DIR_NAMES and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in path.iterdir()):
            dirs.append(path)
    return sorted(dirs)


def find_pose_files_near(sequence_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in POSE_JSON_PATTERNS:
        candidates.extend(sequence_root.glob(pattern))
    for pattern in POSE_TXT_PATTERNS:
        candidates.extend(sequence_root.glob(pattern))
    # Some datasets put pose files one level above the image directory.
    parent = sequence_root.parent
    for pattern in POSE_JSON_PATTERNS:
        candidates.extend(parent.glob(pattern))
    for pattern in POSE_TXT_PATTERNS:
        candidates.extend(parent.glob(pattern))
    # Deduplicate while preserving stable order.
    seen = set()
    unique = []
    for p in sorted(candidates):
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def parse_pose_file(path: Path, image_dir: Path) -> dict[str, list[list[float]]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return parse_json_payload(payload, image_dir)
    return parse_txt_pose_file(path, image_dir)


def sanitize_sequence_name(path: Path, source_root: Path, prefix: str) -> str:
    rel = path.relative_to(source_root).as_posix()
    if rel.endswith("/images") or rel.endswith("/image"):
        rel = str(Path(rel).parent)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel).strip("_")
    return f"{prefix}{name}"


def link_or_copy_images(
    image_dir: Path,
    output_image_dir: Path,
    image_names: Iterable[str],
    *,
    copy: bool,
    overwrite: bool,
) -> int:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    count = 0
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


def convert_sequence(
    image_dir: Path,
    source_root: Path,
    output_root: Path,
    *,
    prefix: str,
    copy: bool,
    overwrite: bool,
    pose_file: Path | None = None,
) -> Path | None:
    pose_candidates = [pose_file] if pose_file else find_pose_files_near(image_dir)
    pose_candidates = [p for p in pose_candidates if p is not None]
    if not pose_candidates:
        print(f"[SKIP] no pose file found near {image_dir}")
        return None

    last_error: Exception | None = None
    poses: dict[str, list[list[float]]] | None = None
    chosen_pose: Path | None = None
    for candidate in pose_candidates:
        try:
            poses = parse_pose_file(candidate, image_dir)
            chosen_pose = candidate
            break
        except Exception as exc:  # try next candidate; report final if all fail
            last_error = exc
            continue
    if poses is None or chosen_pose is None:
        print(f"[SKIP] failed to parse poses near {image_dir}: {last_error}")
        return None

    output_dir = output_root / sanitize_sequence_name(image_dir, source_root, prefix)
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
    print(f"{image_dir} + {chosen_pose.name} -> {output_dir}  ({linked} frames)")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect and convert extracted Omni360-Scene raw panoramas into GP/PanSplat format."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Extracted Omni360-Scene root or one collection root")
    parser.add_argument("--output-root", type=Path, required=True, help="Output mix dataset root")
    parser.add_argument("--image-dir", type=Path, help="Optional explicit image directory; disables auto-detection")
    parser.add_argument("--pose-file", type=Path, help="Optional explicit pose file for --image-dir")
    parser.add_argument("--prefix", default="omni360_", help="Prefix for output sequence directories")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing links/files")
    args = parser.parse_args()

    image_dirs = [args.image_dir] if args.image_dir else find_image_dirs(args.source_root)
    image_dirs = [p for p in image_dirs if p is not None]
    if not image_dirs:
        raise SystemExit(
            f"No image directories found under {args.source_root}. "
            "Extract the Raw zip first, or pass --image-dir explicitly."
        )

    converted = 0
    for image_dir in image_dirs:
        result = convert_sequence(
            image_dir,
            args.source_root,
            args.output_root,
            prefix=args.prefix,
            copy=args.copy,
            overwrite=args.overwrite,
            pose_file=args.pose_file,
        )
        converted += int(result is not None)

    if converted == 0:
        raise SystemExit("No sequences were converted. Check the extracted Raw archive layout and pose-file names.")


if __name__ == "__main__":
    main()
