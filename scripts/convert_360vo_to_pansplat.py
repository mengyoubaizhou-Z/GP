#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def quaternion_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> list[list[float]]:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def parse_gt(gt_path: Path) -> dict[str, list[list[float]]]:
    poses: dict[str, list[list[float]]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) != 9:
                if len(parts) == 1:
                    # Some sequences contain a trailing orphan frame index line.
                    # Skip it as long as it does not describe a complete pose.
                    continue
                raise ValueError(
                    f"{gt_path} line {line_index} should contain 9 columns, got {len(parts)}"
                )

            _, frame_name, tx, ty, tz, qx, qy, qz, qw = parts
            rotation = quaternion_to_rotation_matrix(
                float(qx), float(qy), float(qz), float(qw)
            )
            matrix = [
                [rotation[0][0], rotation[0][1], rotation[0][2], float(tx)],
                [rotation[1][0], rotation[1][1], rotation[1][2], float(ty)],
                [rotation[2][0], rotation[2][1], rotation[2][2], float(tz)],
                [0.0, 0.0, 0.0, 1.0],
            ]
            poses[frame_name] = matrix
    return poses


def symlink_images(source_dir: Path, target_dir: Path, expected_frames: set[str]) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.name not in expected_frames:
            continue

        target_path = target_dir / image_path.name
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        target_path.symlink_to(image_path)
        count += 1
    return count


def convert_sequence(source_root: Path, output_root: Path, sequence: str, prefix: str) -> Path:
    sequence_root = source_root / sequence
    image_dir = sequence_root / "images"
    gt_path = sequence_root / "gt.txt"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing gt file: {gt_path}")

    poses = parse_gt(gt_path)
    output_dir = output_root / f"{prefix}{sequence}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_count = symlink_images(image_dir, output_dir / "images", set(poses.keys()))
    if image_count != len(poses):
        missing_images = sorted(set(poses.keys()) - {p.name for p in image_dir.iterdir() if p.is_file()})
        raise ValueError(
            f"{sequence}: linked image count ({image_count}) does not match pose count ({len(poses)}). "
            f"Missing images: {missing_images[:5]}"
        )

    with (output_dir / "camera_pose.json").open("w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
        f.write("\n")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 360VO sequences into the PanSplat 360uav-style input format."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path containing seq0 ... seq9 directories from 360VO.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where PanSplat-formatted sequences will be written.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        required=True,
        help="Sequence name to convert, e.g. seq0. Repeat for multiple sequences.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="360vo_",
        help="Prefix added to each output sequence directory name.",
    )
    args = parser.parse_args()

    for sequence in args.sequence:
        output_dir = convert_sequence(args.source_root, args.output_root, sequence, args.prefix)
        print(output_dir)


if __name__ == "__main__":
    main()
