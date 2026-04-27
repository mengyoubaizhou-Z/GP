#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
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


def parse_groundtruth(groundtruth_path: Path) -> dict[str, list[list[float]]]:
    poses: dict[str, list[list[float]]] = {}
    with groundtruth_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = [float(x) for x in stripped.split()]
            if len(values) != 7:
                raise ValueError(
                    f"{groundtruth_path} line {index} should contain 7 floats, got {len(values)}"
                )
            tx, ty, tz, qx, qy, qz, qw = values
            rotation = quaternion_to_rotation_matrix(qx, qy, qz, qw)
            matrix = [
                [rotation[0][0], rotation[0][1], rotation[0][2], tx],
                [rotation[1][0], rotation[1][1], rotation[1][2], ty],
                [rotation[2][0], rotation[2][1], rotation[2][2], tz],
                [0.0, 0.0, 0.0, 1.0],
            ]
            poses[f"{index:04d}.jpg"] = matrix
    return poses


def link_images(source_dir: Path, target_dir: Path) -> int:
    count = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(source_dir.glob("*.jpg")):
        target_path = target_dir / image_path.name
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        target_path.symlink_to(image_path)
        count += 1
    return count


def convert_sequence(source_root: Path, output_root: Path, sequence: str) -> Path:
    image_dir = source_root / "Sequences" / sequence
    groundtruth_path = source_root / "GroundTruth" / f"{sequence}.txt"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not groundtruth_path.is_file():
        raise FileNotFoundError(f"Missing groundtruth file: {groundtruth_path}")

    output_dir = output_root / sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = parse_groundtruth(groundtruth_path)
    image_count = link_images(image_dir, output_dir / "images")
    if image_count != len(poses):
        raise ValueError(
            f"{sequence}: image count ({image_count}) does not match pose count ({len(poses)})"
        )

    with (output_dir / "camera_pose.json").open("w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
        f.write("\n")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 360DVO sequences into the PanSplat 360uav-style input format."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path containing GroundTruth/ and Sequences/ from 360DVO.",
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
        help="Sequence name to convert, e.g. mountains or drone_racetrack. Repeat for multiple sequences.",
    )
    args = parser.parse_args()

    for sequence in args.sequence:
        output_dir = convert_sequence(args.source_root, args.output_root, sequence)
        print(output_dir)


if __name__ == "__main__":
    main()
