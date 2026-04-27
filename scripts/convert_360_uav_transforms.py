#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def convert_transform(transform_path: Path, overwrite: bool) -> Path:
    output_path = transform_path.with_name("camera_pose.json")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    with transform_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{transform_path} does not contain a valid 'frames' list")

    camera_pose = {}
    for idx, frame in enumerate(frames):
        image_name = frame.get("image_name")
        transform_matrix = frame.get("transform_matrix")
        if not isinstance(image_name, str):
            raise ValueError(f"{transform_path} frame {idx} is missing 'image_name'")
        if (
            not isinstance(transform_matrix, list)
            or len(transform_matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in transform_matrix)
        ):
            raise ValueError(
                f"{transform_path} frame {idx} has an invalid 'transform_matrix'; expected 4x4 list"
            )
        camera_pose[image_name] = transform_matrix

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(camera_pose, f, indent=2)
        f.write("\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 360 UAV transform.json files into PanSplat camera_pose.json files."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan recursively for transform.json")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing camera_pose.json files if present",
    )
    args = parser.parse_args()

    transform_paths = sorted(
        path for path in args.root.rglob("transform.json") if path.is_file()
    )
    if not transform_paths:
        raise SystemExit(f"No transform.json files found under {args.root}")

    for transform_path in transform_paths:
        output_path = convert_transform(transform_path, args.overwrite)
        print(f"{transform_path} -> {output_path}")


if __name__ == "__main__":
    main()
