#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def find_image_dir(seq_dir: Path) -> Path:
    for name in ["images", "image"]:
        p = seq_dir / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Cannot find images/ or image/ under {seq_dir}")


def extract_sequence(src_seq: Path, dst_root: Path, start: int, end: int, mode: str):
    pose_path = src_seq / "camera_pose.json"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing camera_pose.json: {pose_path}")

    image_dir = find_image_dir(src_seq)
    image_dir_name = image_dir.name

    with pose_path.open("r", encoding="utf-8") as f:
        poses = json.load(f)

    frames = list(poses.keys())
    if start < 0 or end >= len(frames) or start > end:
        raise ValueError(
            f"{src_seq.name}: invalid range [{start}, {end}], "
            f"but sequence has {len(frames)} frames."
        )

    selected = frames[start:end + 1]

    dst_seq = dst_root / src_seq.name
    dst_img_dir = dst_seq / image_dir_name
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    out_poses = {}
    for name in selected:
        src_img = image_dir / name
        if not src_img.is_file():
            raise FileNotFoundError(f"Missing image: {src_img}")

        dst_img = dst_img_dir / name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()

        if mode == "copy":
            shutil.copy2(src_img, dst_img)
        elif mode == "symlink":
            dst_img.symlink_to(src_img.resolve())
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out_poses[name] = poses[name]

    with (dst_seq / "camera_pose.json").open("w", encoding="utf-8") as f:
        json.dump(out_poses, f, indent=2)
        f.write("\n")

    print(f"[OK] {src_seq.name}: {len(selected)} frames -> {dst_seq}")
    print(f"     first: {selected[0]}")
    print(f"     last : {selected[-1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--scene", action="append", required=True)
    args = parser.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)

    for scene in args.scene:
        extract_sequence(
            args.src_root / scene,
            args.dst_root,
            args.start,
            args.end,
            args.mode,
        )


if __name__ == "__main__":
    main()
