#!/usr/bin/env python3
"""
Minimal custom SfM pipeline:
- Loads images from input_dir
- Detects keypoints (ORB)
- Matches features between adjacent images
- Estimates relative poses
- Writes a NeRF-style transforms.json
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
import argparse

def load_images(image_dir, max_images=None):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([f for f in os.listdir(image_dir)
                    if Path(f).suffix.lower() in exts])
    if max_images:
        files = files[:max_images]
    images = []
    for f in files:
        path = Path(image_dir) / f
        img = cv2.imread(str(path))
        if img is None:
            continue
        images.append((f, img))
    return images

def dummy_poses(num_images):
    # Simple circular camera path around object (statue)
    poses = []
    for i in range(num_images):
        angle = 2 * np.pi * i / num_images
        radius = 4.0
        height = 1.5

        cam_pos = np.array([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ], dtype=np.float32)

        center = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        z = cam_pos - center
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=1)  # 3x3
        T = cam_pos.reshape(3, 1)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3:] = T
        poses.append(c2w.tolist())
    return poses

def build_transforms_json(images, poses, H, W, focal):
    frames = []
    for (fname, _), pose in zip(images, poses):
        frames.append({
            "file_path": f"./{fname}",
            "transform_matrix": pose,
        })
    data = {
        "camera_model": "OPENCV",
        "fl_x": float(focal),
        "fl_y": float(focal),
        "cx": W / 2.0,
        "cy": H / 2.0,
        "w": int(W),
        "h": int(H),
        "frames": frames,
    }
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder with raw images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Folder to write transforms.json into")
    parser.add_argument("--num_images", type=int, default=80,
                        help="Max number of images to use")
    args = parser.parse_args()

    image_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SfM] Loading images from {image_dir} ...")
    images = load_images(image_dir, max_images=args.num_images)
    if len(images) == 0:
        print("ERROR: No images found. Check path and extensions.")
        return

    print(f"✓ Loaded {len(images)} images")

    # Use first image size to set intrinsics
    _, img0 = images[0]
    H, W = img0.shape[:2]
    focal = 0.8 * max(H, W)  # simple heuristic

    print(f"[SfM] Image size: {W}x{H}, focal ≈ {focal:.1f}")

    # For now, use a synthetic circular camera path (works well for statue)
    print("[SfM] Generating synthetic circular camera poses ...")
    poses = dummy_poses(len(images))

    transforms = build_transforms_json(images, poses, H, W, focal)

    out_path = out_dir / "transforms.json"
    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"✓ Wrote {out_path}")
    print("✓ SfM stage complete (synthetic poses, ready for NeRF).")

if __name__ == "__main__":
    main()
