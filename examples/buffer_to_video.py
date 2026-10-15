#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


def install_numpy_pickle_compat():
    try:
        import numpy.core as np_core

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a selected segment of HIL-SERL buffer transitions to a video."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Path to a transitions_*.pkl file.")
    source.add_argument("--buffer-dir", type=Path, help="Directory containing transitions_*.pkl files.")
    parser.add_argument(
        "--step",
        default="latest",
        help="Step to load from --buffer-dir, for example 998000. Default: latest.",
    )
    parser.add_argument("--output", type=Path, help="Output video path. Default is under /tmp.")
    parser.add_argument("--start", type=int, default=0, help="Start transition index. Default: 0.")
    parser.add_argument("--end", type=int, help="End transition index, exclusive.")
    parser.add_argument("--count", type=int, default=120, help="Max frames to write. Default: 120.")
    parser.add_argument("--stride", type=int, default=1, help="Transition stride. Default: 1.")
    parser.add_argument("--fps", type=int, default=20, help="Output FPS. Default: 20.")
    parser.add_argument(
        "--camera",
        default="both",
        help="Camera key, comma-separated keys, or 'both'. Default: both -> front,wrist.",
    )
    parser.add_argument(
        "--obs-key",
        choices=("observations", "next_observations"),
        default="observations",
        help="Transition observation dict to render. Default: observations.",
    )
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="How to compose multiple cameras. Default: horizontal.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resize output frames by this factor. Default: 1.0.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Do not draw index/reward/done text.")
    return parser.parse_args()


def step_from_name(path):
    match = re.search(r"transitions_(\d+)\.pkl$", path.name)
    return int(match.group(1)) if match else -1


def resolve_file(args):
    if args.file:
        return args.file

    files = sorted(args.buffer_dir.glob("transitions_*.pkl"), key=step_from_name)
    if not files:
        raise FileNotFoundError(f"No transitions_*.pkl files found in {args.buffer_dir}")

    if args.step == "latest":
        return files[-1]

    target = args.buffer_dir / f"transitions_{int(args.step)}.pkl"
    if not target.exists():
        raise FileNotFoundError(f"Could not find {target}")
    return target


def camera_keys(camera):
    if camera == "both":
        return ["front", "wrist"]
    return [key.strip() for key in camera.split(",") if key.strip()]


def to_uint8_image(value):
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape HxWx3 or 1xHxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def resize_if_needed(image, scale):
    if scale == 1.0:
        return image
    from PIL import Image

    h, w = image.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))


def compose_frame(transition, keys, obs_key, layout, scale):
    obs = transition[obs_key]
    frames = [resize_if_needed(to_uint8_image(obs[key]), scale) for key in keys]
    if len(frames) == 1:
        return frames[0]

    if layout == "horizontal":
        height = max(frame.shape[0] for frame in frames)
        padded = []
        for frame in frames:
            if frame.shape[0] == height:
                padded.append(frame)
            else:
                pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                padded.append(np.concatenate([frame, pad], axis=0))
        return np.concatenate(padded, axis=1)

    width = max(frame.shape[1] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[1] == width:
            padded.append(frame)
        else:
            pad = np.zeros((frame.shape[0], width - frame.shape[1], 3), dtype=np.uint8)
            padded.append(np.concatenate([frame, pad], axis=1))
    return np.concatenate(padded, axis=0)


def add_overlay(frame, source_name, transition_index, transition):
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    reward = transition.get("rewards", "?")
    done = transition.get("dones", "?")
    mask = transition.get("masks", "?")
    text = f"{source_name} | idx={transition_index} reward={reward} done={done} mask={mask}"
    margin = 4
    bbox = draw.textbbox((margin, margin), text)
    draw.rectangle(
        (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2),
        fill=(255, 255, 255),
    )
    draw.text((margin, margin), text, fill=(0, 0, 0))
    return np.asarray(image)


def open_writer(output, fps):
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(output, fps=fps, macro_block_size=1)


def main():
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    install_numpy_pickle_compat()
    pkl_path = resolve_file(args)
    keys = camera_keys(args.camera)

    print(f"Loading {pkl_path}")
    with pkl_path.open("rb") as f:
        transitions = pickle.load(f)

    total = len(transitions)
    end = args.end if args.end is not None else total
    end = min(end, total)
    indices = list(range(args.start, end, args.stride))[: args.count]
    if not indices:
        raise ValueError(f"No frames selected from total={total}, start={args.start}, end={end}")

    if args.output is None:
        step = step_from_name(pkl_path)
        safe_camera = args.camera.replace(",", "_")
        args.output = Path(
            f"/tmp/{pkl_path.stem}_idx{indices[0]}-{indices[-1]}_{safe_camera}.mp4"
        )

    print(
        f"Writing {len(indices)} frames from {pkl_path.name} "
        f"(total transitions: {total}) to {args.output}"
    )
    print(f"cameras={keys}, obs_key={args.obs_key}, fps={args.fps}, stride={args.stride}")

    writer = open_writer(args.output, args.fps)
    try:
        for idx in indices:
            transition = transitions[idx]
            frame = compose_frame(transition, keys, args.obs_key, args.layout, args.scale)
            if not args.no_overlay:
                frame = add_overlay(frame, pkl_path.name, idx, transition)
            writer.append_data(frame)
    finally:
        writer.close()

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Done: {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
