"""Pick diverse frames from recorded flights for annotation.

Consecutive video frames are almost identical, so 600 frames taken back to
back from one clip teach the model far less than 600 frames spread over many
flights, altitudes, times of day and backgrounds. This script samples at most
one frame every --every seconds and drops frames that look too much like the
last frame it kept.

    python extract_frames.py --source flight03.mkv --every 1.0 --max 80
    python extract_frames.py --source rtsp://... --tag field_day2 --max 100

Frames are saved as <tag>__<frame>.png in dataset/raw/images, already
preprocessed exactly as detect.py will preprocess live frames.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config
from camera_feed import FrameSource
from thermal_utils import preprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, folder, USB index or stream URL")
    ap.add_argument("--tag", help="clip name used in file names and for the train/val split (default: file stem)")
    ap.add_argument("--out", default=str(config.RAW_IMAGES))
    ap.add_argument("--every", type=float, default=1.0, help="minimum seconds between kept frames")
    ap.add_argument("--min-diff", type=float, default=4.0,
                    help="minimum mean abs difference (0-255) from the last kept frame")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="drop frames with Laplacian variance below this (motion blur); 0 = off")
    ap.add_argument("--max", type=int, default=100, help="stop after this many frames")
    ap.add_argument("--preprocess", default=config.PREPROCESS, choices=["none", "clahe"])
    args = ap.parse_args()

    tag = args.tag or (Path(args.source).stem if not args.source.isdigit() else f"usb{args.source}")
    tag = tag.replace(config.GROUP_SEP, "_")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept, last_small, last_t, now = 0, None, -1e9, 0.0
    with FrameSource(args.source) as src:
        fps = src.fps
        step_frames = 1 if (src.is_stream or Path(args.source).is_dir()) else max(1, int(round(args.every * fps)))
        idx = 0
        while kept < args.max:
            ok, frame = src.read()
            if not ok:
                break
            idx += 1
            if src.is_stream:
                now = time.time()
                if now - last_t < args.every:
                    continue
            elif (idx - 1) % step_frames:
                continue
            img = preprocess(frame, args.preprocess)
            gray = img[:, :, 0]
            small = cv2.resize(gray, (160, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
            if last_small is not None and np.abs(small - last_small).mean() < args.min_diff:
                continue
            if args.min_sharpness and cv2.Laplacian(gray, cv2.CV_64F).var() < args.min_sharpness:
                continue
            name = out / f"{tag}{config.GROUP_SEP}{idx:06d}.png"
            cv2.imwrite(str(name), img)
            last_small = small
            if src.is_stream:
                last_t = now
            kept += 1
            print(f"\r{kept} frames kept (frame {idx})", end="", flush=True)
    print(f"\nsaved {kept} frames from {args.source} to {out}")


if __name__ == "__main__":
    main()
