"""Split annotated frames into train/val/test and write data.yaml.

    python split_dataset.py                  # 75/15/10 by clip
    python split_dataset.py --val 0.2 --test 0.1 --seed 1

The split is done by CLIP (the part of the file name before "__"), not by
frame. Neighbouring frames from one flight are near-duplicates; if they land in
both train and val, validation mAP looks great and means nothing. With few
clips the fractions are approximate.

Only images that have a label file are used (an empty file = negative image).
The script also prints label statistics and warns about common problems.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

import config
from thermal_utils import group_of, label_path_for, list_images, read_yolo_labels


def assign_groups(groups: dict[str, list], val: float, test: float, seed: int) -> dict[str, str]:
    """Greedy: shuffle clips, fill test then val up to their share of frames."""
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = sum(len(groups[g]) for g in names)
    want = {"test": test * total, "val": val * total}
    have = {"test": 0, "val": 0}
    out = {}
    for g in names:
        n = len(groups[g])
        for split in ("test", "val"):
            if want[split] > 0 and have[split] + n / 2 <= want[split]:
                out[g] = split
                have[split] += n
                break
        else:
            out[g] = "train"
    # Every split needs something; steal the smallest train clip if empty.
    for split, frac in (("val", val), ("test", test)):
        if frac > 0 and split not in out.values():
            train = [g for g in names if out[g] == "train"]
            if len(train) > 1:
                out[min(train, key=lambda g: len(groups[g]))] = split
    return out


def frame_split(items: list, val: float, test: float, seed: int) -> dict:
    """Fallback when there is only one clip: contiguous blocks by frame order,
    so val/test are at least separate stretches of the flight."""
    items = sorted(items)
    n = len(items)
    n_test, n_val = int(round(n * test)), int(round(n * val))
    out = {}
    for i, it in enumerate(items):
        out[it] = "test" if i >= n - n_test else ("val" if i >= n - n_test - n_val else "train")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(config.RAW_IMAGES))
    ap.add_argument("--labels", default=str(config.RAW_LABELS))
    ap.add_argument("--out", default=str(config.SPLIT_DIR))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = ap.parse_args()

    images = [p for p in list_images(args.images) if label_path_for(p, args.labels).exists()]
    if not images:
        raise SystemExit("no labelled images found - run annotate.py first")

    groups = defaultdict(list)
    for p in images:
        groups[group_of(p.stem)].append(p)
    if len(groups) >= 3:
        g2s = assign_groups(groups, args.val, args.test, args.seed)
        split_of = {p: g2s[group_of(p.stem)] for p in images}
    else:
        print(f"WARNING: only {len(groups)} clip(s). Splitting by frame blocks; record more separate "
              "flights so validation measures generalisation, not memory.")
        split_of = frame_split(images, args.val, args.test, args.seed)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    stats = {s: {"images": 0, "negatives": 0, "boxes": 0, "heights": [], "clips": set()} for s in ("train", "val", "test")}
    for p, split in split_of.items():
        lp = label_path_for(p, args.labels)
        for kind, src in (("images", p), ("labels", lp)):
            dst = out / kind / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        boxes = read_yolo_labels(lp, w, h)
        st = stats[split]
        st["images"] += 1
        st["negatives"] += not boxes
        st["boxes"] += len(boxes)
        st["heights"] += [b[4] - b[2] for b in boxes]
        st["clips"].add(group_of(p.stem))

    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(config.CLASSES)}}
    if stats["test"]["images"]:
        data["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"\n{'split':6} {'images':>7} {'neg':>5} {'boxes':>6} {'clips':>6}  person height px (p10/median/p90)")
    for s, st in stats.items():
        hts = np.array(st["heights"]) if st["heights"] else np.array([0.0])
        print(f"{s:6} {st['images']:7d} {st['negatives']:5d} {st['boxes']:6d} {len(st['clips']):6d}  "
              f"{np.percentile(hts, 10):.0f} / {np.median(hts):.0f} / {np.percentile(hts, 90):.0f}")
    print(f"\nwrote {out / 'data.yaml'}")

    tr = stats["train"]
    all_h = np.array(sum((st["heights"] for st in stats.values()), [])) if tr["boxes"] else np.array([])
    if tr["images"] and tr["negatives"] / tr["images"] < 0.05:
        print("TIP: <5% negative (empty) images. Add 5-15% frames with no people but with hot clutter.")
    if len(all_h) and np.median(all_h) < 16:
        print(f"TIP: median person height {np.median(all_h):.0f}px is small - train with --imgsz 960 or 1280, "
              "and/or run detect.py with --tile 2.")
    unknown = [lp for lp in Path(args.labels).glob("*.txt")
               if any(int(l.split()[0]) >= len(config.CLASSES) for l in lp.read_text().split("\n") if l.strip())]
    if unknown:
        print(f"WARNING: {len(unknown)} label files use class ids not in config.CLASSES, e.g. {unknown[0].name}")


if __name__ == "__main__":
    main()
