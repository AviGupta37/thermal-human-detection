"""Active learning: choose which unlabelled frames to annotate next.

After a first model is trained on ~200-300 frames, annotating the frames the
model is *unsure about* improves accuracy much faster than annotating random
ones. This script scores every unlabelled frame by
  * detections with mid confidence (the model is unsure), and
  * warm person-sized blobs (hotspot detector) the model did not detect
    (possible misses),
then writes the top --n frames to a list for annotate.py.

    python extract_frames.py --source new_flight.mkv --max 400      # big pool
    python select_hard_frames.py --weights runs/person/weights/best.pt --n 100
    python annotate.py --list dataset/raw/priority.txt --model runs/person/weights/best.pt
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from thermal_utils import HotspotDetector, group_of, iou_matrix, label_path_for, list_images, trained_imgsz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=str(config.RAW_IMAGES))
    ap.add_argument("--labels", default=str(config.RAW_LABELS))
    ap.add_argument("--out", default=str(config.DATA_DIR / "raw" / "priority.txt"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-per-clip", type=int, default=25, help="keep the selection diverse")
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--high", type=float, default=0.60)
    ap.add_argument("--person-px", type=int, default=30, help="hotspot detector person height")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    imgsz = trained_imgsz(model, config.IMGSZ)
    hot = HotspotDetector(person_px=args.person_px)

    pool = [p for p in list_images(args.images) if not label_path_for(p, args.labels).exists()]
    if not pool:
        raise SystemExit("no unlabelled images in the pool")
    print(f"scoring {len(pool)} unlabelled frames ...")
    scored = []
    for res in model.predict([str(p) for p in pool], imgsz=imgsz, conf=args.low, device=args.device,
                             verbose=False, stream=True):
        conf = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        unsure = conf[(conf < args.high)]
        # 1 at conf 0.5, 0 at the extremes
        s_unsure = float(np.sum(1 - np.abs(2 * unsure - 1)))
        hb, _ = hot(cv2.imread(res.path, cv2.IMREAD_UNCHANGED))
        if len(hb):
            covered = (iou_matrix(hb, boxes).max(axis=1) > 0.1) if len(boxes) else np.zeros(len(hb), bool)
            s_miss = 0.5 * float((~covered).sum())
        else:
            s_miss = 0.0
        scored.append((s_unsure + min(s_miss, 3.0), Path(res.path)))

    scored.sort(key=lambda t: -t[0])
    per_clip = defaultdict(int)
    chosen = []
    for s, p in scored:
        g = group_of(p.stem)
        if per_clip[g] >= args.max_per_clip:
            continue
        per_clip[g] += 1
        chosen.append(p)
        if len(chosen) >= args.n:
            break
    Path(args.out).write_text("\n".join(str(p) for p in chosen) + "\n")
    print(f"wrote {len(chosen)} frames to {args.out} (top score {scored[0][0]:.2f})")
    print(f"next: python annotate.py --list {args.out} --model {args.weights}")


if __name__ == "__main__":
    main()
