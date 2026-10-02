"""Convert the public HIT-UAV aerial thermal dataset to a person-only YOLO set
for pre-training, before fine-tuning on your own 500-700 frames.

HIT-UAV: 2,898 infrared images (640x512) from a DJI drone at 60-130 m, day and
night. Classes: 0 Person, 1 Car, 2 Bicycle, 3 OtherVehicle, 4 DontCare.
Get it from https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
or Kaggle ("HIT-UAV: A high-altitude infrared thermal dataset").

    python prepare_hituav.py --src ~/Downloads/hit-uav
    python train.py --data dataset/hituav_person/data.yaml --name hituav_pre

The script searches --src for YOLO-style images/<split> + labels/<split>
folders, or a COCO json per split, keeps only the person class and writes
dataset/hituav_person/{images,labels}/{train,val,test} + data.yaml.
Images without people are kept as negatives (they contain cars, roads and
buildings - useful hard negatives).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml

import config
from thermal_utils import IMG_EXTS, preprocess

SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def find_yolo_splits(src: Path):
    """-> {split: (images_dir, labels_dir)}"""
    found = {}
    for img_dir in src.rglob("images"):
        if not img_dir.is_dir():
            continue
        for sub in img_dir.iterdir():
            if sub.is_dir() and sub.name.lower() in SPLITS:
                lab = img_dir.parent / "labels" / sub.name
                if lab.is_dir():
                    found[SPLITS[sub.name.lower()]] = (sub, lab)
    return found


def find_coco_splits(src: Path):
    found = {}
    for js in src.rglob("*.json"):
        for key, split in SPLITS.items():
            if key in js.stem.lower() and split not in found:
                found[split] = js
    return found


def write(out, split, img_path, lines, mode, min_px=0.0):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    H, W = img.shape[:2]
    lines[:] = [l for l in lines if min(float(l.split()[3]) * W, float(l.split()[4]) * H) >= min_px]
    dst_img = out / "images" / split / (img_path.stem + ".png")
    dst_lab = out / "labels" / split / (img_path.stem + ".txt")
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lab.parent.mkdir(parents=True, exist_ok=True)
    # Same preprocessing as your own data, so both datasets look alike.
    cv2.imwrite(str(dst_img), preprocess(img, mode))
    dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="extracted HIT-UAV folder")
    ap.add_argument("--out", default=str(config.HITUAV_DIR))
    ap.add_argument("--person-id", type=int, default=0, help="person class id in the source labels")
    ap.add_argument("--min-px", type=float, default=3.0, help="drop person boxes smaller than this (either side)")
    ap.add_argument("--preprocess", default=config.PREPROCESS, choices=["none", "clahe"])
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    counts = {}

    yolo = find_yolo_splits(src)
    if yolo:
        for split, (img_dir, lab_dir) in yolo.items():
            n_img = n_box = 0
            for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
                lp = lab_dir / (img_path.stem + ".txt")
                lines = []
                if lp.exists():
                    for line in lp.read_text().splitlines():
                        parts = line.split()
                        if len(parts) >= 5 and int(float(parts[0])) == args.person_id:
                            lines.append("0 " + " ".join(parts[1:5]))
                n_img += write(out, split, img_path, lines, args.preprocess, args.min_px)
                n_box += len(lines)
            counts[split] = (n_img, n_box)
    else:
        coco = find_coco_splits(src)
        if not coco:
            raise SystemExit(f"no YOLO images/labels or COCO json found under {src}")
        for split, js in coco.items():
            d = json.loads(js.read_text())
            cats = {c["id"]: c["name"].lower() for c in d["categories"]}
            person_ids = {i for i, n in cats.items() if n in ("person", "people", "pedestrian")}
            anns = {}
            for a in d["annotations"]:
                if a["category_id"] in person_ids:
                    anns.setdefault(a["image_id"], []).append(a["bbox"])
            n_img = n_box = 0
            for im in d["images"]:
                cands = list(src.rglob(Path(im["file_name"]).name))
                if not cands:
                    continue
                W, H = im["width"], im["height"]
                lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                         for x, y, w, h in anns.get(im["id"], []) if min(w, h) >= args.min_px]
                n_img += write(out, split, cands[0], lines, args.preprocess)
                n_box += len(lines)
            counts[split] = (n_img, n_box)

    data = {"path": str(out.resolve()), "names": {0: "person"}}
    for split in ("train", "val", "test"):
        if split in counts:
            data[split] = f"images/{split}"
    if "val" not in data and "test" in data:
        data["val"] = data["test"]
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    for split, (n_img, n_box) in counts.items():
        print(f"{split:6} {n_img:5d} images {n_box:6d} person boxes")
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
