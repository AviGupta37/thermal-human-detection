"""Train / fine-tune YOLO on thermal person data with thermal-specific settings.

Recommended two-stage recipe for a 500-700 frame custom set:

  1) pre-train on HIT-UAV (thousands of aerial thermal people):
       python train.py --data dataset/hituav_person/data.yaml --name hituav_pre --epochs 100
  2) fine-tune on your own frames, starting from stage 1:
       python train.py --weights runs/hituav_pre/weights/best.pt --name person_ft

  One stage only (COCO-pretrained start):
       python train.py --name person

Tiny targets (people < ~15 px tall):  add --imgsz 960 (or 1280, batch 8)
Mixed data in one run instead of two stages: --extra-data dataset/hituav_person/data.yaml

Best weights end up in runs/<name>/weights/best.pt. Run evaluate.py next.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import config


def merged_yaml(main_yaml: Path, extra_yaml: Path, out: Path) -> Path:
    """Train on main+extra, validate on main only (your real conditions)."""
    def abs_list(d, key):
        base = Path(d.get("path", "."))
        v = d.get(key)
        if v is None:
            return []
        v = v if isinstance(v, list) else [v]
        return [str((base / p).resolve()) for p in v]

    a = yaml.safe_load(main_yaml.read_text())
    b = yaml.safe_load(extra_yaml.read_text())
    merged = {"train": abs_list(a, "train") + abs_list(b, "train"),
              "val": abs_list(a, "val"),
              "names": a["names"]}
    if a.get("test"):
        merged["test"] = abs_list(a, "test")
    out.write_text(yaml.safe_dump(merged, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_YAML))
    ap.add_argument("--extra-data", help="second data.yaml whose TRAIN images are added (e.g. HIT-UAV)")
    ap.add_argument("--weights", default=config.BASE_MODEL,
                    help="start weights: yolo26s.pt / yolo11s.pt / a previous best.pt")
    ap.add_argument("--name", default="person")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--imgsz", type=int, default=config.IMGSZ)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N layers (try 10 when fine-tuning on very few images)")
    ap.add_argument("--lr0", type=float, help="initial LR (default: Ultralytics auto; use ~0.002 for fine-tuning)")
    ap.add_argument("--nadir", action="store_true",
                    help="camera points straight down: allow vertical flips and 90-degree style rotations")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"{data} not found - run split_dataset.py (or prepare_hituav.py) first")
    if args.extra_data:
        data = merged_yaml(data, Path(args.extra_data), data.parent / "data_merged.yaml")
        print(f"training on merged set: {data}")

    model = YOLO(args.weights)
    hyp = dict(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str(config.RUNS_DIR), name=args.name, exist_ok=True,
        device=args.device, workers=args.workers, seed=args.seed, patience=args.patience,
        single_cls=len(config.CLASSES) == 1, cos_lr=True, freeze=args.freeze or None,
        resume=args.resume, plots=True, cache="ram",

        # --- thermal-specific augmentation ---
        # Thermal is grayscale: hue/saturation jitter is meaningless. Brightness
        # jitter mimics camera gain/AGC changes and day vs. night contrast.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.35,
        fliplr=0.5,
        flipud=0.5 if args.nadir else 0.0,
        degrees=15.0 if args.nadir else 3.0,
        translate=0.1,
        # Altitude changes person size; a wide scale range teaches size invariance.
        scale=0.5,
        # Mosaic multiplies the effective dataset and creates many small targets,
        # mixup adds hard negatives. Turned off for the last epochs so the model
        # finishes on realistic whole frames.
        mosaic=1.0, close_mosaic=15, mixup=0.1,
        erasing=0.0,
    )
    if args.lr0:
        hyp["lr0"] = args.lr0
    model.train(**hyp)

    best = config.RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}\nnext: python evaluate.py --weights {best}")


if __name__ == "__main__":
    main()
