"""Evaluate a trained model on the held-out test split and pick the
confidence threshold to use in the field.

    python evaluate.py --weights runs/person/weights/best.pt
    python evaluate.py --weights best.pt --split val --save-errors

Prints mAP, then a precision / recall / F1 table over confidence thresholds,
recall by person size (tiny / small / medium+), and writes thresholds.json next
to the weights. detect.py reads that file automatically.

For search-and-rescue / perimeter use a missed person usually costs more than
a false alarm, so the 'high_recall' threshold (recall >= --target-recall) is
often the right one; 'best_f1' is the balanced choice.
With --save-errors, images with misses (red) and false alarms (orange) are
written to runs/<name>/errors/ - look at them: they tell you what to annotate next.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

import config
from thermal_utils import iou_matrix, list_images, read_yolo_labels, trained_imgsz

SIZE_BINS = [("tiny <16px", 0, 16), ("small 16-32px", 16, 32), ("medium+ >32px", 32, 1e9)]


def split_images(data_yaml: Path, split: str) -> list[Path]:
    d = yaml.safe_load(data_yaml.read_text())
    if split not in d:
        raise SystemExit(f"split {split!r} not in {data_yaml}")
    base = Path(d.get("path", data_yaml.parent))
    entries = d[split] if isinstance(d[split], list) else [d[split]]
    imgs = []
    for e in entries:
        imgs += list_images(base / e)
    return imgs


def labels_for(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(config.DATA_YAML))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at")
    ap.add_argument("--iou-match", type=float, default=0.3,
                    help="IoU to count a detection as correct (0.3 is fair for boxes a few pixels big)")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--save-errors", action="store_true")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    imgsz = args.imgsz or trained_imgsz(model, config.IMGSZ)

    # 1) standard COCO-style metrics
    m = model.val(data=args.data, split=args.split, imgsz=imgsz, device=args.device, plots=False, verbose=False)
    print(f"\n{args.split}: mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  "
          f"P {m.box.mp:.3f}  R {m.box.mr:.3f}")

    # 2) own matching over all thresholds -> operating point
    imgs = split_images(Path(args.data), args.split)
    all_scores, all_tp, n_gt = [], [], 0
    gt_heights, gt_hit_at = [], []  # per GT box: height, best conf of a matching detection
    err_dir = Path(args.weights).resolve().parent.parent / "errors"
    per_image = []
    for res in model.predict([str(p) for p in imgs], imgsz=imgsz, conf=0.01, iou=config.IOU,
                             device=args.device, verbose=False, stream=True):
        h, w = res.orig_shape
        gt = np.array([b[1:] for b in read_yolo_labels(labels_for(Path(res.path)), w, h)], np.float32).reshape(-1, 4)
        det = res.boxes.xyxy.cpu().numpy()
        sc = res.boxes.conf.cpu().numpy()
        order = np.argsort(-sc)
        det, sc = det[order], sc[order]
        ious = iou_matrix(det, gt)
        matched = np.zeros(len(gt), bool)
        hit_conf = np.zeros(len(gt), np.float32)
        tp = np.zeros(len(det), bool)
        for i in range(len(det)):  # greedy by confidence
            if len(gt) == 0:
                break
            cand = np.where(~matched & (ious[i] >= args.iou_match))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                matched[j], tp[i], hit_conf[j] = True, True, sc[i]
        all_scores.append(sc)
        all_tp.append(tp)
        n_gt += len(gt)
        gt_heights += list(gt[:, 3] - gt[:, 1])
        gt_hit_at += list(hit_conf)
        per_image.append((res.path, gt, det, sc, tp, hit_conf))

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tps = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        keep = scores >= t
        tp = int(tps[keep].sum())
        fp = int(keep.sum() - tp)
        p = tp / max(tp + fp, 1)
        r = tp / max(n_gt, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        rows.append((t, p, r, f1, fp))
    print(f"\n{'conf':>5} {'prec':>6} {'recall':>6} {'F1':>6} {'FP/img':>7}")
    for t, p, r, f1, fp in rows:
        print(f"{t:5.2f} {p:6.3f} {r:6.3f} {f1:6.3f} {fp / max(len(imgs), 1):7.2f}")
    best = max(rows, key=lambda r: r[3])
    hi = [r for r in rows if r[2] >= args.target_recall]
    high_recall = max(hi, key=lambda r: r[0]) if hi else rows[0]
    if high_recall[0] > best[0]:  # best-F1 already meets the recall target
        high_recall = best

    gh, gc = np.array(gt_heights), np.array(gt_hit_at)
    print(f"\nrecall by person height at conf {best[0]:.2f}:")
    for name, lo, hi_ in SIZE_BINS:
        sel = (gh >= lo) & (gh < hi_)
        if sel.any():
            print(f"  {name:15} {np.mean(gc[sel] >= best[0]):.3f}  (n={sel.sum()})")

    out = {"best_f1": float(best[0]), "high_recall": float(high_recall[0]),
           "target_recall": args.target_recall, "imgsz": imgsz,
           "f1": float(best[3]), "precision": float(best[1]), "recall": float(best[2]),
           "map50": float(m.box.map50), "split": args.split}
    tpath = Path(args.weights).with_name("thresholds.json")
    tpath.write_text(json.dumps(out, indent=2))
    print(f"\nbest-F1 conf {best[0]:.2f} (P {best[1]:.3f} R {best[2]:.3f}); "
          f"recall>={args.target_recall} conf {high_recall[0]:.2f} (P {high_recall[1]:.3f} R {high_recall[2]:.3f})")
    print(f"wrote {tpath}")

    if args.save_errors:
        err_dir.mkdir(parents=True, exist_ok=True)
        t = best[0]
        n = 0
        for path, gt, det, sc, tp, hit_conf in per_image:
            misses = gt[hit_conf < t]
            fps = det[(sc >= t) & ~tp]
            if len(misses) == 0 and len(fps) == 0:
                continue
            img = cv2.imread(path)
            for b in det[(sc >= t) & tp]:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 0), 1)
            for b in misses:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 2)
            for b, s in zip(fps, sc[(sc >= t) & ~tp]):
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 140, 255), 2)
                cv2.putText(img, f"{s:.2f}", (int(b[0]), int(b[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
            cv2.imwrite(str(err_dir / Path(path).name), img)
            n += 1
        print(f"{n} error images in {err_dir}  (green=correct, red=missed, orange=false alarm)")


if __name__ == "__main__":
    main()
