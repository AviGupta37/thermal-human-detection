"""Real-time thermal human detection + tracking.

    python detect.py --source synthetic --detector hotspot                    # no model needed
    python detect.py --source 0 --weights runs/person_ft/weights/best.pt      # USB camera
    python detect.py --source rtsp://192.168.144.25:8554/main --weights best.pt --tile 2
    python detect.py --source flight03.mkv --weights best.pt --save out.mp4 --csv out.csv --no-show

Detectors
  yolo     trained model + ByteTrack (accurate, needs weights)
  hotspot  white top-hat warm-blob filter + ByteTrack (no training; tune --person-px/--min-area)

Accuracy options
  --tile N         also run the model on an NxN grid of overlapping crops (small/far people)
  --min-hits K     show a track only after K detections (removes one-frame flickers)
  --operating X    threshold from evaluate.py's thresholds.json: best_f1 or high_recall

Keys: q/Esc quit, space pause, s snapshot.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import config
from camera_feed import FrameSource
from thermal_utils import HotspotDetector, iou_matrix, nms, preprocess, trained_imgsz


# ---- detectors ---------------------------------------------------------------

class YoloDetector:
    def __init__(self, weights, conf, iou, imgsz, device, half, tile, tile_overlap):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.half = device, half
        self.tile, self.overlap = tile, tile_overlap

    def _tiles(self, h, w):
        n = self.tile
        tw, th = int(np.ceil(w / (n - (n - 1) * self.overlap))), int(np.ceil(h / (n - (n - 1) * self.overlap)))
        xs = np.linspace(0, w - tw, n).astype(int)
        ys = np.linspace(0, h - th, n).astype(int)
        return [(x, y, tw, th) for y in ys for x in xs]

    def __call__(self, img):
        h, w = img.shape[:2]
        crops, offsets = [img], [(0, 0)]
        if self.tile and self.tile > 1:
            for x, y, tw, th in self._tiles(h, w):
                crops.append(img[y:y + th, x:x + tw])
                offsets.append((x, y))
        results = self.model.predict(crops, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                     device=self.device, half=self.half, verbose=False)
        boxes, scores = [], []
        for r, (ox, oy) in zip(results, offsets):
            b = r.boxes.xyxy.cpu().numpy()
            if len(b):
                boxes.append(b + [ox, oy, ox, oy])
                scores.append(r.boxes.conf.cpu().numpy())
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        if len(crops) > 1:
            keep = nms(boxes, scores, self.iou)
            boxes, scores = boxes[keep], scores[keep]
        return boxes.astype(np.float32), scores.astype(np.float32)


# ---- tracker -------------------------------------------------------------------

class Tracker:
    """ByteTrack (from Ultralytics) on top of any detector, plus a min-hits
    gate so a person is only reported after being seen in K frames."""

    def __init__(self, fps, min_hits=3, track_buffer_s=2.0, high=0.25, low=0.1):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
        args = IterableSimpleNamespace(
            tracker_type="bytetrack", track_high_thresh=high, track_low_thresh=low,
            new_track_thresh=high, track_buffer=int(track_buffer_s * fps), match_thresh=0.8,
            fuse_score=True)
        try:  # ultralytics < 8.4 scales track_buffer by frame_rate/30 itself
            self.bt = BYTETracker(args, frame_rate=30)
        except TypeError:
            self.bt = BYTETracker(args)
        self.hits: dict[int, int] = {}
        self.min_hits = min_hits

    def update(self, boxes, scores, shape):
        from ultralytics.engine.results import Boxes
        det = np.zeros((len(boxes), 6), np.float32)
        if len(boxes):
            det[:, :4], det[:, 4] = boxes, scores
        tracks = self.bt.update(Boxes(det, shape[:2]), None)
        out = []
        for t in tracks:  # x1 y1 x2 y2 id score cls idx
            tid = int(t[4])
            self.hits[tid] = self.hits.get(tid, 0) + 1
            if self.hits[tid] >= self.min_hits:
                out.append((tid, t[:4], float(t[5])))
        return out


# ---- drawing -----------------------------------------------------------------

def draw(img, items, fps, frame_no, tracked):
    for item in items:
        if tracked:
            tid, b, s = item
            label = f"#{tid} {s:.2f}"
            col = tuple(int(c) for c in np.array([(tid * 67) % 255, 255, (tid * 151) % 255]))
        else:
            b, s = item
            label, col = f"{s:.2f}", (0, 255, 0)
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1 if (x2 - x1) < 20 else 2)
        cv2.putText(img, label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    n = len(items)
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, f"frame {frame_no}  {fps:5.1f} fps  persons: {n}", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if n:
        cv2.putText(img, "PERSON DETECTED", (img.shape[1] - 150, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic", help="synthetic, 0/1 (USB), rtsp://, udp://, gst:..., file or folder")
    ap.add_argument("--detector", choices=["yolo", "hotspot"], default=None,
                    help="default: yolo if --weights is given, else hotspot")
    ap.add_argument("--weights", help="trained YOLO weights (best.pt / .engine / .onnx)")
    ap.add_argument("--conf", type=float, help="confidence threshold (default: thresholds.json or config)")
    ap.add_argument("--operating", choices=["best_f1", "high_recall"], default="best_f1")
    ap.add_argument("--iou", type=float, default=config.IOU)
    ap.add_argument("--imgsz", type=int, help="default: training size")
    ap.add_argument("--tile", type=int, default=0, help="extra NxN tiled pass (2 or 3) for small targets")
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    ap.add_argument("--half", action="store_true", help="FP16 on GPU")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--preprocess", default=config.PREPROCESS, choices=["none", "clahe"])
    ap.add_argument("--invert", action="store_true", help="camera is black-hot but the model was trained white-hot")
    # hotspot
    ap.add_argument("--person-px", type=int, default=30, help="hotspot: expected person height in px")
    ap.add_argument("--min-area", type=int, default=12)
    ap.add_argument("--max-area", type=int, default=5000)
    ap.add_argument("--hot-thresh", type=int, default=0, help="hotspot: top-hat threshold, 0 = automatic")
    # source
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--rtsp-tcp", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--synthetic-people", type=int, default=6)
    # output
    ap.add_argument("--save", help="annotated output video")
    ap.add_argument("--csv", help="per-detection log")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    detector_name = args.detector or ("yolo" if args.weights else "hotspot")
    if detector_name == "yolo":
        if not args.weights:
            raise SystemExit("--detector yolo needs --weights")
        conf, imgsz = args.conf, args.imgsz
        tfile = Path(args.weights).with_name("thresholds.json")
        if tfile.exists():
            t = json.loads(tfile.read_text())
            conf = conf if conf is not None else t[args.operating]
            imgsz = imgsz or t.get("imgsz")
            print(f"using {args.operating} threshold {conf:.2f} from {tfile}")
        conf = conf if conf is not None else config.CONF
        det = YoloDetector(args.weights, conf, args.iou, imgsz, args.device, args.half, args.tile, args.tile_overlap)
        if det.imgsz is None:
            det.imgsz = trained_imgsz(det.model, config.IMGSZ)
        track_high = conf
    else:
        det = HotspotDetector(args.person_px, args.min_area, args.max_area, args.hot_thresh)
        track_high = args.conf if args.conf is not None else 0.25

    src = FrameSource(args.source, args.width, args.height, loop=args.loop, rtsp_tcp=args.rtsp_tcp,
                      synthetic_people=args.synthetic_people)
    tracker = None if args.no_track else Tracker(src.fps, args.min_hits, high=track_high,
                                                 low=max(0.05, track_high * 0.4))
    writer, csv_f, csv_w = None, None, None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2", "conf"])

    n, t_start, fps_ema, paused = 0, time.time(), 0.0, False
    gt_stats = [0, 0, 0]  # tp, fp, fn on synthetic ground truth
    try:
        while True:
            if not paused:
                ok, frame = src.read()
                if not ok:
                    break
                t0 = time.time()
                img = preprocess(frame, args.preprocess, args.invert)
                boxes, scores = det(img)
                if tracker is not None:
                    items = tracker.update(boxes, scores, img.shape)
                    out_boxes = np.array([b for _, b, _ in items], np.float32).reshape(-1, 4)
                else:
                    items = list(zip(boxes, scores))
                    out_boxes = boxes
                dt = time.time() - t0
                fps_ema = 1 / max(dt, 1e-6) if n == 0 else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
                n += 1

                if src.synthetic is not None and n > 5:  # after tracks have warmed up
                    gt = src.synthetic.ground_truth
                    hit = iou_matrix(gt, out_boxes).max(axis=1) >= 0.2 if len(out_boxes) else np.zeros(len(gt), bool)
                    matched_det = iou_matrix(out_boxes, gt).max(axis=1) >= 0.2 if len(gt) else np.zeros(len(out_boxes), bool)
                    gt_stats[0] += int(hit.sum())
                    gt_stats[2] += int((~hit).sum())
                    gt_stats[1] += int((~matched_det).sum())

                if csv_w is not None:
                    ts = round(time.time() - t_start, 3)
                    for it in items:
                        tid, b, s = it if tracker is not None else (-1, it[0], it[1])
                        csv_w.writerow([n, ts, tid, *[round(float(v), 1) for v in b], round(float(s), 3)])

                vis = draw(img.copy(), items, fps_ema, n, tracker is not None)
                if args.save:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src.fps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if args.max_frames and n >= args.max_frames:
                    break
            if not args.no_show:
                cv2.imshow("thermal detect", vis)
                k = cv2.waitKey(1 if not paused else 30) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord(" "):
                    paused = not paused
                if k == ord("s"):
                    name = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(name, img)  # clean frame: ready to add to the dataset
                    print(f"saved {name}")
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if csv_f is not None:
            csv_f.close()
        cv2.destroyAllWindows()

    wall = time.time() - t_start
    print(f"{n} frames, detector {fps_ema:.1f} fps (pipeline {n / max(wall, 1e-6):.1f} fps)")
    if src.synthetic is not None and sum(gt_stats):
        tp, fp, fn = gt_stats
        print(f"synthetic ground truth: precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}")


if __name__ == "__main__":
    main()
