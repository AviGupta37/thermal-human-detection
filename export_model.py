"""Export trained weights for faster field inference.

    python export_model.py --weights runs/person_ft/weights/best.pt --format engine   # TensorRT (Jetson / NVIDIA)
    python export_model.py --weights best.pt --format onnx                             # CPU / other runtimes

Build the TensorRT engine ON the target device (an engine built on the 2080 Ti
will not run on a Jetson). thresholds.json is copied next to the export so
detect.py keeps the calibrated threshold:
    python detect.py --weights runs/person_ft/weights/best.engine --source 0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config
from thermal_utils import trained_imgsz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="engine", choices=["engine", "onnx", "openvino", "tflite", "ncnn"])
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--half", action="store_true", help="FP16 (recommended for TensorRT)")
    ap.add_argument("--int8", action="store_true", help="INT8 calibration on the val split (Jetson Nano/Orin)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    imgsz = args.imgsz or trained_imgsz(model, config.IMGSZ)
    kw = dict(format=args.format, imgsz=imgsz, half=args.half, device=args.device)
    if args.int8:
        kw.update(int8=True, data=str(config.DATA_YAML))
    out = Path(model.export(**kw))
    t = Path(args.weights).with_name("thresholds.json")
    if t.exists() and out.parent != t.parent:
        shutil.copy2(t, out.with_name("thresholds.json"))
    print(f"exported {out}")


if __name__ == "__main__":
    main()
