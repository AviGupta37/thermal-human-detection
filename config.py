"""Shared settings for the thermal human-detection pipeline.

Every script imports from here so that frame extraction, annotation, training
and live detection all agree on classes, paths and preprocessing. If you change
PREPROCESS after annotating, re-extract the frames: the model must see exactly
the same kind of image in training and in the field.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- classes -----------------------------------------------------------------
# Keep a single class unless you really need more. With 500-700 frames, every
# extra class splits the data and lowers accuracy on "person".
CLASSES = ["person"]

# ---- dataset layout ----------------------------------------------------------
DATA_DIR = ROOT / "dataset"
RAW_IMAGES = DATA_DIR / "raw" / "images"   # frames to annotate (from extract_frames.py)
RAW_LABELS = DATA_DIR / "raw" / "labels"   # YOLO .txt written by annotate.py
SPLIT_DIR = DATA_DIR / "split"             # train/val/test written by split_dataset.py
DATA_YAML = SPLIT_DIR / "data.yaml"
HITUAV_DIR = DATA_DIR / "hituav_person"    # converted public dataset (prepare_hituav.py)
RUNS_DIR = ROOT / "runs"

# File-name separator between the source/clip tag and the frame number, e.g.
# "flight03__000120.png". split_dataset.py splits by the part before "__" so
# near-identical frames from one clip never end up in both train and val.
GROUP_SEP = "__"

# ---- preprocessing -----------------------------------------------------------
# "none"  : 8-bit frames are passed through (converted to 3-channel gray)
# "clahe" : local contrast boost; helps low-contrast scenes (hot ground, fog)
# 16-bit radiometric frames are always percentile-normalised to 8-bit first.
PREPROCESS = "none"
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
# Thermal polarity of the camera output. The model is trained on whatever you
# record; set True only if the camera outputs black-hot and you want white-hot.
INVERT = False

# ---- training defaults -------------------------------------------------------
BASE_MODEL = "yolo26s.pt"   # s is the sweet spot for ~600 images on a 2080 Ti
IMGSZ = 640                 # use 960-1280 when people are < ~15 px tall
EPOCHS = 150
BATCH = 16

# ---- detection defaults ------------------------------------------------------
CONF = 0.30
IOU = 0.50
