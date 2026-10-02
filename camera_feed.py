"""Frame sources for the thermal pipeline: USB camera, RTSP/UDP drone downlink,
GStreamer pipeline, video file, image folder or a synthetic test feed.

Library use:
    with FrameSource("rtsp://192.168.1.10:8554/live") as src:
        ok, frame = src.read()

CLI (preview and optionally record raw footage for later annotation):
    python camera_feed.py --source 0
    python camera_feed.py --source rtsp://192.168.1.10:8554/live --record flight03.mp4
    python camera_feed.py --source synthetic
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from thermal_utils import list_images

STREAM_PREFIXES = ("rtsp://", "rtsps://", "udp://", "tcp://", "rtp://", "http://", "https://", "srt://")


class SyntheticThermal:
    """Fake aerial thermal feed: cool textured ground, warm static clutter
    (rocks, a vehicle) and N walking people. `ground_truth` holds the true
    person boxes for the current frame so detectors can be checked."""

    def __init__(self, width=640, height=512, people=6, person_px=18, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.w, self.h, self.ppx = width, height, person_px
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        bg = 70 + 25 * (xx / width) + 15 * np.sin(yy / 37.0)
        bg += cv2.GaussianBlur(rng.normal(0, 18, (height, width)).astype(np.float32), (0, 0), 6)
        for _ in range(8):  # warm rocks
            cx, cy = rng.integers(0, width), rng.integers(0, height)
            cv2.circle(bg, (int(cx), int(cy)), int(rng.integers(4, 9)), float(rng.integers(115, 140)), -1)
        cv2.rectangle(bg, (int(width * .7), int(height * .2)), (int(width * .7) + 60, int(height * .2) + 28), 150, -1)
        self.bg = cv2.GaussianBlur(bg, (0, 0), 1.2)
        self.pos = rng.uniform([20, 20], [width - 20, height - 20], (people, 2))
        self.vel = rng.normal(0, 1.2, (people, 2))
        self.ground_truth = np.zeros((0, 4), np.float32)

    def read(self):
        self.vel = np.clip(self.vel + self.rng.normal(0, 0.15, self.vel.shape), -2, 2)
        self.pos += self.vel
        for k in range(2):
            lim = (self.w, self.h)[k] - 15
            bounce = (self.pos[:, k] < 15) | (self.pos[:, k] > lim)
            self.vel[bounce, k] *= -1
            self.pos[:, k] = np.clip(self.pos[:, k], 15, lim)
        img = self.bg.copy()
        hh, ww = self.ppx, max(4, int(self.ppx * 0.45))
        gt = []
        for x, y in self.pos:
            cv2.ellipse(img, (int(x), int(y)), (ww // 2, hh // 2), 0, 0, 360, 200, -1)
            cv2.circle(img, (int(x), int(y - hh * 0.42)), max(2, ww // 3), 215, -1)
            gt.append([x - ww / 2, y - hh / 2 - ww // 3, x + ww / 2, y + hh / 2])
        img = cv2.GaussianBlur(img, (0, 0), 1.0) + self.rng.normal(0, 3, img.shape)
        self.ground_truth = np.array(gt, np.float32)
        return True, np.clip(img, 0, 255).astype(np.uint8)

    def release(self):
        pass


class _ImageFolder:
    def __init__(self, folder, loop):
        self.files = list_images(folder)
        if not self.files:
            raise FileNotFoundError(f"no images in {folder}")
        self.i, self.loop = 0, loop

    def read(self):
        if self.i >= len(self.files):
            if not self.loop:
                return False, None
            self.i = 0
        frame = cv2.imread(str(self.files[self.i]), cv2.IMREAD_UNCHANGED)
        self.i += 1
        return frame is not None, frame

    def release(self):
        pass


class FrameSource:
    """Uniform `read() -> (ok, frame)` over every kind of source.

    Network streams are read on a background thread that keeps only the
    newest frame, so a slow detector never builds up seconds of latency, and
    the stream is reopened automatically if the downlink drops.
    """

    def __init__(self, source: str | int, width: int | None = None, height: int | None = None,
                 fps: float | None = None, loop: bool = False, rtsp_tcp: bool = False,
                 synthetic_people: int = 6, synthetic_person_px: int = 18):
        self.source = str(source)
        self.is_stream = self.source.lower().startswith(STREAM_PREFIXES) or self.source.startswith("gst:")
        self.loop = loop
        self.synthetic = None
        self._lock = threading.Lock()
        self._latest = None
        self._stop = False
        self._thread = None
        self._width, self._height, self._fps = width, height, fps
        self._rtsp_tcp = rtsp_tcp
        self.frame_index = 0

        if self.source == "synthetic":
            self.synthetic = SyntheticThermal(width or 640, height or 512, synthetic_people, synthetic_person_px)
            self.cap = self.synthetic
        elif Path(self.source).is_dir():
            self.cap = _ImageFolder(self.source, loop)
        else:
            self.cap = self._open()
            if self.is_stream:
                self._thread = threading.Thread(target=self._reader, daemon=True)
                self._thread.start()

    # -- opening -------------------------------------------------------------
    def _open(self):
        src = self.source
        if src.isdigit():
            idx = int(src)
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if os.name == "posix" else cv2.VideoCapture(idx)
            # Many thermal cores (Boson, Lepton breakout, TOPDON) expose raw Y16
            # if you ask; plain YUYV is fine too. Ask for the native size.
            if self._width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if self._fps:
                cap.set(cv2.CAP_PROP_FPS, self._fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif src.startswith("gst:"):
            # e.g. gst:udpsrc port=5600 caps="application/x-rtp" ! rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=1
            cap = cv2.VideoCapture(src[4:], cv2.CAP_GSTREAMER)
        elif self.is_stream:
            opts = ["fflags;nobuffer", "flags;low_delay", "max_delay;0"]
            if src.lower().startswith("rtsp"):
                opts.insert(0, "rtsp_transport;" + ("tcp" if self._rtsp_tcp else "udp"))
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            if not Path(src).exists():
                raise FileNotFoundError(src)
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"could not open source {src!r}")
        return cap

    def _reader(self):
        fails = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                fails = 0
                with self._lock:
                    self._latest = frame
                continue
            fails += 1
            if fails > 50:
                print(f"[camera_feed] stream lost, reconnecting to {self.source} ...")
                self.cap.release()
                time.sleep(1.0)
                try:
                    self.cap = self._open()
                except RuntimeError as e:
                    print(f"[camera_feed] {e}")
                fails = 0
            else:
                time.sleep(0.01)

    # -- reading -------------------------------------------------------------
    def read(self, timeout: float = 5.0):
        if self._thread is not None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                with self._lock:
                    frame, self._latest = self._latest, None
                if frame is not None:
                    self.frame_index += 1
                    return True, frame
                time.sleep(0.002)
            return False, None
        ok, frame = self.cap.read()
        if not ok and self.loop and isinstance(self.cap, cv2.VideoCapture):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok:
            self.frame_index += 1
        return ok, frame

    @property
    def fps(self) -> float:
        if isinstance(self.cap, cv2.VideoCapture):
            f = self.cap.get(cv2.CAP_PROP_FPS)
            if 1 <= f <= 240:
                return f
        return 30.0

    @property
    def frame_count(self) -> int:
        if isinstance(self.cap, cv2.VideoCapture) and not self.is_stream:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if isinstance(self.cap, _ImageFolder):
            return len(self.cap.files)
        return -1

    def release(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def main():
    ap = argparse.ArgumentParser(description="Preview / record a thermal source")
    ap.add_argument("--source", default="0", help="0/1 (USB), rtsp://..., udp://..., gst:<pipeline>, file, folder, synthetic")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--rtsp-tcp", action="store_true", help="RTSP over TCP (more robust on lossy links, more latency)")
    ap.add_argument("--record", help="write the raw feed to this video file (lossless FFV1 if .mkv)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    writer = None
    n, t0 = 0, time.time()
    with FrameSource(args.source, args.width, args.height, args.fps, rtsp_tcp=args.rtsp_tcp) as src:
        print(f"opened {args.source}; press q to quit" + (f", recording to {args.record}" if args.record else ""))
        while True:
            ok, frame = src.read()
            if not ok:
                break
            view = frame
            if view.dtype != np.uint8 or view.ndim == 2:
                from thermal_utils import preprocess
                view = preprocess(frame, mode="none")
            if args.record:
                if writer is None:
                    h, w = view.shape[:2]
                    # FFV1 in .mkv is lossless: compression artefacts on tiny
                    # warm blobs are exactly what you do not want in training data.
                    fourcc = cv2.VideoWriter_fourcc(*("FFV1" if args.record.endswith(".mkv") else "mp4v"))
                    writer = cv2.VideoWriter(args.record, fourcc, src.fps, (w, h))
                writer.write(view)
            n += 1
            if not args.no_show:
                fps = n / max(time.time() - t0, 1e-6)
                disp = view.copy()
                cv2.putText(disp, f"{fps:5.1f} fps  {disp.shape[1]}x{disp.shape[0]}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("camera_feed", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"{n} frames")


if __name__ == "__main__":
    main()
