"""Manual bounding-box annotation tool for thermal frames (YOLO format).

    python annotate.py                                  # dataset/raw/images -> dataset/raw/labels
    python annotate.py --model runs/person/best.pt      # pre-label with a model, then correct
    python annotate.py --images other/frames --labels other/labels
    python annotate.py --list dataset/raw/priority.txt   # frames chosen by select_hard_frames.py

Mouse
    left-drag on empty space ... draw a new box
    left-click a box ........... select it; drag inside to move, drag a corner to resize
    right-click a box .......... delete it
    wheel ...................... zoom in/out at the cursor (essential for tiny people)
    middle-drag ................ pan while zoomed

Keys
    d / Right / Space .. save and next image       a / Left .. save and previous
    n .................. skip (next, discard edits) f ......... jump to next unlabelled
    g .................. go to image number
    x / Delete ......... delete selected box       u ......... undo
    c .................. clear all boxes           e ......... mark as EMPTY (no people) and next
    m .................. re-run model pre-labels    1-9 ....... set class of new / selected box
    h .................. hide/show boxes           i ......... invert display (white/black-hot)
    k .................. cycle display contrast (normal / CLAHE / gamma) - display only
    r .................. reset zoom                 q / Esc ... save and quit

Labels are written when you leave an image. An image with no .txt is
"not labelled yet"; an empty .txt means "checked, nobody here" and is used as
a negative example, which cuts false alarms on hot rocks, engines and animals.

Annotation rules that matter for accuracy:
  * Box every person you can see, including partial, sitting, lying and
    half-hidden ones. A missed person is taught as background.
  * Tight boxes around the whole warm body, including the cooler legs.
  * If you truly cannot tell whether a blob is a person, skip the image (n)
    rather than guess - inconsistent labels hurt more than fewer labels.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

import config
from thermal_utils import label_path_for, list_images, read_yolo_labels, write_yolo_labels

WIN = "annotate"
COLORS = [(0, 255, 0), (0, 200, 255), (255, 120, 0), (255, 0, 255), (0, 255, 255),
          (120, 120, 255), (255, 255, 0), (180, 255, 120), (200, 120, 255)]
HANDLE = 7  # corner grab radius in screen pixels
KEY_LEFT, KEY_RIGHT = {65361, 2424832}, {65363, 2555904}
KEY_DELETE = {65535, 3014656}


class Annotator:
    def __init__(self, images, labels_dir, model=None, conf=0.25, scale=None):
        self.images = images
        self.labels_dir = Path(labels_dir)
        self.model = model
        self.conf = conf
        self.idx = 0
        self.cls = 0
        self.boxes: list[list[float]] = []  # [cls, x1, y1, x2, y2] image px
        self.history: list[list[list[float]]] = []
        self.sel = -1
        self.dirty = False
        self.hide = False
        self.invert = False
        self.contrast = 0
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self.drag = None   # ("new"|"move"|"corner"|"pan", data...)
        self.mouse = (0, 0)
        self.fixed_scale = scale
        self.clahe = cv2.createCLAHE(3.0, (8, 8))
        self.img = None

    # ---- coordinates -----------------------------------------------------------
    def s2i(self, x, y):
        return self.ox + x / (self.base * self.zoom), self.oy + y / (self.base * self.zoom)

    def i2s(self, x, y):
        return (x - self.ox) * self.base * self.zoom, (y - self.oy) * self.base * self.zoom

    def clamp_view(self):
        vw = self.w / self.zoom
        vh = self.h / self.zoom
        self.ox = float(np.clip(self.ox, 0, self.w - vw))
        self.oy = float(np.clip(self.oy, 0, self.h - vh))

    # ---- image I/O -------------------------------------------------------------
    def load(self, i):
        self.idx = i % len(self.images)
        path = self.images[self.idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cannot read {path}")
        from thermal_utils import to_gray8
        self.gray = to_gray8(img)
        self.h, self.w = self.gray.shape
        if self.fixed_scale:
            self.base = self.fixed_scale
        else:  # fit to ~1400x900 screen, at least 1x
            self.base = max(1.0, min(1400 / self.w, 900 / self.h))
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        lp = label_path_for(path, self.labels_dir)
        self.labelled = lp.exists()
        self.boxes = read_yolo_labels(lp, self.w, self.h)
        self.history, self.sel, self.dirty = [], -1, False
        self.done = sum(label_path_for(p, self.labels_dir).exists() for p in self.images)
        if not self.labelled and self.model is not None:
            self.prelabel()

    def save(self):
        if not self.dirty:
            return
        path = self.images[self.idx]
        write_yolo_labels(label_path_for(path, self.labels_dir), self.boxes, self.w, self.h)
        self.labelled, self.dirty = True, False

    def prelabel(self):
        res = self.model.predict(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR), conf=self.conf, verbose=False)[0]
        self.push()
        self.boxes = [[int(c), *map(float, b)] for b, c in
                      zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())]
        self.dirty = True

    def push(self):
        self.history.append(copy.deepcopy(self.boxes))
        self.history = self.history[-50:]

    # ---- hit testing -------------------------------------------------------------
    def box_at(self, ix, iy):
        best, best_area = -1, 1e18
        for k, (_, x1, y1, x2, y2) in enumerate(self.boxes):
            pad = 2 / (self.base * self.zoom)
            if x1 - pad <= ix <= x2 + pad and y1 - pad <= iy <= y2 + pad:
                a = (x2 - x1) * (y2 - y1)
                if a < best_area:
                    best, best_area = k, a
        return best

    def corner_at(self, sx, sy):
        if self.sel < 0:
            return None
        _, x1, y1, x2, y2 = self.boxes[self.sel]
        for cx, cy, name in ((x1, y1, "tl"), (x2, y1, "tr"), (x1, y2, "bl"), (x2, y2, "br")):
            px, py = self.i2s(cx, cy)
            if abs(px - sx) <= HANDLE and abs(py - sy) <= HANDLE:
                return name
        return None

    # ---- mouse -------------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _):
        self.mouse = (x, y)
        ix, iy = self.s2i(x, y)
        ix, iy = float(np.clip(ix, 0, self.w)), float(np.clip(iy, 0, self.h))
        if event == cv2.EVENT_MOUSEWHEEL or event == cv2.EVENT_MOUSEHWHEEL:
            factor = 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8
            new_zoom = float(np.clip(self.zoom * factor, 1.0, 16.0))
            # keep the image point under the cursor fixed
            self.ox = ix - x / (self.base * new_zoom)
            self.oy = iy - y / (self.base * new_zoom)
            self.zoom = new_zoom
            self.clamp_view()
        elif event == cv2.EVENT_LBUTTONDOWN:
            corner = self.corner_at(x, y)
            if corner:
                self.push()
                self.drag = ("corner", corner)
                return
            k = self.box_at(ix, iy)
            if k >= 0:
                self.sel = k
                self.push()
                self.drag = ("move", ix, iy, list(self.boxes[k]))
            else:
                self.sel = -1
                self.drag = ("new", ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            kind = self.drag[0]
            if kind == "corner":
                b = self.boxes[self.sel]
                if "l" in self.drag[1]:
                    b[1] = ix
                else:
                    b[3] = ix
                if "t" in self.drag[1]:
                    b[2] = iy
                else:
                    b[4] = iy
                self.dirty = True
            elif kind == "move":
                _, sx, sy, orig = self.drag
                dx, dy = ix - sx, iy - sy
                dx = float(np.clip(dx, -orig[1], self.w - orig[3]))
                dy = float(np.clip(dy, -orig[2], self.h - orig[4]))
                self.boxes[self.sel] = [orig[0], orig[1] + dx, orig[2] + dy, orig[3] + dx, orig[4] + dy]
                self.dirty = True
            elif kind == "pan":
                _, sx, sy, ox, oy = self.drag
                self.ox = ox - (x - sx) / (self.base * self.zoom)
                self.oy = oy - (y - sy) / (self.base * self.zoom)
                self.clamp_view()
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            kind = self.drag[0]
            if kind == "new":
                _, sx, sy = self.drag
                x1, x2 = sorted((sx, ix))
                y1, y2 = sorted((sy, iy))
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    self.push()
                    self.boxes.append([self.cls, x1, y1, x2, y2])
                    self.sel = len(self.boxes) - 1
                    self.dirty = True
            elif kind == "corner":
                b = self.boxes[self.sel]
                b[1], b[3] = sorted((b[1], b[3]))
                b[2], b[4] = sorted((b[2], b[4]))
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            k = self.box_at(ix, iy)
            if k >= 0:
                self.push()
                self.boxes.pop(k)
                self.sel = -1
                self.dirty = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.drag = ("pan", x, y, self.ox, self.oy)
        elif event == cv2.EVENT_MBUTTONUP:
            self.drag = None

    # ---- drawing -----------------------------------------------------------------
    def render(self):
        g = self.gray
        if self.contrast == 1:
            g = self.clahe.apply(g)
        elif self.contrast == 2:
            g = (255 * (g / 255.0) ** 0.6).astype(np.uint8)
        if self.invert:
            g = 255 - g
        vw, vh = self.w / self.zoom, self.h / self.zoom
        x0, y0 = int(self.ox), int(self.oy)
        x1, y1 = min(self.w, int(np.ceil(self.ox + vw))), min(self.h, int(np.ceil(self.oy + vh)))
        crop = g[y0:y1, x0:x1]
        s = self.base * self.zoom
        out_w, out_h = int(round(self.w * self.base)), int(round(self.h * self.base))
        # nearest-neighbour so individual thermal pixels stay visible when zoomed
        M = np.float32([[s, 0, -(self.ox - x0) * s], [0, s, -(self.oy - y0) * s]])
        view = cv2.warpAffine(crop, M, (out_w, out_h), flags=cv2.INTER_NEAREST)
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

        if not self.hide:
            for k, (c, bx1, by1, bx2, by2) in enumerate(self.boxes):
                p1 = tuple(int(v) for v in self.i2s(bx1, by1))
                p2 = tuple(int(v) for v in self.i2s(bx2, by2))
                col = COLORS[int(c) % len(COLORS)]
                cv2.rectangle(view, p1, p2, col, 2 if k == self.sel else 1)
                if k == self.sel:
                    for px, py in (p1, p2, (p1[0], p2[1]), (p2[0], p1[1])):
                        cv2.rectangle(view, (px - 3, py - 3), (px + 3, py + 3), col, -1)
                if len(config.CLASSES) > 1:
                    cv2.putText(view, config.CLASSES[int(c)] if c < len(config.CLASSES) else str(c),
                                (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if self.drag and self.drag[0] == "new":
            p1 = tuple(int(v) for v in self.i2s(self.drag[1], self.drag[2]))
            cv2.rectangle(view, p1, self.mouse, COLORS[self.cls % len(COLORS)], 1)
        # crosshair helps line up box edges on blurry thermal blobs
        mx, my = self.mouse
        cv2.line(view, (mx, 0), (mx, out_h), (90, 90, 90), 1)
        cv2.line(view, (0, my), (out_w, my), (90, 90, 90), 1)

        done = self.done
        state = "EMPTY" if self.labelled and not self.boxes else ("labelled" if self.labelled else "NEW")
        cls_name = config.CLASSES[self.cls] if self.cls < len(config.CLASSES) else str(self.cls)
        bar = (f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}  {state}{'*' if self.dirty else ''}"
               f"  boxes:{len(self.boxes)}  class:{cls_name}  zoom:{self.zoom:.1f}x  done:{done}")
        cv2.rectangle(view, (0, 0), (out_w, 18), (0, 0, 0), -1)
        cv2.putText(view, bar, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    # ---- main loop ---------------------------------------------------------------
    def run(self, start=0):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load(start)
        while True:
            cv2.imshow(WIN, self.render())
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            ch = key & 0xFF if key < 256 else None
            if ch in (ord("q"), 27):
                self.save()
                break
            elif ch in (ord("d"), ord(" ")) or key in KEY_RIGHT:
                self.save()
                self.load(self.idx + 1)
            elif ch == ord("a") or key in KEY_LEFT:
                self.save()
                self.load(self.idx - 1)
            elif ch == ord("n"):
                self.load(self.idx + 1)
            elif ch == ord("f"):
                self.save()
                for j in range(1, len(self.images) + 1):
                    k = (self.idx + j) % len(self.images)
                    if not label_path_for(self.images[k], self.labels_dir).exists():
                        self.load(k)
                        break
                else:
                    print("every image is labelled")
            elif ch == ord("g"):
                try:
                    n = int(input(f"go to image (1-{len(self.images)}): "))
                    self.save()
                    self.load(n - 1)
                except ValueError:
                    pass
            elif ch in (ord("x"), 8) or key in KEY_DELETE:
                if self.sel >= 0:
                    self.push()
                    self.boxes.pop(self.sel)
                    self.sel, self.dirty = -1, True
            elif ch == ord("u"):
                if self.history:
                    self.boxes = self.history.pop()
                    self.sel, self.dirty = -1, True
            elif ch == ord("c"):
                self.push()
                self.boxes, self.sel, self.dirty = [], -1, True
            elif ch == ord("e"):
                self.boxes, self.sel, self.dirty = [], -1, True
                self.save()
                self.load(self.idx + 1)
            elif ch == ord("m"):
                if self.model is not None:
                    self.prelabel()
                else:
                    print("start with --model weights.pt to use pre-labelling")
            elif ch == ord("h"):
                self.hide = not self.hide
            elif ch == ord("i"):
                self.invert = not self.invert
            elif ch == ord("k"):
                self.contrast = (self.contrast + 1) % 3
            elif ch == ord("r"):
                self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
            elif ch is not None and ord("1") <= ch <= ord("9"):
                c = ch - ord("1")
                if c < len(config.CLASSES):
                    self.cls = c
                    if self.sel >= 0:
                        self.push()
                        self.boxes[self.sel][0] = c
                        self.dirty = True
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(config.RAW_IMAGES))
    ap.add_argument("--labels", default=str(config.RAW_LABELS))
    ap.add_argument("--list", help="text file of image paths to annotate, in order (from select_hard_frames.py)")
    ap.add_argument("--model", help="YOLO weights for pre-labelling unlabelled images")
    ap.add_argument("--conf", type=float, default=0.25, help="pre-label confidence")
    ap.add_argument("--scale", type=float, help="fixed display scale (default: fit screen)")
    ap.add_argument("--start", type=int, default=0, help="start at this image index (0-based); default first unlabelled")
    args = ap.parse_args()

    if args.list:
        images = [Path(l.strip()) for l in Path(args.list).read_text().splitlines() if l.strip()]
    else:
        images = list_images(args.images)
    if not images:
        raise SystemExit(f"no images in {args.images} - run extract_frames.py first")
    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)
    start = args.start
    if start == 0:
        start = next((i for i, p in enumerate(images) if not label_path_for(p, args.labels).exists()), 0)
    Annotator(images, args.labels, model, args.conf, args.scale).run(start)


if __name__ == "__main__":
    main()
