"""
Campus People Detection, Tracking & Counting  (v2)
====================================================
Three counting modes:
  line      – single vertical / horizontal line  (default – fully backward-compatible)
  multiline – multiple parallel lines
  zone      – polygon region-of-interest entry

Smoothing & track-memory features:
  --smooth-boxes   exponential moving average on bounding-box coordinates
  --smooth-alpha   EMA weight on the *previous* smooth value  (default 0.6)
  --track-buffer   keep drawing the last known box N frames after a track disappears
"""

from __future__ import annotations   # allow e.g. "float | None" on Python 3.9

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Visual constants
# ──────────────────────────────────────────────────────────────────────────────
PERSON_CLASS_ID = 0

C_BOX_LIVE      = (0, 255, 0)       # green   – detected, not yet counted
C_BOX_COUNTED   = (0, 128, 255)     # orange  – already counted
C_BOX_PREDICTED = (180, 180, 180)   # grey    – track-memory ghost box
C_ZONE_FILL     = (255, 255, 0)     # yellow  – zone polygon fill (transparent)
C_ZONE_BORDER   = (0, 215, 255)     # gold    – zone polygon border
C_TEXT          = (255, 255, 255)   # white
C_HUD_BG        = (0, 0, 0)        # black

# Colours for multi-line drawing (cycles if more lines than colours)
_LINE_PALETTE = [
    (0,   0,   255),   # red
    (255, 0,   0),     # blue
    (0,   165, 255),   # orange
    (0,   255, 255),   # yellow
    (255, 0,   255),   # magenta
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Campus people detection, tracking, and counting pipeline."
    )

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--video",   default="1.mp4",
                   help="Input video path (default: 1.mp4)")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="YOLOv8 weights file (default: yolov8n.pt)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   choices=["bytetrack.yaml", "botsort.yaml"],
                   help="Tracker config (default: bytetrack.yaml)")
    p.add_argument("--output",  default="outputs/output.mp4",
                   help="Annotated output video path")
    p.add_argument("--csv",     default="outputs/people_counts.csv",
                   help="CSV report path")
    p.add_argument("--show",    action="store_true",
                   help="Display live preview window while processing")

    # ── Detection thresholds ──────────────────────────────────────────────────
    p.add_argument("--conf", type=float, default=0.3,
                   help="Minimum detection confidence (default: 0.3)")
    p.add_argument("--iou",  type=float, default=0.5,
                   help="NMS IoU threshold (default: 0.5)")

    # ── Counting mode ─────────────────────────────────────────────────────────
    p.add_argument("--count-mode", default="line",
                   choices=["line", "zone", "multiline"],
                   help="Counting strategy: line | zone | multiline  (default: line)")

    # ── Line / multiline options ───────────────────────────────────────────────
    p.add_argument("--line", default="vertical",
                   choices=["vertical", "horizontal"],
                   help="Line orientation for 'line' and 'multiline' modes (default: vertical)")
    p.add_argument("--line-pos", type=float, default=0.5,
                   help="Single-line position as a fraction of frame width/height (default: 0.5)")
    p.add_argument("--line-positions", type=str, default=None,
                   help=(
                       "Comma-separated fractions for multiline mode, "
                       "e.g. '0.25,0.42,0.70'"
                   ))

    # ── Zone options ──────────────────────────────────────────────────────────
    p.add_argument("--zone-points", type=str, default=None,
                   help=(
                       "Semicolon-separated x,y polygon vertices for zone mode, "
                       "e.g. '100,300;600,250;1050,350;1000,650;200,650'"
                   ))

    # ── Box smoothing & track memory ──────────────────────────────────────────
    p.add_argument("--smooth-boxes",  action="store_true",
                   help="Apply EMA smoothing to bounding box coordinates")
    p.add_argument("--smooth-alpha",  type=float, default=0.6,
                   help=(
                       "EMA weight on the previous smoothed value. "
                       "Higher = smoother but slower to react (default: 0.6)"
                   ))
    p.add_argument("--track-buffer",  type=int, default=10,
                   help=(
                       "Frames to keep drawing a track's last known box after it disappears. "
                       "0 to disable (default: 10)"
                   ))

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────
def select_device() -> str:
    if torch.backends.mps.is_available():
        print("[INFO] Apple Silicon detected – using MPS.")
        return "mps"
    if torch.cuda.is_available():
        print("[INFO] CUDA GPU detected.")
        return "cuda"
    print("[INFO] No GPU found – using CPU.")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Video I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def open_video(path: str) -> cv2.VideoCapture:
    if not Path(path).exists():
        sys.exit(f"[ERROR] Video file not found: '{path}'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: '{path}'")
    return cap


def make_writer(path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    """
    Create a VideoWriter matching the source video's resolution and FPS.
    Uses H.264 (avc1) so outputs play directly in browsers and Streamlit.
    Falls back to mp4v if avc1 is unavailable on the current platform.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            print(f"[INFO] Video codec    : {codec}")
            return writer

    sys.exit(f"[ERROR] Cannot create video writer: '{path}'")


# ──────────────────────────────────────────────────────────────────────────────
# Argument pre-processing / validation
# ──────────────────────────────────────────────────────────────────────────────
def parse_zone_points(raw: str) -> np.ndarray:
    """
    Parse  '100,300;600,250;1050,350;1000,650;200,650'
    into   np.array([[100,300], [600,250], ...], dtype=int32)
    """
    try:
        pts = [
            [int(v) for v in pair.split(",")]
            for pair in raw.strip().split(";")
            if pair.strip()
        ]
        if len(pts) < 3:
            raise ValueError("Need at least 3 vertices.")
        return np.array(pts, dtype=np.int32)
    except Exception as exc:
        sys.exit(
            f"[ERROR] --zone-points is invalid ({exc}). "
            "Expected semicolon-separated x,y pairs, "
            "e.g.  '100,300;600,250;1050,350'"
        )


def parse_line_positions(raw: str, frame_dim: int) -> list[int]:
    """
    '0.25,0.42,0.70'  +  frame width/height  →  list of pixel coordinates
    """
    try:
        fracs = [float(f) for f in raw.split(",") if f.strip()]
    except Exception:
        sys.exit(
            "[ERROR] --line-positions must be comma-separated fractions, "
            "e.g. '0.25,0.42,0.70'"
        )
    if not fracs:
        sys.exit("[ERROR] --line-positions: provide at least one value.")
    return [int(frame_dim * f) for f in fracs]


# ──────────────────────────────────────────────────────────────────────────────
# Drawing – counting lines
# ──────────────────────────────────────────────────────────────────────────────
def _line_color(index: int) -> tuple[int, int, int]:
    return _LINE_PALETTE[index % len(_LINE_PALETTE)]


def draw_dashed_line(
    frame,
    orientation: str,
    coord: int,
    frame_w: int,
    frame_h: int,
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    """Draw a single dashed line and an optional label."""
    dash, gap = 20, 10
    if orientation == "vertical":
        y = 0
        while y < frame_h:
            cv2.line(frame, (coord, y), (coord, min(y + dash, frame_h)), color, 2)
            y += dash + gap
        if label:
            cv2.putText(frame, label, (coord + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        x = 0
        while x < frame_w:
            cv2.line(frame, (x, coord), (min(x + dash, frame_w), coord), color, 2)
            x += dash + gap
        if label:
            cv2.putText(frame, label, (5, coord - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_all_lines(
    frame,
    orientation: str,
    line_coords: list[int],
    frame_w: int,
    frame_h: int,
) -> None:
    """Draw all counting lines (each in its own colour)."""
    for i, coord in enumerate(line_coords):
        label = f"L{i + 1}" if len(line_coords) > 1 else ""
        draw_dashed_line(frame, orientation, coord, frame_w, frame_h,
                         _line_color(i), label)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing – zone polygon
# ──────────────────────────────────────────────────────────────────────────────
def draw_zone(frame, zone_pts: np.ndarray) -> None:
    """Draw a semi-transparent filled polygon with a border."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_pts], C_ZONE_FILL)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [zone_pts], isClosed=True, color=C_ZONE_BORDER, thickness=2)
    cv2.putText(frame, "COUNTING ZONE", tuple(zone_pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ZONE_BORDER, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing – bounding boxes
# ──────────────────────────────────────────────────────────────────────────────
def draw_box_label(
    frame,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    conf: float,
    counted: bool,
    predicted: bool = False,
) -> None:
    """
    Draw a bounding box with a label chip.
    Colour scheme:
      green  = live, not counted
      orange = counted
      grey   = track-memory (predicted / ghost)
    """
    if predicted:
        color = C_BOX_PREDICTED
        label = f"ID:{track_id} (predicted)"
        thickness = 1
    elif counted:
        color = C_BOX_COUNTED
        label = f"ID:{track_id}  {conf:.0%}"
        thickness = 2
    else:
        color = C_BOX_LIVE
        label = f"ID:{track_id}  {conf:.0%}"
        thickness = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    chip_y1 = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, chip_y1), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TEXT, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing – HUD overlay
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(
    frame,
    total_count: int,
    frame_no: int,
    fps: float,
    count_mode: str,
) -> None:
    hud_lines = [
        f"People counted : {total_count}",
        f"Count mode     : {count_mode}",
        f"Frame          : {frame_no}",
        f"FPS (source)   : {fps:.1f}",
    ]
    pad, line_h, box_w = 8, 22, 285
    box_h = len(hud_lines) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, text in enumerate(hud_lines):
        cv2.putText(frame, text, (10, 5 + pad + (i + 1) * line_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, C_TEXT, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Counting logic
# ──────────────────────────────────────────────────────────────────────────────
def check_line_crossing(
    prev_coord: float | None,
    curr_coord: float,
    line_coord: int,
    orientation: str,
) -> Optional[str]:
    """
    Return a direction string when the centre point crosses `line_coord`,
    otherwise return None.
      vertical   → "left_to_right" | "right_to_left"
      horizontal → "top_to_bottom" | "bottom_to_top"
    """
    if prev_coord is None:
        return None
    crossed = (
        (prev_coord < line_coord <= curr_coord)
        or (prev_coord > line_coord >= curr_coord)
    )
    if not crossed:
        return None
    if orientation == "vertical":
        return "left_to_right" if curr_coord >= prev_coord else "right_to_left"
    return "top_to_bottom" if curr_coord >= prev_coord else "bottom_to_top"


def check_zone_entry(
    was_inside: bool | None,
    cx: float,
    cy: float,
    zone_pts: np.ndarray,
) -> bool:
    """
    Return True on the exact frame the centre point enters the polygon.
    (Transition: outside → inside.)
    """
    now_inside = cv2.pointPolygonTest(zone_pts, (cx, cy), measureDist=False) >= 0
    return (was_inside is False) and now_inside


# ──────────────────────────────────────────────────────────────────────────────
# Box smoothing – exponential moving average
# ──────────────────────────────────────────────────────────────────────────────
def apply_ema_smoothing(
    smoothed_boxes: dict[int, list[float]],
    track_id: int,
    raw: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    """
    Blend the new raw box with the previous smoothed box.

    Formula:  smooth_t = alpha * smooth_{t-1}  +  (1 - alpha) * raw_t

    alpha = 0.6  → 60% weight on history, gentle smoothing.
    alpha = 0.8  → 80% weight on history, very smooth but laggy.
    alpha = 0.0  → no smoothing (raw box).
    """
    rx1, ry1, rx2, ry2 = map(float, raw)
    if track_id not in smoothed_boxes:
        smoothed_boxes[track_id] = [rx1, ry1, rx2, ry2]
    else:
        p = smoothed_boxes[track_id]
        smoothed_boxes[track_id] = [
            alpha * p[0] + (1 - alpha) * rx1,
            alpha * p[1] + (1 - alpha) * ry1,
            alpha * p[2] + (1 - alpha) * rx2,
            alpha * p[3] + (1 - alpha) * ry2,
        ]
    s = smoothed_boxes[track_id]
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])


# ──────────────────────────────────────────────────────────────────────────────
# Track memory – stores the last known state of each track
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackSnapshot:
    """Last confirmed detection state for one track ID."""
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    conf: float
    last_frame: int   # frame number when this snapshot was taken


# ──────────────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "frame_no", "timestamp_sec",
    "track_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2",
    "center_x", "center_y",
    "count_mode",
    "event_type",    # line_crossed | zone_entered | none
    "line_index",    # 0-based index of line crossed (multiline); -1 otherwise
    "counted",       # True only on the frame that triggered a count increment
]


def open_csv(path: str):
    """Create the CSV file, write the header, and return (file_handle, DictWriter)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return fh, writer


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    device = select_device()

    print(f"[INFO] Loading model  : {args.model}")
    model = YOLO(args.model)

    cap    = open_video(args.video)
    writer = make_writer(args.output, cap)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video          : {args.video}  ({frame_w}×{frame_h} @ {fps:.1f} fps, {total_fr} frames)")
    print(f"[INFO] Output         : {args.output}")
    print(f"[INFO] CSV            : {args.csv}")
    print(f"[INFO] Count mode     : {args.count_mode}")
    print(f"[INFO] Smooth boxes   : {args.smooth_boxes}  (alpha={args.smooth_alpha})")
    print(f"[INFO] Track buffer   : {args.track_buffer} frames")

    # ── Build counting geometry based on mode ──────────────────────────────────
    line_coords: list[int]           = []
    zone_pts:    Optional[np.ndarray] = None

    if args.count_mode == "line":
        dim = frame_w if args.line == "vertical" else frame_h
        line_coords = [int(dim * args.line_pos)]
        print(
            f"[INFO] Line           : {args.line}  "
            f"@ {'x' if args.line == 'vertical' else 'y'}={line_coords[0]}px "
            f"({args.line_pos * 100:.0f}% of frame)"
        )

    elif args.count_mode == "multiline":
        if not args.line_positions:
            sys.exit(
                "[ERROR] --count-mode multiline requires --line-positions, "
                "e.g. '0.25,0.42,0.70'"
            )
        dim = frame_w if args.line == "vertical" else frame_h
        line_coords = parse_line_positions(args.line_positions, dim)
        print(f"[INFO] Lines          : {args.line}  at pixel coords {line_coords}")

    elif args.count_mode == "zone":
        if not args.zone_points:
            sys.exit(
                "[ERROR] --count-mode zone requires --zone-points, "
                "e.g. '100,300;600,250;1050,350;1000,650;200,650'"
            )
        zone_pts = parse_zone_points(args.zone_points)
        print(f"[INFO] Zone polygon   : {zone_pts.tolist()}")

    # ── Per-track state dicts ──────────────────────────────────────────────────
    counted_ids:    set[int]                     = set()
    total_count:    int                          = 0
    all_seen_ids:   set[int]                     = set()

    # Line / multiline: last relevant centre coordinate per track ID
    prev_line_coord: dict[int, float]            = {}

    # Zone: was this track ID inside the zone last frame?
    prev_inside:     dict[int, bool]             = {}

    # EMA box buffer
    smoothed_boxes:  dict[int, list[float]]      = {}

    # Track memory for ghost boxes
    track_memory:    dict[int, TrackSnapshot]    = {}

    csv_fh, csv_writer = open_csv(args.csv)

    # ── Processing loop ────────────────────────────────────────────────────────
    frame_no = 0
    print("[INFO] Processing…  (add --show and press Q to stop early)\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            timestamp = frame_no / fps

            # ── YOLOv8 tracking ───────────────────────────────────────────────
            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=args.conf,
                iou=args.iou,
                tracker=args.tracker,
                device=device,
                verbose=False,
            )

            # ── Draw counting geometry ────────────────────────────────────────
            # (drawn first so boxes appear on top)
            if line_coords:
                draw_all_lines(frame, args.line, line_coords, frame_w, frame_h)
            if zone_pts is not None:
                draw_zone(frame, zone_pts)

            # ── Process detections ────────────────────────────────────────────
            active_ids: set[int] = set()

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.id is None:
                        continue   # no track ID yet (can happen on frame 1)

                    track_id = int(box.id.item())
                    raw_conf = float(box.conf.item())
                    xyxy     = box.xyxy[0].cpu().numpy().astype(int)
                    rx1, ry1, rx2, ry2 = xyxy

                    active_ids.add(track_id)
                    all_seen_ids.add(track_id)

                    # ── Apply EMA box smoothing ───────────────────────────────
                    if args.smooth_boxes:
                        x1, y1, x2, y2 = apply_ema_smoothing(
                            smoothed_boxes, track_id,
                            (rx1, ry1, rx2, ry2),
                            args.smooth_alpha,
                        )
                    else:
                        x1, y1, x2, y2 = rx1, ry1, rx2, ry2

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # ── Update track memory ───────────────────────────────────
                    track_memory[track_id] = TrackSnapshot(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        cx=cx, cy=cy,
                        conf=raw_conf,
                        last_frame=frame_no,
                    )

                    # ── Counting logic ────────────────────────────────────────
                    event_type  = "none"
                    line_index  = -1
                    counted_now = False

                    if args.count_mode in ("line", "multiline"):
                        # Centre coordinate relevant for this line orientation
                        centre_coord = cx if args.line == "vertical" else cy

                        if track_id not in counted_ids:
                            for i, lc in enumerate(line_coords):
                                direction = check_line_crossing(
                                    prev_line_coord.get(track_id),
                                    centre_coord, lc, args.line,
                                )
                                if direction:
                                    counted_ids.add(track_id)
                                    total_count += 1
                                    counted_now = True
                                    event_type  = "line_crossed"
                                    line_index  = i
                                    print(
                                        f"  ✓  ID:{track_id:>4}  "
                                        f"{'line L' + str(i + 1) if len(line_coords) > 1 else 'line'}  "
                                        f"dir={direction}  "
                                        f"total={total_count}  "
                                        f"(frame {frame_no})"
                                    )
                                    break  # one count per person even in multiline mode

                        # Always update the previous coordinate (counted or not)
                        prev_line_coord[track_id] = centre_coord

                    elif args.count_mode == "zone":
                        if track_id not in counted_ids:
                            entered = check_zone_entry(
                                prev_inside.get(track_id), cx, cy, zone_pts
                            )
                            if entered:
                                counted_ids.add(track_id)
                                total_count += 1
                                counted_now = True
                                event_type  = "zone_entered"
                                print(
                                    f"  ✓  ID:{track_id:>4}  zone entry  "
                                    f"total={total_count}  (frame {frame_no})"
                                )

                        # Update inside/outside state regardless of whether counted
                        prev_inside[track_id] = (
                            cv2.pointPolygonTest(zone_pts, (cx, cy), measureDist=False) >= 0
                        )

                    # ── Draw bounding box ─────────────────────────────────────
                    draw_box_label(
                        frame, x1, y1, x2, y2,
                        track_id, raw_conf,
                        counted=(track_id in counted_ids),
                        predicted=False,
                    )

                    # ── Write CSV row ─────────────────────────────────────────
                    csv_writer.writerow({
                        "frame_no"      : frame_no,
                        "timestamp_sec" : round(timestamp, 4),
                        "track_id"      : track_id,
                        "class_name"    : "person",
                        "confidence"    : round(raw_conf, 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "center_x"      : round(cx, 2),
                        "center_y"      : round(cy, 2),
                        "count_mode"    : args.count_mode,
                        "event_type"    : event_type,
                        "line_index"    : line_index,
                        "counted"       : counted_now,
                    })

            # ── Track memory – draw ghost boxes for recently lost tracks ───────
            if args.track_buffer > 0:
                for tid, snap in track_memory.items():
                    if tid in active_ids:
                        continue
                    age = frame_no - snap.last_frame
                    if 0 < age <= args.track_buffer:
                        draw_box_label(
                            frame,
                            snap.x1, snap.y1, snap.x2, snap.y2,
                            tid, snap.conf,
                            counted=(tid in counted_ids),
                            predicted=True,
                        )

                # Prune tracks that have been gone longer than the buffer
                stale_ids = [
                    tid for tid, snap in track_memory.items()
                    if frame_no - snap.last_frame > args.track_buffer
                ]
                for tid in stale_ids:
                    del track_memory[tid]

            # ── HUD ───────────────────────────────────────────────────────────
            draw_hud(frame, total_count, frame_no, fps, args.count_mode)

            # ── Progress print ────────────────────────────────────────────────
            if frame_no % 100 == 0:
                pct = frame_no / total_fr * 100 if total_fr else 0.0
                print(
                    f"  … frame {frame_no}/{total_fr}  ({pct:.1f}%)  "
                    f"counted: {total_count}  active_tracks: {len(active_ids)}"
                )

            writer.write(frame)

            if args.show:
                cv2.imshow("Campus People Counter  (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Early stop by user.")
                    break

    finally:
        cap.release()
        writer.release()
        csv_fh.close()
        if args.show:
            cv2.destroyAllWindows()

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print(f"  TOTAL PEOPLE COUNTED  :  {total_count}")
    print(f"  Unique track IDs seen :  {len(all_seen_ids)}")
    print(f"  Frames processed      :  {frame_no}")
    print(f"  Annotated video saved :  {args.output}")
    print(f"  CSV report saved      :  {args.csv}")
    print("═" * 58)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(parse_args())
