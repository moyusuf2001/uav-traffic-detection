"""
Railroad Crossing  —  Robust Multi-Class Detection, Tracking & Counting
========================================================================
Upgraded pipeline with four interlocking defences against parked-car
false positives and box flickering:

    A. Separate class-specific ROIs  — tight vehicle ROI excludes parking lots;
                         optional pedestrian ROI covers sidewalk/crossing area
  B. Sustained motion  — smoothed centre + per-frame avg movement; requires
                         N consecutive frames above threshold before MOVING;
                         stays RECENTLY_MOVING for a configurable buffer before
                         becoming STATIONARY
  C. Camera-jitter compensation — sparse optical flow subtracts drone drift
                         from per-track movement so jitter ≠ motion
  D. Velocity prediction — confirmed moving tracks keep a ghost box that
                         advances with the estimated velocity while the
                         detector is temporarily off the object

Run:
    python railroad_crossing.py --video railroad_crossing.mp4

Best production command:
    python railroad_crossing.py --video railroad_crossing.mp4 \\
        --model yolov8m.pt --tracker bytetrack.yaml \\
        --count-mode multiline --line vertical --line-positions 0.25,0.5,0.75 \\
        --vehicle-roi-points "300,400;1400,380;1500,850;200,870" \\
        --pedestrian-roi-points "100,350;600,330;620,900;80,900" \\
        --use-separate-rois \\
        --smooth-boxes --smooth-alpha 0.75 \\
        --motion-window 12 \\
        --motion-threshold-vehicle 20 --motion-threshold-person 10 \\
        --stationary-confirm-frames 12 --recently-moving-buffer 45 \\
        --track-buffer 15 --predict-missing-tracks --max-prediction-frames 8 \\
        --compensate-camera-motion
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1  —  COCO class registry & visual constants
# ──────────────────────────────────────────────────────────────────────────────
DETECT_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CLASS_IDS: list[int] = sorted(DETECT_CLASSES.keys())

# BGR colour per class for bounding boxes
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":     (0,   210,   0),
    "bicycle":    (210, 210,   0),
    "car":        (200,  60,   0),
    "motorcycle": (0,   150, 255),
    "bus":        (180,   0, 180),
    "truck":      (0,     0, 220),
}

# Motion states
MOVING          = "moving"
RECENTLY_MOVING = "recently_moving"
STATIONARY      = "stationary"
INSUFFICIENT    = "insufficient_data"

# Misc colours
C_TEXT      = (255, 255, 255)
C_PREDICTED = (140, 140, 140)
C_STATIONARY_DEBUG = (90,  90,  90)
C_INSUFFICIENT     = (180, 180, 180)
C_VEH_ROI_BORDER   = (0,   215, 255)   # gold    – vehicle ROI
C_VEH_ROI_FILL     = (0,   215, 255)
C_PED_ROI_BORDER   = (0,   200, 120)   # green   – pedestrian ROI
C_PED_ROI_FILL     = (0,   200, 120)
C_ZONE_FILL        = (255, 255,   0)
C_ZONE_EDGE        = (0,   215, 255)
_LINE_PALETTE = [(255,255,255),(0,215,255),(180,105,255),(0,255,127),(255,140,0)]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2  —  CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust multi-class detection, tracking & counting for railroad crossings."
    )
    # I/O
    p.add_argument("--video",   default="railroad_crossing.mp4")
    p.add_argument("--model",   default="yolov8m.pt")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--output",  default="outputs/railroad_crossing_output.mp4")
    p.add_argument("--csv",     default="outputs/railroad_crossing_results.csv")
    p.add_argument("--show",    action="store_true")

    # Detection
    p.add_argument("--imgsz",  type=int,   default=1280)
    p.add_argument("--conf",   type=float, default=0.15)
    p.add_argument("--iou",    type=float, default=0.45)

    # Counting
    p.add_argument("--count-mode", default="multiline",
                   choices=["line", "multiline", "zone"])
    p.add_argument("--line",           default="vertical",
                   choices=["vertical", "horizontal"])
    p.add_argument("--line-pos",       type=float, default=0.5)
    p.add_argument("--line-positions", type=str,   default=None)
    p.add_argument("--zone-points",    type=str,   default=None)

    # A: Class-specific ROIs
    p.add_argument("--vehicle-roi-points", type=str, default=None,
                   help=(
                       "Tight polygon covering the active roadway only. "
                       "Applies to car, truck, bus, motorcycle, bicycle. "
                       "Excludes parking lots automatically. "
                       "e.g. '300,400;1400,380;1500,850;200,870'"
                   ))
    p.add_argument("--pedestrian-roi-points", type=str, default=None,
                   help=(
                       "Optional separate polygon for person detection. "
                       "Use to cover a sidewalk or crossing zone distinct from the road. "
                       "Falls back to --vehicle-roi-points if not provided."
                   ))
    p.add_argument("--use-separate-rois", action="store_true",
                   help=(
                       "Apply vehicle ROI to vehicle classes and pedestrian ROI to persons. "
                       "Without this flag, vehicle ROI applies to all classes."
                   ))
    p.add_argument("--hide-roi-overlay", action="store_true",
                   help="Do not draw ROI polygons on the output video (clean output for presentation)")
    p.add_argument("--hud-classes", type=str, default="person,car,truck",
                   help="Comma-separated list of classes to show on the HUD (default: person,car,truck)")

    # B: Sustained motion
    p.add_argument("--motion-window",            type=int,   default=12,
                   help="Frames in the rolling motion window (default: 12)")
    p.add_argument("--motion-threshold-person",  type=float, default=10.0,
                   help="Avg px/frame for person to be moving (default: 10)")
    p.add_argument("--motion-threshold-vehicle", type=float, default=20.0,
                   help="Avg px/frame for vehicles to be moving (default: 20)")
    p.add_argument("--stationary-confirm-frames",type=int,   default=12,
                   help="Consecutive below-threshold frames before STATIONARY (default: 12)")
    p.add_argument("--recently-moving-buffer",   type=int,   default=45,
                   help="Frames to stay RECENTLY_MOVING after last confirmed movement (default: 45)")
    p.add_argument("--min-active-frames",        type=int,   default=5,
                   help="Min frames before motion classification starts (default: 5)")

    # C: Camera jitter compensation
    p.add_argument("--compensate-camera-motion", action="store_true",
                   help="Subtract per-frame camera translation from track motion (optical flow)")

    # D: Flicker reduction
    p.add_argument("--track-buffer",         type=int,  default=15,
                   help="Frames to keep a ghost box after a track disappears (default: 15)")
    p.add_argument("--predict-missing-tracks", action="store_true",
                   help="Advance ghost boxes with velocity estimate instead of freezing them")
    p.add_argument("--max-prediction-frames", type=int,  default=8,
                   help="Cap on velocity-predicted frames (default: 8)")

    # E: Box smoothing
    p.add_argument("--smooth-boxes",  action="store_true")
    p.add_argument("--smooth-alpha",  type=float, default=0.75)

    # G: Debug tools
    p.add_argument("--show-stationary",   action="store_true",
                   help="Draw stationary objects in dark grey (debug)")
    p.add_argument("--show-motion-values", action="store_true",
                   help="Overlay displacement and motion state on each box")
    p.add_argument("--save-debug-frames",  action="store_true",
                   help="Save every 50th annotated frame to outputs/debug_frames/")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3  —  Device
# ──────────────────────────────────────────────────────────────────────────────
def select_device() -> str:
    if torch.backends.mps.is_available():
        print("[INFO] Apple Silicon – using MPS.")
        return "mps"
    if torch.cuda.is_available():
        print("[INFO] CUDA GPU detected.")
        return "cuda"
    print("[INFO] Using CPU.")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4  —  Video I/O
# ──────────────────────────────────────────────────────────────────────────────
def open_video(path: str) -> cv2.VideoCapture:
    if not Path(path).exists():
        sys.exit(f"[ERROR] Video not found: '{path}'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: '{path}'")
    return cap


def make_writer(path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for codec in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if wr.isOpened():
            print(f"[INFO] Video codec   : {codec}")
            return wr
    sys.exit(f"[ERROR] Cannot create writer: '{path}'")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5  —  Camera jitter compensation  (sparse optical flow)
# ──────────────────────────────────────────────────────────────────────────────
_LK_PARAMS  = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
_FEAT_PARAMS = dict(maxCorners=300, qualityLevel=0.01, minDistance=10, blockSize=7)


def estimate_camera_motion(
    prev_gray: np.ndarray | None,
    curr_gray: np.ndarray,
) -> tuple[float, float]:
    """
    Estimate global frame-to-frame camera translation using sparse LK optical flow.

    Strategy: track static background features; the median displacement of ALL
    features (including moving objects) is dominated by the background because
    moving objects are a small fraction of the frame → robust via median.

    Returns (dx, dy) in pixels that the camera moved this frame.
    """
    if prev_gray is None:
        return 0.0, 0.0

    pts = cv2.goodFeaturesToTrack(prev_gray, **_FEAT_PARAMS)
    if pts is None or len(pts) < 8:
        return 0.0, 0.0

    pts_new, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **_LK_PARAMS)
    if pts_new is None:
        return 0.0, 0.0

    mask = status.flatten() == 1
    if mask.sum() < 5:
        return 0.0, 0.0

    deltas = (pts_new[mask] - pts[mask]).reshape(-1, 2)
    # Median is robust to the minority of points on moving objects
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))
    return dx, dy


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6  —  Road ROI polygon filter
# ──────────────────────────────────────────────────────────────────────────────
def parse_polygon(raw: str, label: str) -> np.ndarray:
    try:
        pts = [[int(v) for v in pair.split(",")]
               for pair in raw.strip().split(";") if pair.strip()]
        if len(pts) < 3:
            raise ValueError("Need ≥ 3 vertices.")
        return np.array(pts, dtype=np.int32)
    except Exception as e:
        sys.exit(f"[ERROR] {label} invalid ({e}). Use 'x1,y1;x2,y2;...'")


def parse_line_positions(raw: str, dim: int) -> list[int]:
    try:
        return [int(float(f) * dim) for f in raw.split(",") if f.strip()]
    except Exception:
        sys.exit("[ERROR] --line-positions must be comma-separated fractions.")


def point_in_roi(cx: float, cy: float, roi: np.ndarray | None) -> bool:
    """Return True when no ROI is configured, or when (cx,cy) is inside the polygon."""
    if roi is None:
        return True
    return cv2.pointPolygonTest(roi, (cx, cy), measureDist=False) >= 0


def get_roi_for_class(
    class_name:     str,
    vehicle_roi:    np.ndarray | None,
    ped_roi:        np.ndarray | None,
    use_separate:   bool,
) -> np.ndarray | None:
    """
    Return the correct ROI polygon for a given class.

    Logic:
      - Vehicles (car/truck/bus/motorcycle/bicycle) always use vehicle_roi.
      - Persons:
          use_separate=True  → use ped_roi if set, else vehicle_roi as fallback
          use_separate=False → use vehicle_roi (same zone for everyone)
      - If the resolved ROI is None, no spatial filter is applied for that class.
    """
    is_person = (class_name == "person")
    if not is_person:
        return vehicle_roi
    # Person
    if use_separate and ped_roi is not None:
        return ped_roi
    return vehicle_roi   # fallback: person shares the vehicle road zone


def draw_roi_polygon(
    frame:     np.ndarray,
    roi:       np.ndarray,
    label:     str,
    border_c:  tuple[int, int, int],
    fill_c:    tuple[int, int, int],
) -> None:
    """Draw a labelled, semi-transparent ROI polygon."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi], fill_c)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    cv2.polylines(frame, [roi], isClosed=True, color=border_c, thickness=2)
    # Label at the topmost point of the polygon
    top_pt = tuple(roi[roi[:, 1].argmin()])
    cv2.putText(frame, label, top_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_c, 2, cv2.LINE_AA)


def draw_all_rois(
    frame:       np.ndarray,
    vehicle_roi: np.ndarray | None,
    ped_roi:     np.ndarray | None,
    hide:        bool,
) -> None:
    """Draw vehicle and/or pedestrian ROI polygons unless --hide-roi-overlay is set."""
    if hide:
        return
    if vehicle_roi is not None:
        draw_roi_polygon(frame, vehicle_roi, "VEHICLE ROI",
                         C_VEH_ROI_BORDER, C_VEH_ROI_FILL)
    if ped_roi is not None:
        draw_roi_polygon(frame, ped_roi, "PEDESTRIAN ROI",
                         C_PED_ROI_BORDER, C_PED_ROI_FILL)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7  —  Sustained-motion classifier
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionInfo:
    """All motion state for one tracked object."""
    # Rolling window of smoothed, camera-compensated centre points
    center_history:         deque = field(default_factory=lambda: deque(maxlen=12))
    # EMA-smoothed centre (responsive alpha ~0.35 so it follows real motion)
    smooth_cx:              float = 0.0
    smooth_cy:              float = 0.0
    smooth_init:            bool  = False
    # Classification
    state:                  str   = INSUFFICIENT
    displacement:           float = 0.0   # total displacement across window
    avg_per_frame:          float = 0.0   # avg px moved per frame — primary metric
    consecutive_stationary: int   = 0     # frames below threshold in a row
    last_moving_frame:      int   = -1
    frame_count:            int   = 0


def get_threshold(class_name: str, thr_person: float, thr_vehicle: float) -> float:
    return thr_person if class_name == "person" else thr_vehicle


def update_motion(
    trackers:      dict[int, MotionInfo],
    track_id:      int,
    cx:            float,
    cy:            float,
    frame_no:      int,
    class_name:    str,
    window:        int,
    thr_person:    float,
    thr_vehicle:   float,
    stationary_confirm: int,
    recently_buffer:    int,
    min_frames:    int,
    cam_dx:        float = 0.0,
    cam_dy:        float = 0.0,
) -> MotionInfo:
    """
    Update the motion history for one track and re-classify its state.

    Camera-compensated smoothed centres are stored.
    Per-frame average movement (not just start-to-end distance) is used so
    bounding-box jitter (which reverses direction each frame) averages out.

    State transitions:
      INSUFFICIENT     → not enough frames
      MOVING           → avg_per_frame >= threshold
      RECENTLY_MOVING  → was MOVING recently (within recently_buffer frames)
                         OR consecutive_stationary < stationary_confirm
      STATIONARY       → consecutive_stationary >= stationary_confirm
                         AND recently_buffer frames have passed since last MOVING
    """
    threshold = get_threshold(class_name, thr_person, thr_vehicle)

    if track_id not in trackers:
        mi = MotionInfo()
        mi.center_history = deque(maxlen=window)
        trackers[track_id] = mi
    mi = trackers[track_id]
    mi.frame_count += 1

    # ── Step 1: smooth the raw centre  (alpha=0.35 = responsive) ─────────────
    center_alpha = 0.35
    if not mi.smooth_init:
        mi.smooth_cx  = cx - cam_dx
        mi.smooth_cy  = cy - cam_dy
        mi.smooth_init = True
    else:
        # Subtract the camera's motion from the new centre before smoothing
        compensated_cx = cx - cam_dx
        compensated_cy = cy - cam_dy
        mi.smooth_cx = center_alpha * compensated_cx + (1 - center_alpha) * mi.smooth_cx
        mi.smooth_cy = center_alpha * compensated_cy + (1 - center_alpha) * mi.smooth_cy

    mi.center_history.append((mi.smooth_cx, mi.smooth_cy))

    # ── Step 2: early exit if not enough data ─────────────────────────────────
    if mi.frame_count < min_frames or len(mi.center_history) < 2:
        mi.state        = INSUFFICIENT
        mi.displacement = 0.0
        mi.avg_per_frame = 0.0
        return mi

    # ── Step 3: compute average per-frame movement ────────────────────────────
    # Using consecutive-pair deltas (not just start→end) so back-and-forth
    # bounding-box jitter on a parked car sums near zero instead of accumulating.
    pts = list(mi.center_history)
    frame_deltas = [
        math.sqrt((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)
        for i in range(len(pts) - 1)
    ]
    mi.avg_per_frame = sum(frame_deltas) / len(frame_deltas)

    # Total displacement (for display)
    mi.displacement = math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)

    # ── Step 4: classify ──────────────────────────────────────────────────────
    if mi.avg_per_frame >= threshold:
        mi.state                  = MOVING
        mi.last_moving_frame      = frame_no
        mi.consecutive_stationary = 0
    else:
        mi.consecutive_stationary += 1
        frames_since_moving = (
            frame_no - mi.last_moving_frame if mi.last_moving_frame >= 0 else frame_no
        )
        if frames_since_moving <= recently_buffer:
            mi.state = RECENTLY_MOVING
        elif mi.consecutive_stationary >= stationary_confirm:
            mi.state = STATIONARY
        else:
            mi.state = RECENTLY_MOVING   # transitioning, not yet confirmed stationary

    return mi


def is_countable(state: str) -> bool:
    return state in (MOVING, RECENTLY_MOVING)


def should_draw(state: str, show_stationary: bool) -> bool:
    if state in (MOVING, RECENTLY_MOVING, INSUFFICIENT):
        return True
    return show_stationary


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8  —  Track memory with velocity prediction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackSnapshot:
    x1: int; y1: int; x2: int; y2: int
    cx: float; cy: float
    vx: float; vy: float      # velocity estimate in px/frame
    conf: float
    class_name: str
    motion_state: str
    last_frame: int


def _estimate_velocity(history: deque) -> tuple[float, float]:
    """Average of the last ≤5 consecutive centre displacements."""
    pts = list(history)
    n   = min(len(pts) - 1, 5)
    if n < 1:
        return 0.0, 0.0
    recent = pts[-(n + 1):]
    deltas = [(recent[i+1][0] - recent[i][0], recent[i+1][1] - recent[i][1])
              for i in range(len(recent) - 1)]
    vx = sum(d[0] for d in deltas) / len(deltas)
    vy = sum(d[1] for d in deltas) / len(deltas)
    return vx, vy


def predicted_bbox(
    snap: TrackSnapshot, frames_elapsed: int
) -> tuple[int, int, int, int]:
    """Advance the last known bbox by velocity × elapsed frames."""
    dx = snap.vx * frames_elapsed
    dy = snap.vy * frames_elapsed
    return (
        int(snap.x1 + dx), int(snap.y1 + dy),
        int(snap.x2 + dx), int(snap.y2 + dy),
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9  —  EMA bounding-box smoothing
# ──────────────────────────────────────────────────────────────────────────────
def smooth_box(
    buf: dict[int, list[float]],
    track_id: int,
    raw: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    """EMA on box coordinates.  alpha=0.75 = strong smoothing."""
    rx1, ry1, rx2, ry2 = map(float, raw)
    if track_id not in buf:
        buf[track_id] = [rx1, ry1, rx2, ry2]
    else:
        p = buf[track_id]
        buf[track_id] = [
            alpha * p[0] + (1 - alpha) * rx1,
            alpha * p[1] + (1 - alpha) * ry1,
            alpha * p[2] + (1 - alpha) * rx2,
            alpha * p[3] + (1 - alpha) * ry2,
        ]
    s = buf[track_id]
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10  —  Drawing
# ──────────────────────────────────────────────────────────────────────────────
def _lighten(c: tuple[int, int, int], f: float = 0.5) -> tuple[int, int, int]:
    return tuple(int(ch + (255 - ch) * f) for ch in c)  # type: ignore[return-value]


def _line_color(idx: int) -> tuple[int, int, int]:
    return _LINE_PALETTE[idx % len(_LINE_PALETTE)]


def draw_dashed_line(
    frame, orientation: str, coord: int,
    frame_w: int, frame_h: int,
    color: tuple, label: str = "",
) -> None:
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


def draw_all_lines(frame, orientation, coords, frame_w, frame_h) -> None:
    for i, c in enumerate(coords):
        draw_dashed_line(frame, orientation, c, frame_w, frame_h,
                         _line_color(i), f"L{i+1}" if len(coords) > 1 else "")


def draw_zone(frame, pts: np.ndarray) -> None:
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], C_ZONE_FILL)
    cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=C_ZONE_EDGE, thickness=2)
    cv2.putText(frame, "COUNT ZONE", tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ZONE_EDGE, 2, cv2.LINE_AA)


def draw_box_label(
    frame,
    x1: int, y1: int, x2: int, y2: int,
    class_name: str, track_id: int, conf: float,
    counted: bool, motion_state: str,
    predicted: bool = False,
    show_motion_values: bool = False,
    avg_per_frame: float = 0.0,
) -> None:
    """
    Visual encoding by motion state:
      MOVING          → full class colour
      RECENTLY_MOVING → 45% lighter class colour  +  "(slow)"
      INSUFFICIENT    → grey  +  "(new)"
      STATIONARY      → dark grey  +  "(parked)"  — debug only
      predicted       → ghost grey  +  "(pred)"
    """
    if predicted:
        color, label, thick = C_PREDICTED, f"{class_name} ID:{track_id} (pred)", 1
    elif motion_state == MOVING:
        color = CLASS_COLORS.get(class_name, (200, 200, 200))
        label = f"{class_name} ID:{track_id}  {conf:.0%}"
        thick = 2
    elif motion_state == RECENTLY_MOVING:
        color = _lighten(CLASS_COLORS.get(class_name, (200, 200, 200)), 0.45)
        label = f"{class_name} ID:{track_id}  {conf:.0%} (slow)"
        thick = 2
    elif motion_state == INSUFFICIENT:
        color, label, thick = C_INSUFFICIENT, f"{class_name} ID:{track_id} (new)", 1
    else:  # STATIONARY — debug
        color, label, thick = C_STATIONARY_DEBUG, f"parked ID:{track_id}", 1

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cy_chip = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, cy_chip), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_TEXT, 1, cv2.LINE_AA)

    if show_motion_values and not predicted:
        debug_txt = f"avg:{avg_per_frame:.1f}px {motion_state[:3]}"
        cv2.putText(frame, debug_txt, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hud(
    frame, counts: dict[str, int], frame_no: int,
    fps: float, mode: str,
    veh_roi: bool, ped_roi: bool, cam_comp: bool,
    hud_classes: list[str],
) -> None:
    """
    Draw a clean, modern HUD in the top-left corner.

    Layout:
        ┌──────────────────────────┐
        │  TRAFFIC COUNTER         │
        │ ──────────────────────── │
        │  PERSON           0      │
        │  CAR             12      │
        │  TRUCK            4      │
        │ ──────────────────────── │
        │  TOTAL           16      │
        └──────────────────────────┘
    """
    # Filter to requested classes, preserving order
    rows = [(c, counts.get(c, 0)) for c in hud_classes if c in CLASS_COLORS]
    total = sum(n for _, n in rows)

    # Layout constants
    box_w     = 280
    title_h   = 38
    row_h     = 36
    sep_h     = 1
    total_h   = 42
    pad       = 12
    box_h     = title_h + sep_h + len(rows) * row_h + sep_h + total_h
    x0, y0    = 16, 16

    # Translucent dark background with a subtle border
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (12, 12, 14), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 64), 1)

    # ── Title bar ─────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + title_h), (28, 28, 32), -1)
    cv2.putText(frame, "TRAFFIC COUNTER",
                (x0 + pad, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)

    # ── Separator under title ────────────────────────────────────────────────
    sep_y = y0 + title_h
    cv2.line(frame, (x0, sep_y), (x0 + box_w, sep_y), (90, 90, 96), 1)

    # ── Class rows ────────────────────────────────────────────────────────────
    for i, (cls, n) in enumerate(rows):
        ry = y0 + title_h + sep_h + i * row_h
        color = CLASS_COLORS.get(cls, (200, 200, 200))

        # Coloured tag dot on the left
        cv2.circle(frame, (x0 + pad + 6, ry + row_h // 2), 5, color, -1)

        # Class name (uppercase, light grey)
        cv2.putText(frame, cls.upper(),
                    (x0 + pad + 22, ry + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 205), 1, cv2.LINE_AA)

        # Count number, right-aligned, large bold white
        num_txt = str(n)
        (tw, _), _ = cv2.getTextSize(num_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(frame, num_txt,
                    (x0 + box_w - pad - tw, ry + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Separator above total ─────────────────────────────────────────────────
    tot_sep_y = y0 + title_h + sep_h + len(rows) * row_h
    cv2.line(frame, (x0, tot_sep_y), (x0 + box_w, tot_sep_y), (90, 90, 96), 1)

    # ── Total row ─────────────────────────────────────────────────────────────
    ty = tot_sep_y + sep_h
    cv2.rectangle(frame, (x0, ty), (x0 + box_w, ty + total_h), (28, 28, 32), -1)
    cv2.putText(frame, "TOTAL",
                (x0 + pad, ty + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 200, 255), 2, cv2.LINE_AA)
    total_txt = str(total)
    (tw, _), _ = cv2.getTextSize(total_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, total_txt,
                (x0 + box_w - pad - tw, ty + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 220, 255), 2, cv2.LINE_AA)

    # ── Small status footer below the box ─────────────────────────────────────
    roi_str = ("veh+ped" if (veh_roi and ped_roi)
               else "veh" if veh_roi
               else "ped" if ped_roi
               else "none")
    status_txt = f"frame {frame_no}  mode:{mode}  roi:{roi_str}"
    cv2.putText(frame, status_txt,
                (x0 + 2, y0 + box_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 165), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11  —  Counting logic
# ──────────────────────────────────────────────────────────────────────────────
def check_line_crossing(
    prev: float | None, curr: float, coord: int, orientation: str
) -> Optional[str]:
    if prev is None:
        return None
    if (prev < coord <= curr) or (prev > coord >= curr):
        if orientation == "vertical":
            return "left_to_right" if curr >= prev else "right_to_left"
        return "top_to_bottom" if curr >= prev else "bottom_to_top"
    return None


def check_zone_entry(was_inside: bool | None, cx: float, cy: float,
                     pts: np.ndarray) -> bool:
    inside = cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
    return (was_inside is False) and inside


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12  —  CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "frame_no", "timestamp_sec",
    "track_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
    "count_mode", "event_type", "line_index", "counted",
    "motion_state", "avg_per_frame", "displacement", "moving",
    "in_roi",
]


def open_csv(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    w.writeheader()
    return fh, w


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13  —  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    device = select_device()

    print(f"[INFO] Loading model             : {args.model}")
    model = YOLO(args.model)

    cap    = open_video(args.video)
    writer = make_writer(args.output, cap)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video                     : {args.video}  ({frame_w}×{frame_h} @ {fps:.1f}fps)")
    print(f"[INFO] Count mode                : {args.count_mode}")
    print(f"[INFO] Motion window             : {args.motion_window} frames")
    print(f"[INFO] Threshold person/vehicle  : {args.motion_threshold_person} / {args.motion_threshold_vehicle} px/frame")
    print(f"[INFO] Stationary confirm frames : {args.stationary_confirm_frames}")
    print(f"[INFO] Recently-moving buffer    : {args.recently_moving_buffer} frames")
    print(f"[INFO] Camera compensation       : {args.compensate_camera_motion}")
    print(f"[INFO] Predict missing tracks    : {args.predict_missing_tracks}")
    print(f"[INFO] Smooth boxes (alpha)      : {args.smooth_boxes} ({args.smooth_alpha})")

    # ── Parse class-specific ROIs ──────────────────────────────────────────────
    vehicle_roi: Optional[np.ndarray] = None
    ped_roi:     Optional[np.ndarray] = None

    if args.vehicle_roi_points:
        vehicle_roi = parse_polygon(args.vehicle_roi_points, "--vehicle-roi-points")
        print(f"[INFO] Vehicle ROI               : {vehicle_roi.tolist()}")
    else:
        print("[INFO] Vehicle ROI               : none (all vehicle detections pass through)")

    if args.pedestrian_roi_points:
        ped_roi = parse_polygon(args.pedestrian_roi_points, "--pedestrian-roi-points")
        print(f"[INFO] Pedestrian ROI            : {ped_roi.tolist()}")
    elif args.use_separate_rois:
        print("[INFO] Pedestrian ROI            : not set – persons fall back to vehicle ROI")

    print(f"[INFO] Separate ROIs             : {args.use_separate_rois}")
    print(f"[INFO] Hide ROI overlay          : {args.hide_roi_overlay}")

    # ── Parse counting geometry ────────────────────────────────────────────────
    line_coords: list[int]            = []
    zone_pts:    Optional[np.ndarray] = None

    if args.count_mode == "line":
        dim = frame_w if args.line == "vertical" else frame_h
        line_coords = [int(dim * args.line_pos)]
    elif args.count_mode == "multiline":
        if not args.line_positions:
            sys.exit("[ERROR] --count-mode multiline needs --line-positions")
        dim = frame_w if args.line == "vertical" else frame_h
        line_coords = parse_line_positions(args.line_positions, dim)
    elif args.count_mode == "zone":
        if not args.zone_points:
            sys.exit("[ERROR] --count-mode zone needs --zone-points")
        zone_pts = parse_polygon(args.zone_points, "--zone-points")

    # ── Debug frame output dir ─────────────────────────────────────────────────
    debug_dir = Path("outputs/debug_frames")
    if args.save_debug_frames:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Debug frames → {debug_dir}/")

    # ── Per-class counters & ID sets ───────────────────────────────────────────
    class_names      = list(DETECT_CLASSES.values())
    counts:          dict[str, int]       = {c: 0   for c in class_names}
    counted_per_cls: dict[str, set[int]]  = {c: set() for c in class_names}
    all_seen_ids:    set[int]             = set()

    prev_line_coord: dict[int, float]     = {}
    prev_inside:     dict[int, bool]      = {}

    # ── Motion, smoothing, memory ──────────────────────────────────────────────
    motion_trackers: dict[int, MotionInfo]    = {}
    smoothed_boxes:  dict[int, list[float]]   = {}
    track_memory:    dict[int, TrackSnapshot] = {}

    # ── Camera motion compensation state ──────────────────────────────────────
    prev_gray: Optional[np.ndarray] = None
    cam_dx, cam_dy                  = 0.0, 0.0

    csv_fh, csv_writer = open_csv(args.csv)
    frame_no = 0

    print("[INFO] Processing…  (add --show and press Q to quit early)\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            timestamp = frame_no / fps

            # ── Camera motion estimate for this frame ─────────────────────────
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.compensate_camera_motion:
                cam_dx, cam_dy = estimate_camera_motion(prev_gray, curr_gray)
            prev_gray = curr_gray

            # ── YOLO tracking ─────────────────────────────────────────────────
            results = model.track(
                frame,
                persist=True,
                classes=CLASS_IDS,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                tracker=args.tracker,
                device=device,
                verbose=False,
            )

            # ── Draw geometry (behind boxes) ──────────────────────────────────
            draw_all_rois(frame, vehicle_roi, ped_roi, args.hide_roi_overlay)
            if line_coords:
                draw_all_lines(frame, args.line, line_coords, frame_w, frame_h)
            if zone_pts is not None:
                draw_zone(frame, zone_pts)

            active_ids: set[int] = set()

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.id is None:
                        continue

                    track_id   = int(box.id.item())
                    raw_conf   = float(box.conf.item())
                    class_id   = int(box.cls.item())
                    class_name = DETECT_CLASSES.get(class_id, "unknown")
                    xyxy       = box.xyxy[0].cpu().numpy().astype(int)
                    rx1, ry1, rx2, ry2 = xyxy

                    # ── A: Class-specific ROI filter ──────────────────────────
                    # Resolve which ROI applies to this class, then test the
                    # raw (un-smoothed) centre so parked cars outside the
                    # tight road polygon are dropped before any processing.
                    raw_cx  = (rx1 + rx2) / 2.0
                    raw_cy  = (ry1 + ry2) / 2.0
                    class_roi = get_roi_for_class(
                        class_name, vehicle_roi, ped_roi, args.use_separate_rois
                    )
                    in_roi = point_in_roi(raw_cx, raw_cy, class_roi)
                    if not in_roi:
                        continue   # outside class-specific road zone → ignore

                    active_ids.add(track_id)
                    all_seen_ids.add(track_id)

                    # ── E: EMA box smoothing ──────────────────────────────────
                    if args.smooth_boxes:
                        x1, y1, x2, y2 = smooth_box(
                            smoothed_boxes, track_id,
                            (rx1, ry1, rx2, ry2), args.smooth_alpha,
                        )
                    else:
                        x1, y1, x2, y2 = rx1, ry1, rx2, ry2

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # ── B + C: Sustained motion classification ────────────────
                    motion = update_motion(
                        motion_trackers, track_id, cx, cy, frame_no,
                        class_name,
                        window=args.motion_window,
                        thr_person=args.motion_threshold_person,
                        thr_vehicle=args.motion_threshold_vehicle,
                        stationary_confirm=args.stationary_confirm_frames,
                        recently_buffer=args.recently_moving_buffer,
                        min_frames=args.min_active_frames,
                        cam_dx=cam_dx,
                        cam_dy=cam_dy,
                    )

                    # ── Velocity estimate from smoothed centre history ─────────
                    vx, vy = _estimate_velocity(motion.center_history)

                    # ── D: Update track memory ────────────────────────────────
                    track_memory[track_id] = TrackSnapshot(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        cx=cx, cy=cy, vx=vx, vy=vy,
                        conf=raw_conf, class_name=class_name,
                        motion_state=motion.state, last_frame=frame_no,
                    )

                    # ── Draw (skip parked unless debug mode) ──────────────────
                    if should_draw(motion.state, args.show_stationary):
                        draw_box_label(
                            frame, x1, y1, x2, y2,
                            class_name, track_id, raw_conf,
                            counted=(track_id in counted_per_cls[class_name]),
                            motion_state=motion.state,
                            predicted=False,
                            show_motion_values=args.show_motion_values,
                            avg_per_frame=motion.avg_per_frame,
                        )

                    # ── F: Counting — only moving/recently_moving in ROI ──────
                    event_type  = "none"
                    line_index  = -1
                    counted_now = False
                    already     = (track_id in counted_per_cls[class_name])

                    if is_countable(motion.state) and not already:
                        if args.count_mode in ("line", "multiline"):
                            centre = cx if args.line == "vertical" else cy
                            for i, lc in enumerate(line_coords):
                                direction = check_line_crossing(
                                    prev_line_coord.get(track_id), centre, lc, args.line
                                )
                                if direction:
                                    counted_per_cls[class_name].add(track_id)
                                    counts[class_name] += 1
                                    counted_now = True
                                    event_type  = "line_crossed"
                                    line_index  = i
                                    print(
                                        f"  ✓  {class_name:<12} ID:{track_id:>4}  L{i+1}  "
                                        f"dir={direction}  "
                                        f"avg_motion={motion.avg_per_frame:.1f}px  "
                                        f"state={motion.state}  "
                                        f"total={counts[class_name]}  "
                                        f"(frame {frame_no})"
                                    )
                                    break
                            prev_line_coord[track_id] = cx if args.line == "vertical" else cy

                        elif args.count_mode == "zone":
                            entered = check_zone_entry(
                                prev_inside.get(track_id), cx, cy, zone_pts
                            )
                            if entered:
                                counted_per_cls[class_name].add(track_id)
                                counts[class_name] += 1
                                counted_now = True
                                event_type  = "zone_entered"
                                print(
                                    f"  ✓  {class_name:<12} ID:{track_id:>4}  zone  "
                                    f"avg_motion={motion.avg_per_frame:.1f}px  "
                                    f"total={counts[class_name]}  "
                                    f"(frame {frame_no})"
                                )
                            prev_inside[track_id] = (
                                cv2.pointPolygonTest(zone_pts, (cx, cy), False) >= 0
                            )

                    # Keep line coord updated for stationary tracks too — avoids
                    # a false crossing event the moment they start moving.
                    elif args.count_mode in ("line", "multiline"):
                        prev_line_coord[track_id] = cx if args.line == "vertical" else cy

                    # ── CSV ───────────────────────────────────────────────────
                    csv_writer.writerow({
                        "frame_no"       : frame_no,
                        "timestamp_sec"  : round(timestamp, 4),
                        "track_id"       : track_id,
                        "class_name"     : class_name,
                        "confidence"     : round(raw_conf, 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "center_x"       : round(cx, 2),
                        "center_y"       : round(cy, 2),
                        "count_mode"     : args.count_mode,
                        "event_type"     : event_type,
                        "line_index"     : line_index,
                        "counted"        : counted_now,
                        "motion_state"   : motion.state,
                        "avg_per_frame"  : round(motion.avg_per_frame, 2),
                        "displacement"   : round(motion.displacement, 2),
                        "moving"         : is_countable(motion.state),
                        "in_roi"         : in_roi,
                    })

            # ── D: Ghost / velocity-predicted boxes for missing tracks ─────────
            if args.track_buffer > 0:
                for tid, snap in track_memory.items():
                    if tid in active_ids:
                        continue
                    age = frame_no - snap.last_frame
                    if age < 1 or age > args.track_buffer:
                        continue
                    if not should_draw(snap.motion_state, args.show_stationary):
                        continue

                    # Velocity prediction or frozen box
                    if args.predict_missing_tracks and age <= args.max_prediction_frames:
                        px1, py1, px2, py2 = predicted_bbox(snap, age)
                    else:
                        px1, py1, px2, py2 = snap.x1, snap.y1, snap.x2, snap.y2

                    draw_box_label(
                        frame, px1, py1, px2, py2,
                        snap.class_name, tid, snap.conf,
                        counted=(tid in counted_per_cls.get(snap.class_name, set())),
                        motion_state=snap.motion_state,
                        predicted=True,
                    )

                # Prune stale entries
                stale = [t for t, s in track_memory.items()
                         if frame_no - s.last_frame > args.track_buffer]
                for t in stale:
                    del track_memory[t]

            # ── HUD & debug frames ─────────────────────────────────────────────
            hud_classes = [c.strip() for c in args.hud_classes.split(",") if c.strip()]
            draw_hud(frame, counts, frame_no, fps, args.count_mode,
                     vehicle_roi is not None, ped_roi is not None,
                     args.compensate_camera_motion,
                     hud_classes)

            if frame_no % 100 == 0:
                pct = frame_no / total_fr * 100 if total_fr else 0.0
                print(f"  … {frame_no}/{total_fr} ({pct:.1f}%)  "
                      f"counted:{sum(counts.values())}  "
                      f"active:{len(active_ids)}  "
                      f"cam_drift:({cam_dx:.1f},{cam_dy:.1f})")

            writer.write(frame)

            if args.save_debug_frames and frame_no % 50 == 0:
                cv2.imwrite(str(debug_dir / f"frame_{frame_no:06d}.jpg"), frame)

            if args.show:
                cv2.imshow("Railroad Crossing  (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Early stop.")
                    break

    finally:
        cap.release()
        writer.release()
        csv_fh.close()
        if args.show:
            cv2.destroyAllWindows()

    # ── Final summary ──────────────────────────────────────────────────────────
    total = sum(counts.values())
    print("\n" + "═" * 62)
    print("  CLASS-BY-CLASS RESULTS  (motion-filtered + ROI)")
    print("  " + "─" * 46)
    for cls, n in counts.items():
        thr = get_threshold(cls, args.motion_threshold_person, args.motion_threshold_vehicle)
        print(f"  {cls:<14}: {n:>4}  (threshold {thr:.0f} px/frame)")
    print("  " + "─" * 46)
    print(f"  {'TOTAL':<14}: {total:>4}")
    print(f"\n  Unique track IDs seen : {len(all_seen_ids)}")
    print(f"  Frames processed      : {frame_no}")
    print(f"  Output video          : {args.output}")
    print(f"  CSV report            : {args.csv}")
    print("═" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(parse_args())
