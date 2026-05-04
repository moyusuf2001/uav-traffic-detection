"""
ROI Picker  —  Click corners on a paused video frame to get polygon coordinates.

Usage:
    python pick_roi.py --video railroad_crossing.mp4
    python pick_roi.py --video railroad_crossing.mp4 --frame 100

Controls:
    Left-click   : add a polygon vertex
    Right-click  : remove the last vertex
    n            : start a new polygon (saves the current one to console)
    s            : print all polygons collected so far
    q / ESC      : quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive polygon-ROI picker.")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--frame", type=int, default=30,
                   help="Frame number to pause on (default: 30)")
    p.add_argument("--out",   default="outputs/roi_preview.jpg",
                   help="Path to save the annotated preview image")
    return p.parse_args()


def grab_frame(video_path: str, frame_no: int):
    """Open the video and seek to the requested frame."""
    if not Path(video_path).exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[ERROR] Could not read frame {frame_no}")
    return frame


# Mutable state shared with the OpenCV mouse callback
state = {
    "current":   [],     # list[(x, y)] – polygon being drawn now
    "polygons":  [],     # list[list[(x, y)]] – all completed polygons
    "redraw":    True,
}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["current"].append((x, y))
        state["redraw"] = True
        print(f"  + vertex  ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if state["current"]:
            removed = state["current"].pop()
            state["redraw"] = True
            print(f"  - removed {removed}")


def fmt_polygon(pts: list[tuple[int, int]]) -> str:
    """Format vertex list as the CLI string the main scripts expect."""
    return ";".join(f"{x},{y}" for x, y in pts)


def render(frame, current, polygons):
    canvas = frame.copy()

    # Draw each completed polygon in a different colour
    palette = [(0, 215, 255), (0, 200, 120), (200, 60, 0), (180, 0, 180)]
    for i, poly in enumerate(polygons):
        if not poly:
            continue
        color = palette[i % len(palette)]
        if len(poly) >= 3:
            import numpy as np
            pts = np.array(poly, dtype="int32").reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        else:
            for j in range(len(poly) - 1):
                cv2.line(canvas, poly[j], poly[j + 1], color, 2)
        for (vx, vy) in poly:
            cv2.circle(canvas, (vx, vy), 5, color, -1)
        cv2.putText(canvas, f"P{i+1}", poly[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Draw the in-progress polygon in white
    if current:
        for j in range(len(current) - 1):
            cv2.line(canvas, current[j], current[j + 1], (255, 255, 255), 2)
        for (vx, vy) in current:
            cv2.circle(canvas, (vx, vy), 5, (255, 255, 255), -1)

    # Help banner
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas,
                "Left-click=add  Right-click=undo  n=next polygon  s=show all  q=quit",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args  = parse_args()
    frame = grab_frame(args.video, args.frame)
    h, w  = frame.shape[:2]
    print(f"\nVideo frame loaded:  {w} x {h}  (frame #{args.frame})\n")

    cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Picker", min(w, 1400), min(h, 900))
    cv2.setMouseCallback("ROI Picker", on_mouse)

    while True:
        if state["redraw"]:
            canvas = render(frame, state["current"], state["polygons"])
            state["redraw"] = False
        cv2.imshow("ROI Picker", canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("n"):
            if len(state["current"]) >= 3:
                state["polygons"].append(state["current"])
                print(f"\n  Polygon P{len(state['polygons'])} saved:")
                print(f'    "{fmt_polygon(state["current"])}"\n')
                state["current"] = []
                state["redraw"]  = True
            else:
                print("  ! Need at least 3 vertices before starting a new polygon.")
        elif key == ord("s"):
            print("\n──────── Polygons collected so far ────────")
            for i, poly in enumerate(state["polygons"]):
                print(f"  P{i+1}:  \"{fmt_polygon(poly)}\"")
            if state["current"]:
                print(f"  In-progress:  \"{fmt_polygon(state['current'])}\"")
            print()

    cv2.destroyAllWindows()

    # Final save + console output
    if state["current"] and len(state["current"]) >= 3:
        state["polygons"].append(state["current"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, render(frame, [], state["polygons"]))
    print(f"\nPreview image saved: {args.out}")

    print("\n══════════ COPY THESE INTO YOUR COMMAND ══════════")
    if not state["polygons"]:
        print("  (no polygons drawn)")
    for i, poly in enumerate(state["polygons"]):
        flag = "--vehicle-roi-points" if i == 0 else \
               "--pedestrian-roi-points" if i == 1 else \
               f"--polygon-{i + 1}"
        print(f'  {flag}  "{fmt_polygon(poly)}"')
    print("═══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
