# Campus Pedestrian Detection, Tracking & Counting

> Two scripts for two camera angles — see the table below for which to use.

| Script | Camera | Classes detected | Default conf |
|---|---|---|---|
| `main.py` | University Park | person only | 0.35 |
| `railroad_crossing.py` | Railroad Crossing | person, car, truck, bus, motorcycle, bicycle | 0.15 |

---

## What's new in v2

| Feature | Description |
|---|---|
| **Zone counting** | Define a walkway polygon; count anyone who enters it |
| **Multi-line counting** | Place 2–5 lines across the frame; count once per person regardless of which line they cross |
| **EMA box smoothing** | Exponential moving average flattens jitter between frames |
| **Track memory** | Ghost boxes keep the last known position visible for N frames when the detector briefly loses someone |

All original `--line` arguments still work unchanged.

---

## Project Structure

```
Directed Study Project 2/
├── main.py               ← full detection → tracking → counting pipeline
├── requirements.txt      ← Python dependencies
├── README.md             ← this file
├── 1.mp4                 ← campus video
└── outputs/
    ├── output.mp4        ← annotated video (created on run)
    └── people_counts.csv ← per-detection CSV report
```

---

## 1. Installation

### Step 1 – Create a virtual environment

```bash
cd "Directed Study Project 2"
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 2 – Install PyTorch

**Apple Silicon (M1 / M2 / M3)** – MPS is used automatically:

```bash
pip install torch torchvision
```

**Linux / Windows + CUDA 12.1:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 – Install remaining dependencies

```bash
pip install -r requirements.txt
```

YOLOv8 model weights are downloaded automatically on first run.

---

## 2. Running the project

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | `1.mp4` | Input video path |
| `--model` | `yolov8n.pt` | YOLOv8 weights |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--output` | `outputs/output.mp4` | Annotated video path |
| `--csv` | `outputs/people_counts.csv` | CSV report path |
| `--show` | *(flag)* | Open live preview window (press **Q** to stop) |
| `--conf` | `0.3` | Minimum detection confidence |
| `--iou` | `0.5` | NMS IoU threshold |
| `--count-mode` | `line` | `line` \| `zone` \| `multiline` |
| `--line` | `vertical` | Line orientation: `vertical` \| `horizontal` |
| `--line-pos` | `0.5` | Single-line position (fraction 0–1) |
| `--line-positions` | *(none)* | Multiline positions, e.g. `0.25,0.42,0.70` |
| `--zone-points` | *(none)* | Polygon vertices, e.g. `100,300;600,250;...` |
| `--smooth-boxes` | *(flag)* | Enable EMA box smoothing |
| `--smooth-alpha` | `0.6` | EMA weight on previous box (0 = no smoothing, 1 = frozen) |
| `--track-buffer` | `10` | Frames to keep ghost box after track disappears (`0` = off) |

---

## 3. Command examples

### Original single-line counting (fully backward-compatible)

```bash
python main.py \
  --video 1.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --line vertical \
  --line-pos 0.42
```

### Smoother boxes + track memory

```bash
python main.py \
  --video 1.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --line vertical \
  --line-pos 0.42 \
  --smooth-boxes \
  --smooth-alpha 0.6 \
  --track-buffer 10
```

### Multiple counting lines

```bash
python main.py \
  --video 1.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode multiline \
  --line vertical \
  --line-positions 0.25,0.42,0.70 \
  --smooth-boxes
```

> People who take different routes and walk past any of the three lines are
> all counted. The same person is **never** counted twice even if they cross
> more than one line.

### Zone-based counting

```bash
python main.py \
  --video 1.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode zone \
  --zone-points "100,300;600,250;1050,350;1000,650;200,650" \
  --smooth-boxes
```

> Replace the x,y pairs with coordinates that match your video's walkway or
> entrance. Use any image viewer that shows pixel coordinates to find the
> corners.

---

## 4. How to change the counting line position

`--line-pos` is a **fraction** (0.0 – 1.0) of the frame width (vertical) or
frame height (horizontal):

| Command | Effect |
|---|---|
| `--line vertical --line-pos 0.5` | Centre of frame |
| `--line vertical --line-pos 0.3` | 30 % from left edge |
| `--line vertical --line-pos 0.7` | 70 % from left edge |
| `--line horizontal --line-pos 0.25` | 25 % from top |

For **multiline** mode use `--line-positions` instead:

```bash
--count-mode multiline --line vertical --line-positions 0.25,0.42,0.70
```

---

## 5. ByteTrack vs BOTSort

```bash
# ByteTrack – default, faster, recommended for most scenes
python main.py --tracker bytetrack.yaml

# BOTSort – uses appearance (Re-ID) features; better when people overlap
python main.py --tracker botsort.yaml
```

| Tracker | Best for |
|---|---|
| **ByteTrack** | Speed, crowded scenes, real-time use |
| **BOTSort** | Heavy occlusion, people walking very close together |

---

## 6. Box smoothing explained

### `--smooth-boxes` + `--smooth-alpha`

Each bounding box coordinate is updated using an exponential moving average:

```
smooth_t = alpha × smooth_{t-1}  +  (1 - alpha) × raw_t
```

| `--smooth-alpha` | Effect |
|---|---|
| `0.0` | No smoothing (same as raw detection) |
| `0.4` | Light smoothing – faster response |
| `0.6` | **Default** – good balance |
| `0.8` | Heavy smoothing – very stable, but lags behind fast movement |

### `--track-buffer`

When the detector misses a person for a few frames (common near the edge of
the frame or under occlusion), a **grey "predicted" box** is drawn at the last
known position. This prevents boxes from popping in and out.

```bash
--track-buffer 10    # keep ghost box for up to 10 frames (default)
--track-buffer 0     # disable ghost boxes entirely
--track-buffer 20    # keep longer – good for slow-moving cameras
```

> **Note:** predicted/ghost boxes are **never** used for counting. A count
> is only triggered by a real confirmed detection crossing the line or entering
> the zone.

---

## 7. Zone counting – how to get polygon coordinates

1. Open the video in QuickTime or VLC and pause on a representative frame.
2. Take a screenshot and open it in Preview (macOS) or Paint.
3. Hover over the corners of the walkway/entrance area and note the x,y pixel
   positions.
4. Pass them as `--zone-points "x1,y1;x2,y2;x3,y3;..."` (at least 3 points,
   clockwise or counter-clockwise order).

Example for a doorway spanning the lower half of a 1280×720 frame:

```bash
--zone-points "200,400;1080,400;1080,720;200,720"
```

---

## 8. Improving accuracy

### Use a larger model

```bash
python main.py --model yolov8m.pt    # medium  – good accuracy/speed balance
python main.py --model yolov8l.pt    # large
python main.py --model yolov8x.pt    # extra-large – highest accuracy
```

### Fine-tune on campus footage

1. Annotate pedestrian clips with [Roboflow](https://roboflow.com) or [Label Studio](https://labelstud.io).
2. Train:

```bash
yolo train model=yolov8n.pt data=campus.yaml epochs=50 imgsz=640
```

3. Point to your custom weights:

```bash
python main.py --model runs/detect/train/weights/best.pt
```

### Tune detection thresholds

```bash
# Lower confidence to catch more people (may add false positives)
python main.py --conf 0.25

# Higher confidence for cleaner detections only
python main.py --conf 0.45 --iou 0.4
```

---

## 9. CSV report columns

| Column | Description |
|---|---|
| `frame_no` | Sequential frame number |
| `timestamp_sec` | Seconds from video start |
| `track_id` | Unique tracker-assigned ID |
| `class_name` | Always `person` |
| `confidence` | YOLOv8 confidence score (0–1) |
| `x1, y1, x2, y2` | Bounding box corners (after smoothing if enabled) |
| `center_x, center_y` | Centre point of the box |
| `count_mode` | `line` \| `zone` \| `multiline` |
| `event_type` | `line_crossed` \| `zone_entered` \| `none` |
| `line_index` | Index (0-based) of which line was crossed; `-1` for zone/none |
| `counted` | `True` only on the frame that incremented the total count |

---

## 10. Visual annotation guide

| Element | Meaning |
|---|---|
| Green box | Detected person, not yet counted |
| Orange box | Person already counted |
| Grey dashed box | Track-memory ghost (detector temporarily lost this person) |
| Red dashed line | Single counting line |
| Coloured dashed lines | Multiple counting lines (each a different colour) |
| Gold polygon border + yellow fill | Zone counting area |
| Top-left HUD | Live count, count mode, frame number, source FPS |

---

## Requirements

- Python ≥ 3.9
- macOS 12.3+ with Apple Silicon **or** any Linux/Windows system
- No dedicated GPU required (MPS / CPU selected automatically)

---

## Railroad Crossing Script  (`railroad_crossing.py`)

Detects, tracks, and counts **both pedestrians and vehicles** at the railroad
crossing camera angle. The pipeline is identical to `main.py` in structure but
adds multi-class support, per-class counting, and higher-resolution inference.

### COCO class IDs used

| Class ID | Label |
|---|---|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

### Default settings (differ from `main.py`)

| Setting | `main.py` | `railroad_crossing.py` | Why |
|---|---|---|---|
| `--imgsz` | 640 | **1280** | Wider scene; vehicles at distance need more resolution |
| `--conf` | 0.35 | **0.15** | Vehicles & bikes at range have lower confidence |
| `--count-mode` | line | **multiline** | Different classes travel different paths |

### Example commands

**Basic single-line run:**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode line \
  --line vertical \
  --line-pos 0.5
```

**Best overall command — three lines + smoothing + motion filter:**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode multiline \
  --line vertical \
  --line-positions 0.25,0.5,0.75 \
  --smooth-boxes \
  --smooth-alpha 0.6 \
  --track-buffer 10 \
  --motion-threshold 25 \
  --motion-window 8 \
  --min-active-frames 5 \
  --output outputs/railroad_crossing_output.mp4 \
  --csv outputs/railroad_crossing_results.csv
```

**Only moving traffic (ignore parked cars):**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode multiline \
  --line vertical \
  --line-positions 0.25,0.5,0.75 \
  --motion-threshold 25 \
  --motion-window 8 \
  --min-active-frames 5 \
  --smooth-boxes
```

**Debug mode — see parked cars highlighted in grey:**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode multiline \
  --line vertical \
  --line-positions 0.25,0.5,0.75 \
  --motion-threshold 25 \
  --show-stationary \
  --show
```

**Stricter filtering (larger displacement required):**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --motion-threshold 40 \
  --motion-window 12
```

**Looser filtering (useful for slow pedestrians):**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --motion-threshold 15 \
  --motion-window 6
```

**Zone mode — for a specific crossing area:**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode zone \
  --zone-points "100,300;500,250;1100,280;1200,600;200,650" \
  --smooth-boxes \
  --output outputs/railroad_crossing_output.mp4 \
  --csv outputs/railroad_crossing_results.csv
```

**Live preview window:**
```bash
python railroad_crossing.py --video railroad_crossing.mp4 --show
```

### Per-class terminal summary

At the end of every run the script prints a breakdown like:

```
══════════════════════════════════════════════════════
  CLASS-BY-CLASS RESULTS
  ────────────────────────────────────────
  person        :   42  counted
  bicycle       :    3  counted
  car           :   18  counted
  motorcycle    :    2  counted
  bus           :    1  counted
  truck         :    7  counted
  ────────────────────────────────────────
  TOTAL         :   73

  Unique track IDs seen : 89
  Frames processed      : 1800
  Output video          : outputs/railroad_crossing_output.mp4
  CSV report            : outputs/railroad_crossing_results.csv
══════════════════════════════════════════════════════
```

### Bounding box colour scheme

| Class | Box colour |
|---|---|
| person | Green |
| bicycle | Cyan |
| car | Blue |
| motorcycle | Orange |
| bus | Magenta |
| truck | Red |

### Best production command (v4 — tight class-specific ROIs)

Use **two separate polygons**: a tight road-only polygon for vehicles, and an
optional crossing/sidewalk polygon for pedestrians.  The upper parking lot is
simply outside both polygons and is never processed.

```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 \
  --model yolov8m.pt \
  --tracker bytetrack.yaml \
  --count-mode multiline --line vertical --line-positions 0.25,0.5,0.75 \
  --vehicle-roi-points "300,400;1400,380;1500,850;200,870" \
  --pedestrian-roi-points "100,350;600,330;620,900;80,900" \
  --use-separate-rois \
  --smooth-boxes --smooth-alpha 0.75 \
  --motion-window 12 \
  --motion-threshold-vehicle 20 --motion-threshold-person 10 \
  --stationary-confirm-frames 12 --recently-moving-buffer 45 \
  --track-buffer 15 --predict-missing-tracks --max-prediction-frames 8 \
  --compensate-camera-motion \
  --output outputs/railroad_crossing_output.mp4 \
  --csv outputs/railroad_crossing_results.csv
```

> Adjust the polygon coordinates to match your specific video frame.
> Use `--show-motion-values --show` on the first run to see what is being detected where.

**Clean output (no ROI polygons drawn on video — for presentation):**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 --model yolov8m.pt \
  --vehicle-roi-points "300,400;1400,380;1500,850;200,870" \
  --use-separate-rois --hide-roi-overlay \
  --smooth-boxes --smooth-alpha 0.75 \
  --motion-window 12 --motion-threshold-vehicle 20
```

**Vehicle-only ROI (shared for all classes, no separate pedestrian zone):**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 --model yolov8m.pt \
  --vehicle-roi-points "300,400;1400,380;1500,850;200,870" \
  --smooth-boxes --motion-threshold-vehicle 20
```

**Debug — see parked cars and motion values:**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 --model yolov8m.pt \
  --vehicle-roi-points "300,400;1400,380;1500,850;200,870" \
  --motion-window 12 --motion-threshold-vehicle 20 \
  --show-stationary --show-motion-values --show
```

**Zone mode (count only objects entering the crossing zone):**
```bash
python railroad_crossing.py \
  --video railroad_crossing.mp4 --model yolov8m.pt \
  --count-mode zone \
  --zone-points "400,420;1200,400;1250,780;350,800" \
  --vehicle-roi-points "300,400;1400,380;1500,850;200,870" \
  --use-separate-rois \
  --pedestrian-roi-points "100,350;600,330;620,900;80,900" \
  --smooth-boxes --smooth-alpha 0.75
```

### All motion + ROI arguments

| Argument | Default | Description |
|---|---|---|
| `--vehicle-roi-points` | *(none)* | Tight road polygon for car/truck/bus/motorcycle/bicycle |
| `--pedestrian-roi-points` | *(none)* | Optional separate polygon for persons |
| `--use-separate-rois` | *(flag)* | Apply each ROI only to its class; without it vehicle ROI covers all |
| `--hide-roi-overlay` | *(flag)* | Don't draw ROI polygons in the output video |
| `--motion-window` | `12` | Rolling window of frames for motion analysis |
| `--motion-threshold-person` | `10` | Avg px/frame threshold for persons |
| `--motion-threshold-vehicle` | `20` | Avg px/frame threshold for vehicles |
| `--stationary-confirm-frames` | `12` | Consecutive below-threshold frames before STATIONARY |
| `--recently-moving-buffer` | `45` | Frames to stay RECENTLY_MOVING after last movement |
| `--min-active-frames` | `5` | Frames before classification begins |
| `--compensate-camera-motion` | *(flag)* | Subtract drone drift from track motion via optical flow |
| `--predict-missing-tracks` | *(flag)* | Advance ghost boxes with velocity estimate |
| `--max-prediction-frames` | `8` | Max frames to velocity-predict a missing track |
| `--smooth-alpha` | `0.75` | EMA weight on previous box (stronger than v2) |
| `--show-stationary` | *(flag)* | Draw parked objects in grey for debugging |
| `--show-motion-values` | *(flag)* | Overlay avg px/frame + state on each box |
| `--save-debug-frames` | *(flag)* | Save every 50th frame to `outputs/debug_frames/` |

### Motion states

| State | Meaning | Drawn? | Counted? | Box style |
|---|---|---|---|---|
| `moving` | Avg motion ≥ threshold | Yes | Yes | Full class colour |
| `recently_moving` | Paused within buffer (e.g. red light) | Yes | Yes | 45% lighter class colour + "(slow)" |
| `stationary` | Confirmed parked / background | Only with `--show-stationary` | Never | Dark grey + "(parked)" |
| `insufficient_data` | New track, not enough frames | Yes | No | Grey + "(new)" |

### Integrating the CSV into the Streamlit dashboard

1. Run the script and confirm the CSV is at `outputs/railroad_crossing_results.csv`.
2. Copy output files into the data directory:
   ```bash
   cp railroad_crossing.mp4                          data/railroad_crossing/input.mp4
   cp outputs/railroad_crossing_output.mp4           data/railroad_crossing/output.mp4
   cp outputs/railroad_crossing_results.csv          data/railroad_crossing/results.csv
   ```
3. Open `app.py` and set `"ready": True` in the `LOCATIONS["Railroad Crossing"]` entry.
4. Refresh the browser — the Railroad Crossing tab activates automatically.

> **Note:** The dashboard currently groups all classes together for the total
> count metric. To show per-class breakdowns in the UI, filter by `class_name`
> column in the CSV preview, or add a grouped bar chart using
> `df.groupby("class_name")["counted"].sum()`.
