import { SectionLabel } from "./VideoComparison";

const METHODS = [
  {
    tag: "Detection",
    title: "YOLOv8 — Object Detection",
    body: "YOLOv8m (medium) runs inference on every decoded frame and returns bounding boxes for all target classes. Confidence threshold: 0.35 for pedestrians · 0.15 for vehicles. IoU threshold: 0.45.",
  },
  {
    tag: "Tracking",
    title: "ByteTrack — Multi-Frame Association",
    body: "ByteTrack maintains a persistent unique ID for each object across frames using a Kalman filter for motion prediction and the Hungarian algorithm for assignment. It leverages every detection — including low-confidence ones — to reduce ID switches.",
  },
  {
    tag: "Counting",
    title: "Multi-Line Crossing Detection",
    body: "Three vertical counting lines at 25%, 50%, and 75% of the frame width ensure every path is covered. An object is counted exactly once when its bounding-box centre crosses any line for the first time.",
  },
  {
    tag: "ROI Filtering",
    title: "Class-Specific Region of Interest",
    body: "A tight polygon covers the active roadway only. Vehicles outside this zone (e.g. parked cars in an adjacent lot) are discarded before tracking. A separate pedestrian ROI can cover a sidewalk or crossing zone.",
  },
  {
    tag: "Motion Analysis",
    title: "Sustained-Motion Classifier",
    body: "Each track is classified as MOVING, RECENTLY_MOVING, STATIONARY, or INSUFFICIENT_DATA based on the average per-frame displacement over a rolling 12-frame window. Stationary objects are never counted.",
  },
  {
    tag: "Smoothing",
    title: "Exponential Moving Average (EMA) on Bounding Boxes",
    body: "Raw detections jitter between frames. Each box coordinate is smoothed using: smooth_t = 0.75 × smooth_{t−1} + 0.25 × raw_t. This eliminates high-frequency noise while preserving real movement.",
  },
];

export default function Methodology() {
  return (
    <div className="py-4">
      <SectionLabel>Pipeline Overview</SectionLabel>
      <div className="space-y-3">
        {METHODS.map(({ tag, title, body }) => (
          <div key={tag} className="bg-white border border-[#e2e8f0] rounded-xl p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-widest text-[#2563eb] mb-1">{tag}</p>
            <p className="text-lg font-semibold text-[#0f172a] mb-2">{title}</p>
            <p className="text-[#475569] leading-relaxed text-sm">{body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
