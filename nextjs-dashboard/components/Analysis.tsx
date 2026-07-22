import { SectionLabel } from "./VideoComparison";

const FINDINGS = [
  {
    title: "What Worked",
    items: [
      "YOLOv8m detected pedestrians reliably across variable lighting conditions",
      "ByteTrack maintained stable IDs through most of the 59-second clip",
      "Three-line coverage eliminated missed counts from single-line gaps",
      "EMA smoothing visibly reduced bounding-box jitter between frames",
      "Track buffer filled short occlusion gaps without false counts",
    ],
  },
  {
    title: "Challenges",
    items: [
      "ID switches occur when two people walk very close together",
      "People near frame edges are partially clipped, reducing detection confidence",
      "Shadows cause false-positive detections at conf < 0.3",
      "Long absences from frame assign a new ID on re-entry, risking double-count",
      "Parked cars in adjacent lot triggered false vehicle detections without ROI",
    ],
  },
  {
    title: "Future Improvements",
    items: [
      "Fine-tune YOLOv8 on campus-specific annotated data for better precision",
      "Adopt BOTSort with Re-ID for denser, occluded crowd scenes",
      "Add zone-based counting for entrance/exit monitoring",
      "Deploy on Apple M-series device for real-time live stream analysis",
      "Aggregate hourly traffic data into a persistent dashboard for planners",
    ],
  },
];

export default function Analysis() {
  return (
    <div className="py-4">
      <SectionLabel>Key Findings</SectionLabel>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {FINDINGS.map(({ title, items }) => (
          <div key={title} className="bg-white border border-[#e2e8f0] rounded-xl p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-widest text-[#64748b] border-b border-[#f1f5f9] pb-3 mb-4">
              {title}
            </p>
            <ul className="space-y-3">
              {items.map((item) => (
                <li key={item} className="text-[#334155] text-sm leading-relaxed pl-4 relative before:content-['–'] before:absolute before:left-0 before:text-[#cbd5e1]">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
