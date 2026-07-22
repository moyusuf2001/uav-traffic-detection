import type { Metrics } from "@/lib/parseCSV";

function Card({ value, label, large }: { value: string | number; label: string; large?: boolean }) {
  return (
    <div className="bg-white border border-[#e2e8f0] rounded-xl p-6 text-center shadow-sm">
      <div className={`font-bold text-[#0f172a] leading-none mb-2 ${large ? "text-4xl" : "text-2xl"}`}>
        {value}
      </div>
      <div className="text-xs font-semibold uppercase tracking-widest text-[#94a3b8]">{label}</div>
    </div>
  );
}

export default function MetricsGrid({ m }: { m: Metrics }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card value={m.counted}          label="People Counted"          large />
        <Card value={m.confirmedTracks}  label="Confirmed Tracks (≥1s)"  large />
        <Card value={m.duration}         label="Video Duration"           large />
        <Card value={m.fps}              label="Source FPS"               large />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card value={m.model}           label="Model" />
        <Card value={m.tracker}         label="Tracker" />
        <Card value={m.method}          label="Counting Method" />
        <Card value={m.avgConfidence}   label="Avg. Confidence" />
      </div>
    </div>
  );
}
