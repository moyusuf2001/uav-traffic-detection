"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ScatterChart, Scatter, ReferenceLine,
} from "recharts";
import type { DetectionRow } from "@/lib/parseCSV";
import { SectionLabel } from "./VideoComparison";

const ACCENT = "#2563eb";
const CLASS_COLORS: Record<string, string> = {
  person: "#22c55e", bicycle: "#06b6d4", car: "#2563eb",
  motorcycle: "#f97316", bus: "#a855f7", truck: "#ef4444",
};

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-[#e2e8f0] rounded-xl p-5 shadow-sm">
      <p className="text-xs font-semibold text-[#334155] mb-4">{title}</p>
      {children}
    </div>
  );
}

function ClassBreakdown({ rows }: { rows: DetectionRow[] }) {
  const counted = rows.filter((r) => r.counted);
  const map = new Map<string, number>();
  counted.forEach((r) => map.set(r.class_name, (map.get(r.class_name) ?? 0) + 1));
  const data = [...map.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([name, count]) => ({ name, count, fill: CLASS_COLORS[name] ?? ACCENT }));

  return (
    <ChartCard title="Counted Events by Class">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={80} />
          <Tooltip />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => (
              <rect key={i} fill={d.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function DetectionsPerSecond({ rows }: { rows: DetectionRow[] }) {
  const map = new Map<number, number>();
  rows.forEach((r) => {
    const sec = Math.floor(r.timestamp_sec);
    map.set(sec, (map.get(sec) ?? 0) + 1);
  });
  const data = [...map.entries()].sort((a, b) => a[0] - b[0]).map(([sec, n]) => ({ sec, n }));

  return (
    <ChartCard title="Detections per Second">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ left: -10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis dataKey="sec" tick={{ fontSize: 10 }} label={{ value: "sec", position: "insideBottomRight", offset: 0, fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="n" fill={ACCENT} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function CumulativeCount({ rows }: { rows: DetectionRow[] }) {
  const events = rows.filter((r) => r.counted).sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  const data = [{ ts: 0, cum: 0 }, ...events.map((r, i) => ({ ts: r.timestamp_sec, cum: i + 1 }))];

  return (
    <ChartCard title="Cumulative Count Over Time">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ left: -10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis dataKey="ts" tick={{ fontSize: 10 }} label={{ value: "sec", position: "insideBottomRight", offset: 0, fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Line type="stepAfter" dataKey="cum" stroke={ACCENT} strokeWidth={2.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function TrackFrequency({ rows }: { rows: DetectionRow[] }) {
  const countedIds = new Set(rows.filter((r) => r.counted).map((r) => r.track_id));
  const map = new Map<number, number>();
  rows.forEach((r) => map.set(r.track_id, (map.get(r.track_id) ?? 0) + 1));
  const data = [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([id, frames]) => ({ id: String(id), frames, fill: countedIds.has(id) ? ACCENT : "#cbd5e1" }));

  return (
    <ChartCard title="Top 20 Tracks — Frames Detected">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ left: -10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis dataKey="id" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="frames" radius={[2, 2, 0, 0]} fill={ACCENT} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function ConfidenceHistogram({ rows }: { rows: DetectionRow[] }) {
  const bins = 30;
  const counts = new Array(bins).fill(0);
  rows.forEach((r) => {
    const i = Math.min(Math.floor(r.confidence * bins), bins - 1);
    counts[i]++;
  });
  const data = counts.map((n, i) => ({ bin: (i / bins).toFixed(2), n }));
  const mean = (rows.reduce((s, r) => s + r.confidence, 0) / rows.length).toFixed(2);

  return (
    <ChartCard title={`Detection Confidence Distribution (mean ${mean})`}>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ left: -10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis dataKey="bin" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="n" fill={ACCENT} opacity={0.85} radius={[1, 1, 0, 0]} />
          <ReferenceLine x={mean} stroke="#dc2626" strokeDasharray="4 2" />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function SpatialHeatmap({ rows }: { rows: DetectionRow[] }) {
  const sample = rows.filter((_, i) => i % Math.ceil(rows.length / 3000) === 0);
  const data = sample.map((r) => ({ x: r.center_x, y: r.center_y }));

  return (
    <ChartCard title="Detection Position Density">
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ left: -10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis type="number" dataKey="x" tick={{ fontSize: 10 }} name="X" />
          <YAxis type="number" dataKey="y" tick={{ fontSize: 10 }} name="Y" reversed />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter data={data} fill={ACCENT} opacity={0.15} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export default function Charts({ rows }: { rows: DetectionRow[] }) {
  const multiClass = new Set(rows.map((r) => r.class_name)).size > 1;

  return (
    <div>
      <SectionLabel>Analytics</SectionLabel>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {multiClass ? <ClassBreakdown rows={rows} /> : <DetectionsPerSecond rows={rows} />}
        <CumulativeCount rows={rows} />
        {multiClass && <DetectionsPerSecond rows={rows} />}
        <TrackFrequency rows={rows} />
        <ConfidenceHistogram rows={rows} />
      </div>
      <div className="mt-4">
        <SpatialHeatmap rows={rows} />
      </div>
    </div>
  );
}
