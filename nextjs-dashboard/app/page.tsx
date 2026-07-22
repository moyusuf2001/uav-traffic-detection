"use client";

import { useEffect, useState } from "react";
import Hero from "@/components/Hero";
import MetricsGrid from "@/components/MetricsGrid";
import VideoComparison from "@/components/VideoComparison";
import Charts from "@/components/Charts";
import DataTable from "@/components/DataTable";
import Analysis from "@/components/Analysis";
import Methodology from "@/components/Methodology";
import { SectionLabel } from "@/components/VideoComparison";
import { loadCSV, computeMetrics, type DetectionRow, type Metrics } from "@/lib/parseCSV";

const LOCATIONS = {
  "University Park": {
    csv: "/university_park.csv",
    inputUrl: "https://youtu.be/c2A0UmQJAGg",
    outputUrl: "https://youtu.be/38EDFQcwma8",
    model: "YOLOv8m",
    tracker: "ByteTrack",
    method: "Multi-Line · 3 Vertical Lines",
    slug: "university_park",
  },
  "Railroad Crossing": {
    csv: "/railroad_crossing.csv",
    inputUrl: "https://youtu.be/TriC4MtOLMU",
    outputUrl: "https://youtu.be/WCw7U15TSQo",
    model: "YOLOv8l",
    tracker: "BOTSort",
    method: "Multi-Line · 3 Vertical Lines",
    slug: "railroad_crossing",
  },
} as const;

type LocationKey = keyof typeof LOCATIONS;
type Tab = LocationKey | "Analysis" | "Methodology";

const TABS: Tab[] = ["University Park", "Railroad Crossing", "Analysis", "Methodology"];

function LocationTab({ locKey }: { locKey: LocationKey }) {
  const cfg = LOCATIONS[locKey];
  const [rows, setRows] = useState<DetectionRow[] | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    loadCSV(cfg.csv).then((data) => {
      setRows(data);
      setMetrics(computeMetrics(data, { model: cfg.model, tracker: cfg.tracker, method: cfg.method }));
    });
  }, [cfg]);

  if (!rows || !metrics) {
    return (
      <div className="flex items-center justify-center py-24 text-[#94a3b8] text-sm">
        Loading data…
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <VideoComparison inputUrl={cfg.inputUrl} outputUrl={cfg.outputUrl} />
      <div>
        <SectionLabel>Key Metrics</SectionLabel>
        <MetricsGrid m={metrics} />
      </div>
      <Charts rows={rows} />
      <DataTable rows={rows} slug={cfg.slug} />
    </div>
  );
}

export default function Home() {
  const [active, setActive] = useState<Tab>("University Park");

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <Hero />

      {/* Tab bar */}
      <div className="bg-[#0f172a] px-12 flex gap-0">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
              active === tab
                ? "text-[#f1f5f9] border-[#2563eb] font-semibold"
                : "text-[#64748b] border-transparent hover:text-[#94a3b8]"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-8 py-8">
        {active === "University Park"   && <LocationTab locKey="University Park" />}
        {active === "Railroad Crossing" && <LocationTab locKey="Railroad Crossing" />}
        {active === "Analysis"          && <Analysis />}
        {active === "Methodology"       && <Methodology />}
      </div>
    </div>
  );
}
