"use client";

import { useState } from "react";
import type { DetectionRow } from "@/lib/parseCSV";
import { SectionLabel } from "./VideoComparison";

export default function DataTable({ rows, slug }: { rows: DetectionRow[]; slug: string }) {
  const [eventsOnly, setEventsOnly] = useState(false);
  const display = eventsOnly ? rows.filter((r) => r.counted) : rows;
  const shown = display.slice(0, 200);

  const headers = ["frame_no", "timestamp_sec", "track_id", "class_name", "confidence",
    "center_x", "center_y", "counted", "event_type"];

  function downloadCSV() {
    const csvContent = [headers.join(","),
      ...display.map((r) =>
        headers.map((h) => (r as unknown as Record<string, unknown>)[h] ?? "").join(",")
      ),
    ].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${slug}_results.csv`;
    a.click();
  }

  return (
    <div>
      <SectionLabel>Detection Log</SectionLabel>
      <div className="flex items-center justify-between mb-3">
        <label className="flex items-center gap-2 text-sm text-[#475569] cursor-pointer">
          <input
            type="checkbox"
            checked={eventsOnly}
            onChange={(e) => setEventsOnly(e.target.checked)}
            className="rounded"
          />
          Count events only
        </label>
        <button
          onClick={downloadCSV}
          className="bg-[#0f172a] text-[#f1f5f9] text-sm font-medium px-5 py-2 rounded-lg hover:bg-[#1e293b] transition-colors"
        >
          Download CSV
        </button>
      </div>
      <div className="overflow-auto rounded-xl border border-[#e2e8f0] max-h-72 text-sm">
        <table className="min-w-full">
          <thead className="bg-[#f8fafc] sticky top-0">
            <tr>
              {headers.map((h) => (
                <th key={h} className="px-3 py-2 text-left text-xs font-semibold text-[#64748b] uppercase tracking-wide whitespace-nowrap border-b border-[#e2e8f0]">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-[#f8fafc]"}>
                {headers.map((h) => (
                  <td key={h} className="px-3 py-1.5 text-[#334155] whitespace-nowrap border-b border-[#f1f5f9]">
                    {String((row as unknown as Record<string, unknown>)[h] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-[#94a3b8] mt-2">Showing {shown.length} of {display.length} rows</p>
    </div>
  );
}
