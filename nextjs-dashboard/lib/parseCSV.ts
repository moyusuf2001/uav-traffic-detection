import Papa from "papaparse";

export interface DetectionRow {
  frame_no: number;
  timestamp_sec: number;
  track_id: number;
  class_name: string;
  confidence: number;
  x1: number; y1: number; x2: number; y2: number;
  center_x: number; center_y: number;
  count_mode: string;
  event_type: string;
  line_index: number;
  counted: boolean;
  motion_state?: string;
  avg_per_frame?: number;
  displacement?: number;
  moving?: boolean;
  in_roi?: boolean;
}

export interface Metrics {
  counted: number;
  confirmedTracks: number;
  rawTracks: number;
  duration: string;
  fps: string;
  model: string;
  tracker: string;
  method: string;
  avgConfidence: string;
}

export async function loadCSV(path: string): Promise<DetectionRow[]> {
  const res = await fetch(path);
  const text = await res.text();
  const result = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });
  return result.data.map((r) => ({
    ...r,
    counted: String(r.counted).toLowerCase() === "true",
    timestamp_sec: Number(r.timestamp_sec) || 0,
  })) as unknown as DetectionRow[];
}

export function computeMetrics(
  rows: DetectionRow[],
  cfg: { model: string; tracker: string; method: string }
): Metrics {
  const maxTs = Math.max(...rows.map((r) => r.timestamp_sec));
  const m = Math.floor(maxTs / 60);
  const s = Math.floor(maxTs % 60);
  const fps = maxTs > 0 ? (Math.max(...rows.map((r) => r.frame_no)) / maxTs).toFixed(1) : "0";

  const trackLengths = new Map<number, number>();
  rows.forEach((r) => trackLengths.set(r.track_id, (trackLengths.get(r.track_id) ?? 0) + 1));

  const confirmedTracks = [...trackLengths.values()].filter((n) => n >= 30).length;
  const rawTracks = trackLengths.size;
  const counted = rows.filter((r) => r.counted).length;
  const avgConf = (rows.reduce((s, r) => s + r.confidence, 0) / rows.length).toFixed(2);

  return {
    counted,
    confirmedTracks,
    rawTracks,
    duration: `${m}:${String(s).padStart(2, "0")}`,
    fps,
    model: cfg.model,
    tracker: cfg.tracker,
    method: cfg.method,
    avgConfidence: avgConf,
  };
}
