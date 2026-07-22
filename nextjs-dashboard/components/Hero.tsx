export default function Hero() {
  return (
    <div className="bg-[#0f172a] text-[#f8fafc] px-12 pt-14 pb-12">
      <p className="text-xs font-semibold tracking-widest uppercase text-[#64748b] mb-3">
        Bridgewater State University &nbsp;·&nbsp; Directed Study &nbsp;·&nbsp; Spring 2026
      </p>
      <h1 className="text-4xl font-light text-[#f1f5f9] leading-snug tracking-tight mb-2">
        A Robust Computer Vision Framework
        <br />
        for{" "}
        <strong className="font-bold">
          Multi-Object and Vehicle Detection in UAV Imagery
        </strong>
      </h1>
      <p className="text-[#94a3b8] text-base mb-6">Faculty Advisor: Dr. Uma Shama</p>
      <div className="flex flex-wrap gap-2 mb-6">
        {["YOLOv8m", "ByteTrack", "Multi-Line Counting", "EMA Smoothing"].map((b) => (
          <span
            key={b}
            className="border border-[#334155] rounded-full px-4 py-1 text-sm font-medium text-[#cbd5e1] tracking-wide"
          >
            {b}
          </span>
        ))}
      </div>
      <p className="text-sm text-[#64748b] tracking-wide">
        Built by&nbsp;
        <strong className="text-[#cbd5e1] font-semibold tracking-wide">
          Muhammad Ovais Yusuf
        </strong>
      </p>
    </div>
  );
}
