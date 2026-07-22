"use client";

function YouTubeEmbed({ url, label }: { url: string; label: string }) {
  const videoId = url.includes("youtu.be/")
    ? url.split("youtu.be/")[1].split("?")[0]
    : new URL(url).searchParams.get("v") ?? "";

  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-widest text-[#94a3b8] mb-2">{label}</p>
      <div className="rounded-xl overflow-hidden border border-[#e2e8f0] aspect-video">
        <iframe
          className="w-full h-full"
          src={`https://www.youtube.com/embed/${videoId}`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    </div>
  );
}

export default function VideoComparison({
  inputUrl,
  outputUrl,
}: {
  inputUrl: string;
  outputUrl: string;
}) {
  return (
    <div>
      <SectionLabel>Video Comparison</SectionLabel>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <YouTubeEmbed url={inputUrl} label="Original Footage" />
        <YouTubeEmbed url={outputUrl} label="Processed Output" />
      </div>
    </div>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-widest text-[#94a3b8] border-b border-[#e2e8f0] pb-2 mb-4 mt-10">
      {children}
    </p>
  );
}
