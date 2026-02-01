"use client";

interface SystemLogProps {
  entry: string;
  timestamp: number;
}

export function SystemLog({ entry, timestamp }: SystemLogProps) {
  const timeStr = new Date(timestamp * 1000).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-4 font-mono text-sm">
      <div className="flex items-start gap-3">
        <span className="text-white/40 shrink-0 tabular-nums">{timeStr}</span>
        <span className="text-white/80">{entry}</span>
      </div>
    </div>
  );
}
