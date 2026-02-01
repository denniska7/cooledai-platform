"use client";

interface SystemLogProps {
  entry: string;
  timestamp: number;
  critical?: boolean;
}

export function SystemLog({ entry, timestamp, critical }: SystemLogProps) {
  const timeStr = new Date(timestamp * 1000).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      className={`rounded border p-4 font-mono text-sm ${
        critical
          ? "border-red-500/50 bg-red-500/5"
          : "border-[rgba(255,255,255,0.1)] bg-black"
      }`}
    >
      <div className="flex items-start gap-3">
        <span className="text-white/40 shrink-0 tabular-nums">{timeStr}</span>
        <span className={critical ? "text-red-400" : "text-white/80"}>{entry}</span>
      </div>
    </div>
  );
}
