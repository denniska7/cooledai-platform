"use client";

import { useEffect, useState } from "react";

type Metrics = {
  efficiency_score: number;
  thermal_lag_seconds: number;
  state: string;
  nodes_active: number;
  cooling_delta: number;
} | null;

async function fetchMetrics(): Promise<Metrics> {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) return null;
  try {
    const base = url.replace(/\/$/, "");
    const res = await fetch(`${base}/simulated-metrics`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export function SystemPulse() {
  const [metrics, setMetrics] = useState<Metrics>(null);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      const data = await fetchMetrics();
      if (mounted) setMetrics(data);
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  const bars = metrics
    ? [
        Math.min(100, Math.max(0, metrics.efficiency_score)),
        Math.min(100, 100 - metrics.thermal_lag_seconds * 10),
        Math.min(100, metrics.nodes_active * 4),
      ]
    : [60, 70, 80];

  return (
    <div className="rounded border border-white/10 bg-white/[0.02] p-6">
      <div className="mb-4 flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-accent-lime animate-pulse" />
        <span className="text-xs font-medium uppercase tracking-widest text-white/60">
          System Pulse
        </span>
      </div>
      <div className="flex h-20 items-end gap-2">
        {bars.map((h, i) => (
          <div
            key={i}
            className="w-4 rounded-sm bg-accent-lime/30 transition-all duration-500"
            style={{ height: `${h}%`, minHeight: "8px" }}
          />
        ))}
      </div>
      {metrics && (
        <div className="mt-4 grid grid-cols-3 gap-4 text-xs text-white/50">
          <div>
            <span className="text-accent-lime">{metrics.efficiency_score.toFixed(1)}%</span> efficiency
          </div>
          <div>
            <span className="text-white/70">{metrics.nodes_active}</span> nodes
          </div>
          <div>
            <span className="text-white/70">{metrics.state}</span>
          </div>
        </div>
      )}
    </div>
  );
}
