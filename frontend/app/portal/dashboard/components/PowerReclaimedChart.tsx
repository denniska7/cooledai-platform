"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type DataPoint = { time: string; power: number };

async function fetchMetrics() {
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

export function PowerReclaimedChart() {
  const [data, setData] = useState<DataPoint[]>(() => {
    const now = new Date();
    return Array.from({ length: 12 }, (_, i) => {
      const d = new Date(now.getTime() - (11 - i) * 5 * 60 * 1000);
      return {
        time: `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`,
        power: 80 + Math.random() * 40,
      };
    });
  });

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      const m = await fetchMetrics();
      if (!mounted || !m) return;
      const power = Math.round(70 + m.efficiency_score * 0.4 + (Math.random() - 0.5) * 10);
      const time = new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
      setData((prev) => [...prev.slice(1), { time, power }]);
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            dataKey="time"
            stroke="rgba(255,255,255,0.5)"
            tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 10 }}
            axisLine={{ stroke: "rgba(255,255,255,0.2)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.2)" }}
          />
          <YAxis
            stroke="rgba(255,255,255,0.5)"
            tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 10 }}
            axisLine={{ stroke: "rgba(255,255,255,0.2)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.2)" }}
            domain={["dataMin - 10", "dataMax + 10"]}
            tickFormatter={(v) => `${v} kW`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#000",
              border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: "4px",
            }}
            labelStyle={{ color: "#fff" }}
            formatter={(value) => [`${value ?? 0} kW`, "Power Reclaimed"]}
          />
          <Line
            type="monotone"
            dataKey="power"
            stroke="#FFFFFF"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#00FFCC" }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-white/50 mt-2 text-center">Power Reclaimed (kW) — Last 24h</p>
    </div>
  );
}
