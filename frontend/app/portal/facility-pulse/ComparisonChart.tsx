"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { api } from "@/lib/api";
import { useAuth } from "@clerk/nextjs";

type RangeKey = "1h" | "6h" | "24h" | "7d";
const RANGE_HOURS: Record<RangeKey, number> = { "1h": 1, "6h": 6, "24h": 24, "7d": 168 };

type MetricKey = "gpu" | "cpu" | "fan" | "power" | "peak";

const METRIC_CONFIG: Record<MetricKey, {
  label: string;
  pilotKey: string;
  baselineKey: string;
  unit: string;
}> = {
  gpu: { label: "GPU Temp", pilotKey: "pilot", baselineKey: "baseline", unit: "\u00B0C" },
  cpu: { label: "CPU Temp", pilotKey: "pilot_cpu_temp", baselineKey: "baseline_cpu_temp", unit: "\u00B0C" },
  fan: { label: "Fan RPM", pilotKey: "pilot_fan_rpm", baselineKey: "baseline_fan_rpm", unit: " RPM" },
  power: { label: "GPU Power", pilotKey: "pilot_gpu_power_w", baselineKey: "baseline_gpu_power_w", unit: "W" },
  peak: { label: "Peak GPU Power", pilotKey: "pilot_peak_gpu_power_w", baselineKey: "baseline_peak_gpu_power_w", unit: "W" },
};

function downsample<T>(data: T[], maxPts: number): T[] {
  if (data.length <= maxPts) return data;
  const step = Math.ceil(data.length / maxPts);
  return data.filter((_, i) => i % step === 0);
}

export function ComparisonChart() {
  const { getToken } = useAuth();
  const [range, setRange] = useState<RangeKey>("1h");
  const [metric, setMetric] = useState<MetricKey>("gpu");
  const [data, setData] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async (hours: number) => {
    try {
      const token = await getToken();
      const res = await api.getThermalHistoryRaw(hours, token);
      if (!mountedRef.current) return;
      if (res.ok) {
        const json = await res.json();
        const points = (json.points || []).filter(
          (p: Record<string, unknown>) => p.pilot && p.pilot !== 0
        );
        setData(points.map((p: Record<string, unknown>) => ({
          ...p,
          ts: Math.round((p.ts as number) * 1000),
        })));
      }
    } catch { /* ignore */ }
    finally { if (mountedRef.current) setLoading(false); }
  }, [getToken]);

  useEffect(() => {
    mountedRef.current = true;
    setLoading(true);
    fetchData(RANGE_HOURS[range]);
    const id = setInterval(() => fetchData(RANGE_HOURS[range]), 30_000);
    return () => { mountedRef.current = false; clearInterval(id); };
  }, [range, fetchData]);

  const displayData = useMemo(() => {
    const maxPts = range === "7d" ? 500 : range === "24h" ? 360 : data.length;
    return downsample(data, maxPts);
  }, [data, range]);

  const cfg = METRIC_CONFIG[metric];

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base font-semibold text-white">Multi-Metric Comparison</h3>
        <div className="flex gap-1">
          {(Object.keys(RANGE_HOURS) as RangeKey[]).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                range === r
                  ? "bg-white/10 text-white"
                  : "text-white/40 hover:text-white/70"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Metric tabs */}
      <div className="flex gap-1 mb-4">
        {(Object.keys(METRIC_CONFIG) as MetricKey[]).map((m) => (
          <button
            key={m}
            onClick={() => setMetric(m)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              metric === m
                ? "bg-[#00FFCC]/10 text-[#00FFCC] border border-[#00FFCC]/30"
                : "text-white/50 hover:text-white/70 border border-transparent"
            }`}
          >
            {METRIC_CONFIG[m].label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="h-[300px]">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-white/20 border-t-[#00FFCC] rounded-full animate-spin" />
          </div>
        ) : displayData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-white/40">
            Collecting thermal data...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="ts"
                tickFormatter={formatTime}
                stroke="rgba(255,255,255,0.2)"
                fontSize={10}
                minTickGap={40}
              />
              <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid rgba(255,255,255,0.1)" }}
                labelFormatter={formatTime}
                formatter={(v: number) => [`${v?.toFixed?.(1) ?? v}${cfg.unit}`, ""]}
              />
              <Line
                type="monotone"
                dataKey={cfg.pilotKey}
                stroke="#22c55e"
                dot={false}
                strokeWidth={2}
                name="CooledAI (Pilot)"
                connectNulls
              />
              <Line
                type="monotone"
                dataKey={cfg.baselineKey}
                stroke="#ef4444"
                dot={false}
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Control"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
