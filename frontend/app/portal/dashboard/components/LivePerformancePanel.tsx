"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api } from "@/lib/api";

type TimeRange = "30 min" | "2h" | "24h";

const RANGE_HOURS: Record<TimeRange, number> = {
  "30 min": 1,
  "2h": 2,
  "24h": 24,
};

type ChartPoint = {
  time: string;
  ts: number;
  pilotGpuTemp?: number;
  baselineGpuTemp?: number;
  pilotFanRpm?: number;
  baselineFanRpm?: number;
  pilotGpuPowerW?: number;
  baselineGpuPowerW?: number;
};

function downsample(data: ChartPoint[], maxPoints: number): ChartPoint[] {
  if (data.length <= maxPoints) return data;
  const bucketSize = Math.max(1, Math.floor(data.length / maxPoints));
  const result: ChartPoint[] = [];
  for (let i = 0; i < data.length; i += bucketSize) {
    const bucket = data.slice(i, i + bucketSize);
    if (bucket.length === 0) continue;
    result.push(bucket[Math.floor(bucket.length / 2)]);
  }
  return result;
}

export function LivePerformancePanel({ loading }: { loading: boolean }) {
  const [range, setRange] = useState<TimeRange>("2h");
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [chartLoading, setChartLoading] = useState(true);
  const [showPower, setShowPower] = useState(false);
  const [exporting, setExporting] = useState(false);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    try {
      const hours = RANGE_HOURS[range];
      const res = await api.getThermalHistoryRaw(hours);
      if (!mountedRef.current) return;
      if (!res.ok) return;

      const json = await res.json();
      const points = (json.points || []) as {
        ts: number;
        time: string;
        pilot: number;
        baseline: number;
        pilot_fan_rpm?: number | null;
        control_fan_rpm?: number | null;
        pilot_gpu_power_w?: number | null;
        baseline_gpu_power_w?: number | null;
      }[];

      let mapped: ChartPoint[] = points.map((p) => ({
        time: p.time,
        ts: Math.round(p.ts * 1000),
        pilotGpuTemp: p.pilot ?? undefined,
        baselineGpuTemp: p.baseline ?? undefined,
        pilotFanRpm: p.pilot_fan_rpm ?? undefined,
        baselineFanRpm: p.control_fan_rpm ?? undefined,
        pilotGpuPowerW: p.pilot_gpu_power_w ?? undefined,
        baselineGpuPowerW: p.baseline_gpu_power_w ?? undefined,
      }));

      if (range === "24h") {
        mapped = downsample(mapped, 500);
      }

      setChartData(mapped);
    } catch {
      // ignore
    } finally {
      if (mountedRef.current) setChartLoading(false);
    }
  }, [range]);

  useEffect(() => {
    mountedRef.current = true;
    setChartLoading(true);
    fetchData();
    const id = setInterval(fetchData, 5_000);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [fetchData]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const daysMap: Record<TimeRange, number> = {
        "30 min": 1,
        "2h": 1,
        "24h": 1,
      };
      const res = await api.getThermalExport({ days: daysMap[range] });
      if (res.ok) {
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition");
        const match = disposition?.match(/filename="?([^";\n]+)"?/);
        const filename = match?.[1] || "cooledai_thermal_export.csv";
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {
      // ignore
    } finally {
      setExporting(false);
    }
  };

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
        <h2 className="text-xl font-semibold text-white">
          Live System Performance
        </h2>
        <button
          type="button"
          onClick={handleExport}
          disabled={exporting}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-white/70 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50"
        >
          {exporting ? (
            <svg
              className="animate-spin h-3.5 w-3.5"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          ) : (
            <svg
              width={14}
              height={14}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          )}
          Download Report
        </button>
      </div>

      {/* Range selector + power toggle */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="flex gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
          {(["30 min", "2h", "24h"] as TimeRange[]).map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setRange(r)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                range === r
                  ? "bg-[#00FFCC]/20 text-[#00FFCC]"
                  : "text-white/60 hover:text-white/90"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setShowPower((p) => !p)}
          className={`px-3 py-1.5 text-xs font-medium rounded-md border transition-colors ${
            showPower
              ? "bg-purple-500/20 text-purple-400 border-purple-500/30"
              : "bg-white/5 text-white/50 border-white/10 hover:text-white/70"
          }`}
        >
          GPU Power W
        </button>
      </div>

      {/* Chart */}
      {chartLoading || loading ? (
        <div className="h-[350px] flex items-center justify-center">
          <div className="animate-spin h-6 w-6 border-2 border-white/20 border-t-[#00FFCC] rounded-full" />
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData}>
            <CartesianGrid
              stroke="rgba(255,255,255,0.05)"
              strokeDasharray="3 3"
            />
            <XAxis
              dataKey="ts"
              tickFormatter={formatTime}
              tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
            />
            <YAxis
              yAxisId="temp"
              tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
              label={{
                value: "Temperature (\u00B0C)",
                angle: -90,
                position: "insideLeft",
                fill: "rgba(255,255,255,0.3)",
                fontSize: 11,
              }}
            />
            <YAxis
              yAxisId="rpm"
              orientation="right"
              tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
              label={{
                value: "Fan RPM",
                angle: 90,
                position: "insideRight",
                fill: "rgba(255,255,255,0.3)",
                fontSize: 11,
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px",
                color: "white",
                fontSize: 12,
              }}
              labelFormatter={formatTime}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }}
            />

            {/* Temperature lines */}
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="pilotGpuTemp"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="CooledAI GPU Temp"
            />
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="baselineGpuTemp"
              stroke="#6b7280"
              strokeWidth={1.5}
              dot={false}
              name="Control GPU Temp"
            />

            {/* Fan RPM lines */}
            <Line
              yAxisId="rpm"
              type="monotone"
              dataKey="pilotFanRpm"
              stroke="#00FFCC"
              strokeWidth={1.5}
              dot={false}
              name="CooledAI Fan RPM"
            />
            <Line
              yAxisId="rpm"
              type="monotone"
              dataKey="baselineFanRpm"
              stroke="#6b7280"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              name="Control Fan RPM"
            />

            {/* Power lines (optional) */}
            {showPower && (
              <>
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="pilotGpuPowerW"
                  stroke="#a855f7"
                  strokeWidth={1.5}
                  dot={false}
                  name="CooledAI GPU Power W"
                />
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="baselineGpuPowerW"
                  stroke="#a855f7"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  name="Control GPU Power W"
                />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
