"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
import { useAuth } from "@clerk/nextjs";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type LiveStats = {
  efficiency_gain_pct: number;
  last_telemetry_at: number | null;
  power_reclaimed_kwh: number;
  power_reclaimed_watts: number;
  power_comparison_mode?: "raw_fan_wattage" | "fan_rpm_model" | "not_comparable";
  estimated_annual_savings_usd: number;
  has_live_data: boolean;
  uptime_hours: number;
  message?: string;
  pilot_node: {
    node_id: string;
    fan_rpm: number;
    fan_power_watts: number | null;
    temp_c: number | null;
    cpu_temp_c?: number | null;
    gpu_power_w?: number | null;
    peak_gpu_power_w?: number | null;
    last_seen_s_ago: number | null;
  };
  baseline_node: {
    node_id: string;
    fan_rpm: number;
    fan_power_watts: number | null;
    temp_c: number | null;
    cpu_temp_c?: number | null;
    gpu_power_w?: number | null;
    peak_gpu_power_w?: number | null;
    source: string;
    last_seen_s_ago: number | null;
  };
  calibration_profile?: Record<string, unknown>;
  prediction_confidence?: number;
  efficiency_score?: number;
};

type Reasoning = {
  diagnostic_reasoning: string;
  recommendations: string[];
  reasoning_log: string[];
  efficiency_score: number | null;
  recommended_cooling_delta: number | null;
  prediction_confidence?: number;
};

type PulseRange = "1h" | "24h" | "7d";
const RANGE_HOURS: Record<PulseRange, number> = { "1h": 1, "24h": 24, "7d": 168 };

type ChartPoint = {
  time: string;
  ts: number;
  pilot: number;
  baseline: number;
  pilotFanRpm?: number;
  controlFanRpm?: number;
  pilotCpuTemp?: number;
  baselineCpuTemp?: number;
  pilotGpuPowerW?: number;
  baselineGpuPowerW?: number;
  delta?: number;
};

const STATS_POLL_MS = 5_000;
const GRAPH_POLL_MS = 60_000;
const EXPORT_DAY_OPTIONS = [7, 14, 30, 60] as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function downsample(data: ChartPoint[], maxPoints: number): ChartPoint[] {
  if (data.length <= maxPoints) return data;
  const bucketSize = Math.max(1, Math.floor(data.length / maxPoints));
  const result: ChartPoint[] = [];
  for (let i = 0; i < data.length; i += bucketSize) {
    const bucket = data.slice(i, i + bucketSize);
    if (bucket.length === 0) continue;
    const avg = (arr: (number | undefined)[]): number | undefined => {
      const valid = arr.filter((v): v is number => v != null && !isNaN(v));
      return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : undefined;
    };
    result.push({
      time: bucket[Math.floor(bucket.length / 2)].time,
      ts: bucket[Math.floor(bucket.length / 2)].ts,
      pilot: avg(bucket.map((p) => p.pilot)) ?? 0,
      baseline: avg(bucket.map((p) => p.baseline)) ?? 0,
      controlFanRpm: avg(bucket.map((p) => p.controlFanRpm)),
      pilotFanRpm: avg(bucket.map((p) => p.pilotFanRpm)),
      pilotCpuTemp: avg(bucket.map((p) => p.pilotCpuTemp)),
      baselineCpuTemp: avg(bucket.map((p) => p.baselineCpuTemp)),
      pilotGpuPowerW: avg(bucket.map((p) => p.pilotGpuPowerW)),
      baselineGpuPowerW: avg(bucket.map((p) => p.baselineGpuPowerW)),
    });
  }
  return result;
}

// ---------------------------------------------------------------------------
// Export CSV Section
// ---------------------------------------------------------------------------
function ExportCSVSection({ getToken }: { getToken: () => Promise<string | null> }) {
  const [exportMode, setExportMode] = useState<"preset" | "custom">("preset");
  const [presetDays, setPresetDays] = useState<number>(7);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExport = async () => {
    setExportError(null);
    setExporting(true);
    try {
      const token = await getToken();
      if (!token) {
        setExportError("Sign in required to export.");
        return;
      }
      let params: { start_date?: string; end_date?: string; days?: number };
      if (exportMode === "preset") {
        params = { days: presetDays };
      } else {
        if (!startDate || !endDate) {
          setExportError("Select both start and end dates.");
          return;
        }
        params = { start_date: startDate, end_date: endDate };
      }
      const res = await api.getThermalExport(params, token);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setExportError((err as { detail?: string }).detail || `Export failed (${res.status})`);
        return;
      }
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
    } catch (e) {
      setExportError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  const today = new Date().toISOString().slice(0, 10);
  const maxStart = new Date();
  maxStart.setDate(maxStart.getDate() - 60);
  const minStart = maxStart.toISOString().slice(0, 10);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.25 }}
      className="rounded-xl border border-white/10 bg-[#141414] p-6"
    >
      <h2 className="text-base font-semibold text-white mb-1">Export Data</h2>
      <p className="text-sm text-white/50 mb-4">
        Download thermal telemetry as CSV (up to 60 days).
      </p>
      <div className="flex flex-col sm:flex-row flex-wrap gap-4 items-start">
        <div className="flex flex-col gap-2">
          <span className="text-xs text-white/50">Range</span>
          <div className="flex gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
            <button
              type="button"
              onClick={() => setExportMode("preset")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                exportMode === "preset" ? "bg-[#00FFCC]/20 text-[#00FFCC]" : "text-white/60 hover:text-white/90"
              }`}
            >
              Last N days
            </button>
            <button
              type="button"
              onClick={() => setExportMode("custom")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                exportMode === "custom" ? "bg-[#00FFCC]/20 text-[#00FFCC]" : "text-white/60 hover:text-white/90"
              }`}
            >
              Custom dates
            </button>
          </div>
        </div>
        {exportMode === "preset" ? (
          <div className="flex flex-col gap-2">
            <span className="text-xs text-white/50">Days</span>
            <select
              value={presetDays}
              onChange={(e) => setPresetDays(Number(e.target.value))}
              className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-[#00FFCC]"
            >
              {EXPORT_DAY_OPTIONS.map((d) => (
                <option key={d} value={d}>
                  Last {d} days
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="flex flex-wrap gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-xs text-white/50">Start date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                min={minStart}
                max={today}
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-[#00FFCC]"
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-white/50">End date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                min={startDate || minStart}
                max={today}
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-[#00FFCC]"
              />
            </div>
          </div>
        )}
        <div className="flex items-end">
          <button
            type="button"
            onClick={handleExport}
            disabled={exporting}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#00FFCC]/20 text-[#00FFCC] font-medium text-sm hover:bg-[#00FFCC]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {exporting ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Exporting...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download CSV
              </>
            )}
          </button>
        </div>
      </div>
      {exportError && <p className="mt-3 text-sm text-red-400">{exportError}</p>}
      <p className="text-xs text-white/40 mt-3">
        CSV includes: Timestamp, Date, Time, CooledAI &amp; Control GPU temp, Temp delta, Fan RPM, CPU temp, GPU power. Max 60 days per export.
      </p>
    </motion.section>
  );
}

// ---------------------------------------------------------------------------
// Main Advanced Page
// ---------------------------------------------------------------------------
export default function AdvancedPage() {
  const { getToken } = useAuth();

  // Stats polling
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [authError, setAuthError] = useState<string | null>(null);

  // Chart data
  const [pulseRange, setPulseRange] = useState<PulseRange>("1h");
  const [rangeData, setRangeData] = useState<ChartPoint[]>([]);
  const [rangeLoading, setRangeLoading] = useState(false);
  const [chartMetric, setChartMetric] = useState<"gpu" | "cpu" | "fan" | "power">("gpu");

  // Reasoning
  const [reasoning, setReasoning] = useState<Reasoning | null>(null);
  const [showReasoning, setShowReasoning] = useState(false);

  // Raw JSON inspector
  const [showRawJson, setShowRawJson] = useState(false);

  // ---- Fetch Stats ----
  const fetchStats = useCallback(async () => {
    try {
      setAuthError(null);
      const token = await getToken();
      if (!token) {
        setAuthError("Sign in required.");
        setStatsLoading(false);
        return;
      }
      const res = await api.getStats(token);
      if (!res.ok) {
        if (res.status === 401) setAuthError("Sign in required.");
        setStatsLoading(false);
        return;
      }
      const data: LiveStats = await res.json();
      setStats(data);
      setStatsLoading(false);
    } catch {
      setAuthError("Failed to fetch stats.");
      setStatsLoading(false);
    }
  }, [getToken]);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, STATS_POLL_MS);
    return () => clearInterval(id);
  }, [fetchStats]);

  // ---- Fetch Chart Data ----
  const fetchRangeRaw = useCallback(
    async (hours: number) => {
      setRangeLoading(true);
      try {
        const token = await getToken();
        if (!token) {
          setRangeData([]);
          setRangeLoading(false);
          return;
        }
        const res = await api.getThermalHistoryRaw(hours, token);
        if (!res.ok) {
          setRangeData([]);
          setRangeLoading(false);
          return;
        }
        const data = await res.json();
        const points = (data.points || []) as {
          ts: number;
          time: string;
          pilot: number;
          baseline: number;
          pilot_fan_rpm?: number | null;
          control_fan_rpm?: number | null;
          pilot_cpu_temp?: number | null;
          baseline_cpu_temp?: number | null;
          pilot_gpu_power_w?: number | null;
          baseline_gpu_power_w?: number | null;
        }[];
        setRangeData(
          points.map((p) => ({
            time: p.time,
            pilot: p.pilot,
            baseline: p.baseline,
            controlFanRpm: p.control_fan_rpm ?? undefined,
            pilotFanRpm: p.pilot_fan_rpm ?? undefined,
            pilotCpuTemp: p.pilot_cpu_temp ?? undefined,
            baselineCpuTemp: p.baseline_cpu_temp ?? undefined,
            pilotGpuPowerW: p.pilot_gpu_power_w ?? undefined,
            baselineGpuPowerW: p.baseline_gpu_power_w ?? undefined,
            delta: p.baseline - p.pilot,
            ts: Math.round(p.ts * 1000),
          }))
        );
      } catch {
        setRangeData([]);
      } finally {
        setRangeLoading(false);
      }
    },
    [getToken]
  );

  useEffect(() => {
    fetchRangeRaw(RANGE_HOURS[pulseRange]);
  }, [pulseRange, fetchRangeRaw]);

  useEffect(() => {
    const id = setInterval(() => fetchRangeRaw(RANGE_HOURS[pulseRange]), GRAPH_POLL_MS);
    return () => clearInterval(id);
  }, [pulseRange, fetchRangeRaw]);

  // ---- Fetch Reasoning ----
  const fetchReasoning = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;
      const res = await api.getOptimizationReasoning(token);
      if (res.ok) setReasoning(await res.json());
    } catch {
      // ignore
    }
  }, [getToken]);

  useEffect(() => {
    fetchReasoning();
    const id = setInterval(fetchReasoning, 30_000);
    return () => clearInterval(id);
  }, [fetchReasoning]);

  // ---- Derived ----
  const displayedData = useMemo(() => {
    const maxPts = pulseRange === "7d" ? 500 : pulseRange === "24h" ? 360 : rangeData.length;
    return downsample(rangeData, maxPts);
  }, [rangeData, pulseRange]);

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  };

  // Chart lines config
  const metricConfigs: Record<string, { dataKeys: { key: string; name: string; color: string; dashed?: boolean; yAxis?: string }[] }> = {
    gpu: {
      dataKeys: [
        { key: "pilot", name: "CooledAI GPU Temp", color: "#3b82f6" },
        { key: "baseline", name: "Control GPU Temp", color: "#6b7280", dashed: true },
      ],
    },
    cpu: {
      dataKeys: [
        { key: "pilotCpuTemp", name: "CooledAI CPU Temp", color: "#f97316" },
        { key: "baselineCpuTemp", name: "Control CPU Temp", color: "#6b7280", dashed: true },
      ],
    },
    fan: {
      dataKeys: [
        { key: "pilotFanRpm", name: "CooledAI Fan RPM", color: "#00FFCC", yAxis: "right" },
        { key: "controlFanRpm", name: "Control Fan RPM", color: "#6b7280", dashed: true, yAxis: "right" },
      ],
    },
    power: {
      dataKeys: [
        { key: "pilotGpuPowerW", name: "CooledAI GPU Power", color: "#a855f7" },
        { key: "baselineGpuPowerW", name: "Control GPU Power", color: "#6b7280", dashed: true },
      ],
    },
  };

  const pilotNode = stats?.pilot_node;
  const baselineNode = stats?.baseline_node;

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="mb-8"
      >
        <h1 className="text-2xl font-semibold tracking-tight text-white">
          Advanced Diagnostics
        </h1>
        <p className="text-sm text-white/50 mt-0.5">
          Raw telemetry, AI reasoning, and engineering data
        </p>
      </motion.div>

      {authError && (
        <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-sm text-red-400">{authError}</p>
        </div>
      )}

      {/* ========== Raw Telemetry Grid ========== */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
        className="rounded-xl border border-white/10 bg-[#141414] p-6 mb-6"
      >
        <h2 className="text-base font-semibold text-white mb-4">
          Raw Telemetry
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Pilot Node */}
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-[#3b82f6] mb-2">
              CooledAI Node {pilotNode?.node_id ? `(${pilotNode.node_id})` : ""}
            </p>
            {statsLoading ? (
              <div className="space-y-2 animate-pulse">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-5 w-48 rounded bg-white/10" />
                ))}
              </div>
            ) : pilotNode ? (
              <div className="space-y-1.5 text-sm">
                <Row label="GPU Temp" value={pilotNode.temp_c != null ? `${pilotNode.temp_c.toFixed(1)}\u00B0C` : "--"} />
                <Row label="CPU Temp" value={pilotNode.cpu_temp_c != null ? `${pilotNode.cpu_temp_c.toFixed(1)}\u00B0C` : "--"} />
                <Row label="Fan RPM" value={pilotNode.fan_rpm != null ? `${pilotNode.fan_rpm.toFixed(0)}` : "--"} />
                <Row label="Fan Power" value={pilotNode.fan_power_watts != null ? `${pilotNode.fan_power_watts.toFixed(1)}W` : "--"} />
                <Row label="GPU Power" value={pilotNode.gpu_power_w != null ? `${pilotNode.gpu_power_w.toFixed(0)}W` : "--"} />
                <Row label="Peak GPU Power" value={(pilotNode as Record<string, unknown>).peak_gpu_power_w != null ? `${Number((pilotNode as Record<string, unknown>).peak_gpu_power_w).toFixed(0)}W` : "--"} />
                <Row label="Last Seen" value={pilotNode.last_seen_s_ago != null ? `${pilotNode.last_seen_s_ago.toFixed(0)}s ago` : "--"} />
              </div>
            ) : (
              <p className="text-sm text-white/40">No data</p>
            )}
          </div>

          {/* Baseline Node */}
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">
              Control Node {baselineNode?.node_id ? `(${baselineNode.node_id})` : ""}
            </p>
            {statsLoading ? (
              <div className="space-y-2 animate-pulse">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-5 w-48 rounded bg-white/10" />
                ))}
              </div>
            ) : baselineNode ? (
              <div className="space-y-1.5 text-sm">
                <Row label="GPU Temp" value={baselineNode.temp_c != null ? `${baselineNode.temp_c.toFixed(1)}\u00B0C` : "--"} />
                <Row label="CPU Temp" value={baselineNode.cpu_temp_c != null ? `${baselineNode.cpu_temp_c.toFixed(1)}\u00B0C` : "--"} />
                <Row label="Fan RPM" value={baselineNode.fan_rpm != null ? `${baselineNode.fan_rpm.toFixed(0)}` : "--"} />
                <Row label="Fan Power" value={baselineNode.fan_power_watts != null ? `${baselineNode.fan_power_watts.toFixed(1)}W` : "--"} />
                <Row label="GPU Power" value={baselineNode.gpu_power_w != null ? `${baselineNode.gpu_power_w.toFixed(0)}W` : "--"} />
                <Row label="Peak GPU Power" value={(baselineNode as Record<string, unknown>).peak_gpu_power_w != null ? `${Number((baselineNode as Record<string, unknown>).peak_gpu_power_w).toFixed(0)}W` : "--"} />
                <Row label="Last Seen" value={baselineNode.last_seen_s_ago != null ? `${baselineNode.last_seen_s_ago.toFixed(0)}s ago` : "--"} />
                <Row label="Source" value={baselineNode.source || "--"} />
              </div>
            ) : (
              <p className="text-sm text-white/40">No data</p>
            )}
          </div>
        </div>

        {/* Summary row */}
        {stats && (
          <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MiniStat label="Efficiency Gain" value={`${stats.efficiency_gain_pct.toFixed(1)}%`} color="#22c55e" />
            <MiniStat
              label="Power Reclaimed"
              value={
                stats.power_reclaimed_kwh < 1
                  ? `${(stats.power_reclaimed_kwh * 1000).toFixed(0)} Wh`
                  : `${stats.power_reclaimed_kwh.toFixed(1)} kWh`
              }
              color="#22c55e"
            />
            <MiniStat label="Est. Annual Savings" value={`$${stats.estimated_annual_savings_usd.toFixed(0)}`} color="#00FFCC" />
            <MiniStat label="Uptime" value={`${stats.uptime_hours.toFixed(1)}h`} color="white" />
          </div>
        )}
      </motion.section>

      {/* ========== Full Multi-Metric Chart ========== */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className="rounded-xl border border-white/10 bg-[#141414] p-6 mb-6"
      >
        <h2 className="text-base font-semibold text-white mb-4">
          Multi-Metric Chart
        </h2>

        {/* Range + metric selectors */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <div className="flex gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
            {(["1h", "24h", "7d"] as PulseRange[]).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setPulseRange(r)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                  pulseRange === r
                    ? "bg-[#00FFCC]/20 text-[#00FFCC]"
                    : "text-white/60 hover:text-white/90"
                }`}
              >
                {r}
              </button>
            ))}
          </div>
          <div className="flex gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
            {(["gpu", "cpu", "fan", "power"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setChartMetric(m)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize ${
                  chartMetric === m
                    ? "bg-[#3b82f6]/20 text-[#3b82f6]"
                    : "text-white/60 hover:text-white/90"
                }`}
              >
                {m === "gpu" ? "GPU Temp" : m === "cpu" ? "CPU Temp" : m === "fan" ? "Fan RPM" : "GPU Power"}
              </button>
            ))}
          </div>
        </div>

        {rangeLoading ? (
          <div className="h-[350px] flex items-center justify-center">
            <div className="animate-spin h-6 w-6 border-2 border-white/20 border-t-[#00FFCC] rounded-full" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={displayedData}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
              <XAxis
                dataKey="ts"
                tickFormatter={formatTime}
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
              />
              <YAxis
                yAxisId="left"
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
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
              <Legend wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }} />
              {metricConfigs[chartMetric].dataKeys.map((dk) => (
                <Line
                  key={dk.key}
                  yAxisId={dk.yAxis === "right" ? "right" : "left"}
                  type="monotone"
                  dataKey={dk.key}
                  stroke={dk.color}
                  strokeWidth={dk.dashed ? 1.5 : 2}
                  strokeDasharray={dk.dashed ? "4 4" : undefined}
                  dot={false}
                  name={dk.name}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </motion.section>

      {/* ========== CSV Export ========== */}
      <div className="mb-6">
        <ExportCSVSection getToken={getToken} />
      </div>

      {/* ========== AI Reasoning Panel ========== */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.15 }}
        className="rounded-xl border border-white/10 bg-[#141414] p-6 mb-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-white">
            AI Reasoning &amp; Diagnostics
          </h2>
          <button
            type="button"
            onClick={() => setShowReasoning((v) => !v)}
            className="text-xs text-white/50 hover:text-white/80 transition-colors"
          >
            {showReasoning ? "Collapse" : "Expand"}
          </button>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <MiniStat
            label="Efficiency Score"
            value={reasoning?.efficiency_score != null ? `${reasoning.efficiency_score.toFixed(0)}` : stats?.efficiency_score != null ? `${stats.efficiency_score}` : "--"}
            color="#22c55e"
          />
          <MiniStat
            label="Prediction Confidence"
            value={reasoning?.prediction_confidence != null ? `${(reasoning.prediction_confidence * 100).toFixed(0)}%` : stats?.prediction_confidence != null ? `${(stats.prediction_confidence * 100).toFixed(0)}%` : "--"}
            color="#3b82f6"
          />
          <MiniStat
            label="Cooling Delta"
            value={reasoning?.recommended_cooling_delta != null ? `${reasoning.recommended_cooling_delta.toFixed(1)}\u00B0C` : "--"}
            color="#00FFCC"
          />
          <MiniStat
            label="Recommendations"
            value={reasoning?.recommendations ? `${reasoning.recommendations.length}` : "0"}
            color="white"
          />
        </div>

        <AnimatePresence>
          {showReasoning && reasoning && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              {/* Recommendations */}
              {reasoning.recommendations.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">
                    Recommendations
                  </p>
                  <ul className="space-y-1">
                    {reasoning.recommendations.map((r, i) => (
                      <li key={i} className="text-sm text-white/70 flex items-start gap-2">
                        <span className="text-[#22c55e] mt-0.5">&#8226;</span>
                        {r}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Diagnostic Reasoning */}
              <div className="mb-4">
                <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">
                  Diagnostic Reasoning
                </p>
                <p className="text-sm text-white/60 whitespace-pre-wrap font-mono bg-[#0a0a0a] rounded-lg p-3 border border-white/5">
                  {reasoning.diagnostic_reasoning || "No diagnostic data."}
                </p>
              </div>

              {/* Reasoning Log */}
              {reasoning.reasoning_log.length > 0 && (
                <div>
                  <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">
                    Reasoning Log
                  </p>
                  <div className="text-xs text-white/40 whitespace-pre-wrap font-mono bg-[#0a0a0a] rounded-lg p-3 border border-white/5 max-h-60 overflow-auto">
                    {reasoning.reasoning_log.join("\n")}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>

      {/* ========== Calibration Profile ========== */}
      {stats?.calibration_profile && (
        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6 mb-6"
        >
          <h2 className="text-base font-semibold text-white mb-4">
            Calibration Profile
          </h2>
          <div className="text-xs text-white/50 font-mono bg-[#0a0a0a] rounded-lg p-3 border border-white/5 max-h-60 overflow-auto whitespace-pre-wrap">
            {JSON.stringify(stats.calibration_profile, null, 2)}
          </div>
        </motion.section>
      )}

      {/* ========== Raw JSON Inspector ========== */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.25 }}
        className="rounded-xl border border-white/10 bg-[#141414] mb-6"
      >
        <button
          type="button"
          onClick={() => setShowRawJson((v) => !v)}
          className="w-full flex items-center justify-between p-6 text-left"
        >
          <h2 className="text-base font-semibold text-white">
            Raw JSON Inspector
          </h2>
          <svg
            width={20}
            height={20}
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth={2}
            className={`transition-transform duration-200 ${showRawJson ? "rotate-180" : ""}`}
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
        <AnimatePresence initial={false}>
          {showRawJson && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-6 pb-6">
                <div className="text-xs text-white/40 font-mono bg-[#0a0a0a] rounded-lg p-3 border border-white/5 max-h-96 overflow-auto whitespace-pre-wrap">
                  {stats ? JSON.stringify(stats, null, 2) : "Loading..."}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small reusable components
// ---------------------------------------------------------------------------
function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-white/50">{label}</span>
      <span className="text-white font-mono tabular-nums">{value}</span>
    </div>
  );
}

function MiniStat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div>
      <p className="text-xs text-white/40 uppercase tracking-wider">{label}</p>
      <p className="text-lg font-bold tabular-nums" style={{ color }}>
        {value}
      </p>
    </div>
  );
}
