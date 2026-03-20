"use client";

import { Suspense, useState, useEffect, useCallback, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import { api } from "@/lib/api";
import { useAuth } from "@clerk/nextjs";

const STATS_POLL_MS = 5_000;  // Real-time stats below overview
const GRAPH_POLL_MS = 60_000;  // Graph refreshes every 1 minute
const PULSE_STORAGE_KEY = "COOLEDAI_PULSE_DATA";
const PULSE_STORAGE_VERSION = 4; // bump to invalidate stale/non-comparable cached data
const MAX_STORED_POINTS = 60_480; // 7 days at 10s interval
const EFFICIENCY_DELTA_WINDOW_MS = 30_000; // Instantaneous: last 30 seconds

type PulseRange = "1h" | "24h" | "7d";

const RANGE_MS: Record<PulseRange, number> = {
  "1h": 60 * 60 * 1000,
  "24h": 24 * 60 * 60 * 1000,
  "7d": 7 * 24 * 60 * 60 * 1000,
};

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
  pilot_node: { node_id: string; fan_rpm: number; fan_power_watts: number | null; temp_c: number | null; cpu_temp_c?: number | null; gpu_power_w?: number | null; last_seen_s_ago: number | null };
  baseline_node: { node_id: string; fan_rpm: number; fan_power_watts: number | null; temp_c: number | null; cpu_temp_c?: number | null; gpu_power_w?: number | null; source: string; last_seen_s_ago: number | null };
};

type PulsePoint = {
  time: string;
  pilot: number;
  baseline: number;
  controlFanRpm?: number;
  pilotFanRpm?: number;
  pilotCpuTemp?: number;
  baselineCpuTemp?: number;
  pilotGpuPowerW?: number;
  baselineGpuPowerW?: number;
  delta?: number;
  ts: number;
};

const EXPORT_DAY_OPTIONS = [7, 14, 30, 60] as const;

function ExportCSVSection({
  getToken,
  authError,
}: {
  getToken: () => Promise<string | null>;
  authError: boolean;
}) {
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
        setExportError(err.detail || `Export failed (${res.status})`);
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
        Download thermal telemetry as CSV (up to 60 days) to evaluate CooledAI efficiency in spreadsheets.
      </p>
      <div className="flex flex-col sm:flex-row flex-wrap gap-4 items-start">
        <div className="flex flex-col gap-2">
          <span className="text-xs text-white/50">Range</span>
          <div className="flex gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
            <button
              type="button"
              onClick={() => setExportMode("preset")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                exportMode === "preset" ? "bg-accent-cyan/20 text-accent-cyan" : "text-white/60 hover:text-white/90"
              }`}
            >
              Last N days
            </button>
            <button
              type="button"
              onClick={() => setExportMode("custom")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                exportMode === "custom" ? "bg-accent-cyan/20 text-accent-cyan" : "text-white/60 hover:text-white/90"
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
              className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
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
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
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
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
              />
            </div>
          </div>
        )}
        <div className="flex items-end">
          <button
            type="button"
            onClick={handleExport}
            disabled={authError || exporting}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan font-medium text-sm hover:bg-accent-cyan/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {exporting ? (
              <>
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Exporting…
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
      {exportError && (
        <p className="mt-3 text-sm text-red-400">{exportError}</p>
      )}
      <p className="text-xs text-white/40 mt-3">
        CSV includes: Timestamp, Date, Time, CooledAI & Control GPU temp, Temp delta, Fan RPM, CPU temp, GPU power. Max 60 days per export.
      </p>
    </motion.section>
  );
}

function PortalOverviewContent() {
  const searchParams = useSearchParams();
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);

  const { getToken } = useAuth();
  const [authError, setAuthError] = useState<string | null>(null);

  const [efficiencyGain, setEfficiencyGain] = useState<number | null>(null);
  const [efficiencyLoading, setEfficiencyLoading] = useState(true);
  const [powerReclaimed, setPowerReclaimed] = useState(0);
  const [annualSavings, setAnnualSavings] = useState(0);
  const [hasLiveData, setHasLiveData] = useState(false);
  const [pilotWatts, setPilotWatts] = useState<number | null>(null);
  const [baselineWatts, setBaselineWatts] = useState<number | null>(null);
  const [powerMode, setPowerMode] = useState<LiveStats["power_comparison_mode"]>("not_comparable");
  const [pilotNodeId, setPilotNodeId] = useState("");
  const [baselineNodeId, setBaselineNodeId] = useState("");
  const [pilotNode, setPilotNode] = useState<LiveStats["pilot_node"] | null>(null);
  const [baselineNode, setBaselineNode] = useState<LiveStats["baseline_node"] | null>(null);
  const [pulseRange, setPulseRange] = useState<PulseRange>("1h");
  const [rangeData, setRangeData] = useState<PulsePoint[]>([]);
  const [rangeLoading, setRangeLoading] = useState(false);
  const [chartMetric, setChartMetric] = useState<"gpu" | "cpu" | "fan" | "power">("gpu");
  const [pulseData, setPulseData] = useState<PulsePoint[]>([]);
  const [serverMessage, setServerMessage] = useState<string | null>(null);
  const [showReasoning, setShowReasoning] = useState(false);
  const [reasoning, setReasoning] = useState<{
    diagnostic_reasoning: string;
    recommendations: string[];
    reasoning_log: string[];
    efficiency_score: number | null;
    recommended_cooling_delta: number | null;
  } | null>(null);

  // Force fresh data on mount: clear stale localStorage (82.1%, 0, old versions)
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = localStorage.getItem(PULSE_STORAGE_KEY);
      const parsed = raw ? (JSON.parse(raw) as { version?: number; points?: unknown[] }) : null;
      const versionOk = parsed?.version === PULSE_STORAGE_VERSION;
      const hasStalePoints = Array.isArray(parsed?.points) && parsed.points.some(
        (p: unknown) => {
          const arr = Array.isArray(p) ? p : [];
          const pilot = arr[1] as number;
          const baseline = arr[2] as number;
          return pilot === 0 || baseline === 0 || (pilot > 81 && pilot < 83);
        }
      );
      if (!versionOk || hasStalePoints) {
        localStorage.removeItem(PULSE_STORAGE_KEY);
        setPulseData([]);
      }
    } catch {
      localStorage.removeItem(PULSE_STORAGE_KEY);
      setPulseData([]);
    }
  }, [getToken]);

  const fetchStats = useCallback(async () => {
    try {
      setAuthError(null);
      setServerMessage(null);
      const token = await getToken();
      if (!token) {
        setAuthError("Sign in required.");
        setEfficiencyLoading(false);
        setEfficiencyGain(null);
        setHasLiveData(false);
        return;
      }

      const res = await api.getStats(token);
      if (!res.ok) {
        if (res.status === 401) {
          let detail: string | null = null;
          try {
            const body = await res.json();
            detail = typeof body?.detail === "string" ? body.detail : null;
          } catch {
            // ignore
          }
          setAuthError(detail ? `Sign in required. ${detail}` : "Sign in required.");
          setEfficiencyLoading(false);
          setEfficiencyGain(null);
          setHasLiveData(false);
        }
        console.warn("[CooledAI] /api/v1/stats returned", res.status);
        return;
      }
      const data: LiveStats = await res.json();
      console.log("[CooledAI] stats response:", JSON.stringify(data, null, 2));

      if (data.message === "No recent data") {
        setServerMessage("No recent data");
        setEfficiencyLoading(false);
        setEfficiencyGain(null);
        setHasLiveData(false);
        setPowerReclaimed(0);
        setAnnualSavings(0);
        setPilotWatts(null);
        setBaselineWatts(null);
        setPowerMode("not_comparable");
        setPilotNodeId("");
        setBaselineNodeId("");
        setPilotNode(null);
        setBaselineNode(null);
        setPulseData([]);
        return;
      }

      // Efficiency: (Baseline_Temp - Pilot_Temp) / Baseline_Temp * 100 from live ST550 data
      const pilotTemp = data.pilot_node.temp_c;
      const baselineTemp = data.baseline_node.temp_c;
      const hasBothTemps = pilotTemp != null && baselineTemp != null && baselineTemp > 0;
      setEfficiencyLoading(!hasBothTemps);
      setEfficiencyGain(data.efficiency_gain_pct);
      setPowerReclaimed(data.power_reclaimed_kwh);
      setAnnualSavings(data.estimated_annual_savings_usd);
      setHasLiveData(data.has_live_data);
      setPilotWatts(data.pilot_node.fan_power_watts);
      setBaselineWatts(data.baseline_node.fan_power_watts);
      setPowerMode(data.power_comparison_mode ?? "not_comparable");
      setPilotNodeId(data.pilot_node.node_id || "");
      setBaselineNodeId(data.baseline_node.node_id || "");
      setPilotNode(data.pilot_node);
      setBaselineNode(data.baseline_node);

      // Only add point when BOTH temps are present — avoid defaulting to 0 (which produces bogus deltas)
      if (pilotTemp != null && baselineTemp != null) {
        const now = Date.now();
        const point: PulsePoint = {
          time: new Date(now).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          pilot: pilotTemp,
          baseline: baselineTemp,
          controlFanRpm: data.baseline_node.fan_rpm ?? undefined,
          delta: baselineTemp - pilotTemp,
          ts: now,
        };
        setPulseData((prev) => {
          const next = [...prev, point].slice(-MAX_STORED_POINTS);
          try {
            const compact = next.map((p) => [p.ts, p.pilot, p.baseline] as [number, number, number]);
            localStorage.setItem(PULSE_STORAGE_KEY, JSON.stringify({ version: PULSE_STORAGE_VERSION, points: compact }));
          } catch {
            // localStorage full or unavailable
          }
          return next;
        });
      }
    } catch (err) {
      console.error("[CooledAI] stats fetch failed:", err);
      setAuthError("Sign in required.");
      setEfficiencyLoading(false);
      setEfficiencyGain(null);
      setHasLiveData(false);
    }
  }, [getToken]);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, STATS_POLL_MS);
    return () => clearInterval(id);
  }, [fetchStats]);

  // Fetch RAW thermal history for selected range (no averaging).
  const fetchRangeRaw = useCallback(async (hours: 1 | 24 | 168) => {
    setRangeLoading(true);
    try {
      const token = await getToken();
      if (!token) {
        setAuthError("Sign in required.");
        setRangeData([]);
        setRangeLoading(false);
        return;
      }

      const res = await api.getThermalHistoryRaw(hours, token);
      if (!res.ok) {
        if (res.status === 401) {
          let detail: string | null = null;
          try {
            const body = await res.json();
            detail = typeof body?.detail === "string" ? body.detail : null;
          } catch {
            // ignore
          }
          setAuthError(detail ? `Sign in required. ${detail}` : "Sign in required.");
        }
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
      setAuthError("Sign in required.");
      setRangeData([]);
    } finally {
      setRangeLoading(false);
    }
  }, [getToken]);

  useEffect(() => {
    if (pulseRange === "1h") fetchRangeRaw(1);
    else if (pulseRange === "24h") fetchRangeRaw(24);
    else if (pulseRange === "7d") fetchRangeRaw(168);
  }, [pulseRange, fetchRangeRaw]);

  useEffect(() => {
    const id = setInterval(() => {
      if (pulseRange === "1h") fetchRangeRaw(1);
      else if (pulseRange === "24h") fetchRangeRaw(24);
      else fetchRangeRaw(168);
    }, GRAPH_POLL_MS);
    return () => clearInterval(id);
  }, [pulseRange, fetchRangeRaw]);

  const fetchReasoning = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;
      const res = await api.getOptimizationReasoning(token);
      if (res.ok) {
        const data = await res.json();
        setReasoning(data);
      }
    } catch {
      // ignore
    }
  }, [getToken]);

  useEffect(() => {
    fetchReasoning();
    const id = setInterval(fetchReasoning, 30_000);
    return () => clearInterval(id);
  }, [fetchReasoning]);

  useEffect(() => {
    if (searchParams.get("success") === "true") {
      setShowSuccessBanner(true);
      const t = setTimeout(() => setShowSuccessBanner(false), 8000);
      return () => clearTimeout(t);
    }
  }, [searchParams]);

  const now = Date.now();
  const rangeMs = RANGE_MS[pulseRange];
  const displayedPulseData = rangeData.filter((p) => now - p.ts <= rangeMs);

  // Efficiency Delta: same pulseData as Chart, last 30s window, most recent point
  const efficiencyDeltaC = useMemo(() => {
    const cutoff = Date.now() - EFFICIENCY_DELTA_WINDOW_MS;
    const recent = pulseData.filter((p) => p.ts >= cutoff && p.pilot != null && p.baseline != null);
    if (recent.length === 0) return null;
    const last = recent[recent.length - 1];
    return last.baseline - last.pilot;
  }, [pulseData]);

  const efficiencyDeltaDisplay = efficiencyDeltaC;

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <AnimatePresence>
        {showSuccessBanner && (
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="mb-6 rounded-xl border border-[#22c55e]/30 bg-[#22c55e]/10 px-4 py-3 flex items-center justify-between gap-4"
          >
            <p className="text-sm font-medium text-[#22c55e]">Checkout successful. Fleet Optimization is now active.</p>
            <button
              type="button"
              onClick={() => setShowSuccessBanner(false)}
              className="shrink-0 p-1 text-[#22c55e]/80 hover:text-[#22c55e] rounded"
              aria-label="Dismiss"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="mb-8"
      >
        <h1 className="text-2xl font-semibold tracking-tight text-white">Overview</h1>
        <p className="text-sm text-white/50 mt-0.5">Real-time efficiency and savings at a glance</p>
      </motion.div>

      {/* Stat cards — live from /api/v1/stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.05 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-1">
            Live Efficiency Gain
          </p>
          <p className="text-3xl font-bold text-[#22c55e] tabular-nums">
            {authError ? (
              authError
            ) : serverMessage ? (
              serverMessage
            ) : efficiencyLoading ? (
              <span className="inline-flex items-center gap-2 text-white/70">
                <svg className="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Calculating…
              </span>
            ) : (
              `${(efficiencyGain ?? 0).toFixed(1)}%`
            )}
          </p>
          <p className="text-xs text-white/40 mt-1">
            {serverMessage
              ? serverMessage
              : hasLiveData
                ? powerMode === "not_comparable" || pilotWatts == null || baselineWatts == null
                  ? "Power comparison requires fan telemetry on both nodes"
                  : `${pilotWatts.toFixed(0)}W CooledAI vs ${baselineWatts.toFixed(0)}W Control`
                : "awaiting telemetry…"}
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-1">
            Total Power Reclaimed
          </p>
          <p className="text-3xl font-bold text-[#22c55e] tabular-nums">
            {powerReclaimed < 1
              ? `${(powerReclaimed * 1000).toFixed(0)} Wh`
              : `${powerReclaimed.toLocaleString(undefined, { maximumFractionDigits: 1 })} kWh`}
          </p>
          <p className="text-xs text-white/40 mt-1">
            {hasLiveData ? "cumulative" : "awaiting telemetry…"}
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.15 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-1">
            Estimated Annual Savings
          </p>
          <p className="text-3xl font-bold text-[#22c55e] tabular-nums">
            ${annualSavings.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </p>
          <p className="text-xs text-white/40 mt-1">
            {hasLiveData ? "projected at current rate" : "awaiting telemetry…"}
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-1">
            Efficiency Delta
          </p>
          <p className="text-3xl font-bold tabular-nums text-accent-cyan">
            {efficiencyDeltaDisplay != null
              ? `${efficiencyDeltaDisplay >= 0 ? "+" : ""}${efficiencyDeltaDisplay.toFixed(1)}°C`
              : "—"}
          </p>
          <p className="text-xs text-white/40 mt-1">
            {hasLiveData
              ? "Control − Pilot (positive = Pilot cooler)"
              : "awaiting telemetry…"}
          </p>
        </motion.div>
      </div>

      {/* Real-time data */}
      {hasLiveData && (pilotNode || baselineNode) && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="mb-8 rounded-xl border border-white/10 bg-[#141414] p-4"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-3">Live telemetry</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <p className="text-sm font-medium text-[#22c55e]">{pilotNodeId || "CooledAI"}</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-white/80">
                <span className="text-white/50">GPU temp</span>
                <span>{pilotNode?.temp_c != null ? `${pilotNode.temp_c.toFixed(1)}°C` : "—"}</span>
                <span className="text-white/50">CPU temp</span>
                <span>{pilotNode?.cpu_temp_c != null ? `${pilotNode.cpu_temp_c.toFixed(1)}°C` : "—"}</span>
                <span className="text-white/50">Fan RPM</span>
                <span>{pilotNode?.fan_rpm != null ? pilotNode.fan_rpm : "—"}</span>
                <span className="text-white/50">GPU power</span>
                <span>{pilotNode?.gpu_power_w != null ? `${pilotNode.gpu_power_w} W` : "—"}</span>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium text-[#ef4444]">{baselineNodeId || "Control"}</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-white/80">
                <span className="text-white/50">GPU temp</span>
                <span>{baselineNode?.temp_c != null ? `${baselineNode.temp_c.toFixed(1)}°C` : "—"}</span>
                <span className="text-white/50">CPU temp</span>
                <span>{baselineNode?.cpu_temp_c != null ? `${baselineNode.cpu_temp_c.toFixed(1)}°C` : "—"}</span>
                <span className="text-white/50">Fan RPM</span>
                <span>{baselineNode?.fan_rpm != null ? baselineNode.fan_rpm : "—"}</span>
                <span className="text-white/50">GPU power</span>
                <span>{baselineNode?.gpu_power_w != null ? `${baselineNode.gpu_power_w} W` : "—"}</span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Thermal Chart */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="rounded-xl border border-white/10 bg-[#141414] p-6"
      >
        <div className="flex flex-col gap-4 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2 className="text-base font-semibold text-white">Live Comparison</h2>
              <p className="text-sm text-white/50 mt-0.5">
                {pilotNodeId || "CooledAI"} vs {baselineNodeId || "Control"} · Raw telemetry
              </p>
            </div>
            <div className="flex items-center gap-2">
              {hasLiveData && (
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-[#22c55e]/15 text-[#22c55e] text-xs font-medium">
                  <span className="relative flex h-1.5 w-1.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22c55e] opacity-75" />
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#22c55e]" />
                  </span>
                  Live
                </span>
              )}
              <span className="text-xs text-white/40">Green = CooledAI · Red = Control</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs text-white/50">Time range</span>
              <div className="flex gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
                {(["1h", "24h", "7d"] as const).map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setPulseRange(r)}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                      pulseRange === r ? "bg-accent-cyan/20 text-accent-cyan" : "text-white/60 hover:text-white/90"
                    }`}
                  >
                    {r === "1h" ? "1H" : r === "24h" ? "24H" : "7D"}
                  </button>
                ))}
              </div>
            </div>
            <div className="h-px w-px bg-white/20 sm:hidden" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-white/50">Metric</span>
              <div className="flex flex-wrap gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5">
                {[
                  { id: "gpu" as const, label: "GPU Temp" },
                  { id: "cpu" as const, label: "CPU Temp" },
                  { id: "fan" as const, label: "Fan Speed" },
                  { id: "power" as const, label: "GPU Power" },
                ].map(({ id, label }) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setChartMetric(id)}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                      chartMetric === id ? "bg-accent-cyan/20 text-accent-cyan" : "text-white/60 hover:text-white/90"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
        {rangeLoading ? (
          <div className="h-[320px] flex items-center justify-center text-white/40 text-sm">
            <span className="inline-flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Loading raw data…
            </span>
          </div>
        ) : displayedPulseData.length === 0 ? (
          <div className="h-[320px] flex items-center justify-center text-white/40 text-sm">
            {pulseData.length === 0 ? "Waiting for telemetry…" : "No raw data in selected range yet."}
          </div>
        ) : (
        <div className="h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={displayedPulseData}
              margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="time"
                tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
                domain={chartMetric === "power" ? [0, "auto"] : [0, "dataMax + 10"]}
                tickFormatter={(v) =>
                  chartMetric === "fan" ? `${v}` : chartMetric === "power" ? `${v} W` : `${v}°C`
                }
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1a1a",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: "8px",
                  padding: "12px 16px",
                }}
                labelStyle={{ color: "rgba(255,255,255,0.9)", marginBottom: 8 }}
                formatter={(value, name) => {
                  const v = value != null ? value : 0;
                  const unit =
                    chartMetric === "fan" ? " RPM" : chartMetric === "power" ? " W" : "°C";
                  const label =
                    name === "pilot" ||
                    name === "pilotCpuTemp" ||
                    name === "pilotFanRpm" ||
                    name === "pilotGpuPowerW"
                      ? "CooledAI"
                      : "Control";
                  return [`${v}${unit}`, label];
                }}
                labelFormatter={(label) => label}
              />
              {chartMetric === "gpu" && (
                <>
                  <ReferenceLine y={65} stroke="rgba(234,179,8,0.5)" strokeDasharray="4 4" />
                  <ReferenceLine y={85} stroke="rgba(239,68,68,0.5)" strokeDasharray="4 4" />
                </>
              )}
              {chartMetric === "gpu" && (
                <>
                  <Line type="linear" dataKey="pilot" name="pilot" stroke="#22c55e" strokeWidth={2} dot={false} />
                  <Line type="linear" dataKey="baseline" name="baseline" stroke="#ef4444" strokeWidth={2} dot={false} />
                </>
              )}
              {chartMetric === "cpu" && (
                <>
                  <Line type="linear" dataKey="pilotCpuTemp" name="pilotCpuTemp" stroke="#22c55e" strokeWidth={2} dot={false} connectNulls={false} />
                  <Line type="linear" dataKey="baselineCpuTemp" name="baselineCpuTemp" stroke="#ef4444" strokeWidth={2} dot={false} connectNulls={false} />
                </>
              )}
              {chartMetric === "fan" && (
                <>
                  <Line type="linear" dataKey="pilotFanRpm" name="pilotFanRpm" stroke="#22c55e" strokeWidth={2} dot={false} connectNulls={false} />
                  <Line type="linear" dataKey="controlFanRpm" name="controlFanRpm" stroke="#ef4444" strokeWidth={2} dot={false} connectNulls={false} />
                </>
              )}
              {chartMetric === "power" && (
                <>
                  <Line type="linear" dataKey="pilotGpuPowerW" name="pilotGpuPowerW" stroke="#22c55e" strokeWidth={2} dot={false} connectNulls />
                  <Line type="linear" dataKey="baselineGpuPowerW" name="baselineGpuPowerW" stroke="#ef4444" strokeWidth={2} dot={false} connectNulls />
                </>
              )}
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                iconType="line"
                iconSize={10}
                formatter={(value) =>
                  value === "pilot" ||
                  value === "pilotCpuTemp" ||
                  value === "pilotFanRpm" ||
                  value === "pilotGpuPowerW"
                    ? "CooledAI"
                    : "Control"
                }
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        )}
        <p className="text-xs text-white/40 mt-4">
          {chartMetric === "gpu" && "GPU temp: avg of all GPUs per node. Yellow: 65°C · Red: 85°C"}
          {chartMetric === "cpu" && "CPU temp from cooledai_agent"}
          {chartMetric === "fan" && "Fan RPM: chassis tach (IPMI) when available; otherwise GPU fan % scaled to RPM (nvidia-smi / NVML)."}
          {chartMetric === "power" && "GPU power (W) from nvidia-smi. Both nodes report when telemetry includes power_draw_w."}
        </p>

        {/* AI Reasoning — expandable like Grok/Gemini/Claude */}
        <div className="mt-6 rounded-xl border border-white/10 bg-[#141414] overflow-hidden">
          <button
            type="button"
            onClick={() => setShowReasoning(!showReasoning)}
            className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-white/5 transition-colors"
          >
            <span className="text-sm font-medium text-white/90 flex items-center gap-2">
              <span className="text-accent-cyan">🧠</span>
              How the predictive model thinks
            </span>
            <svg className={`w-4 h-4 text-white/50 transition-transform ${showReasoning ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {showReasoning && (
            <div className="border-t border-white/10 px-4 py-4 bg-black/20 space-y-4 text-sm">
              {reasoning?.diagnostic_reasoning ? (
                <>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">Reasoning</p>
                    <p className="text-white/90 leading-relaxed">{reasoning.diagnostic_reasoning}</p>
                  </div>
                  {reasoning.recommendations?.length > 0 && (
                    <div>
                      <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">Recommendations</p>
                      <ul className="list-disc list-inside space-y-1 text-white/80">
                        {reasoning.recommendations.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {reasoning.reasoning_log?.length > 0 && (
                    <div>
                      <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-2">Step-by-step</p>
                      <ul className="space-y-1 text-white/70 text-xs">
                        {reasoning.reasoning_log.map((step, i) => (
                          <li key={i} className="flex gap-2">
                            <span className="text-white/40 shrink-0">{i + 1}.</span>
                            <span>{step}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {(reasoning.efficiency_score != null || reasoning.recommended_cooling_delta != null) && (
                    <div className="flex gap-4 text-xs">
                      {reasoning.efficiency_score != null && (
                        <span className="text-white/60">Efficiency score: <strong className="text-white">{reasoning.efficiency_score}</strong></span>
                      )}
                      {reasoning.recommended_cooling_delta != null && (
                        <span className="text-white/60">Cooling delta: <strong className="text-white">{reasoning.recommended_cooling_delta > 0 ? "+" : ""}{(reasoning.recommended_cooling_delta * 100).toFixed(0)}%</strong></span>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <p className="text-white/50">
                  {reasoning
                    ? (reasoning.diagnostic_reasoning || "No thermal history yet. Reasoning will appear once telemetry is flowing.")
                    : "Loading reasoning…"}
                </p>
              )}
            </div>
          )}
        </div>

        <div className="mt-4 rounded-lg border border-white/10 bg-white/5 p-4">
          <p className="text-xs font-medium text-white/70 mb-1">CooledAI vs Control: Efficiency</p>
          <p className="text-xs text-white/50 leading-relaxed">
            CooledAI runs cooler on average (2–3°C lower) by predicting workload and adjusting fans proactively. Control stays flat because it runs fans at a constant higher rate. Energy savings come from <strong className="text-white/70">fan power</strong>: power ∝ RPM³, so lower RPM when safe = less cooling energy. CooledAI may show brief temp spikes when inference starts—fans ramp up in response—but overall uses less fan energy than always-on high cooling.
          </p>
        </div>
      </motion.section>

      {/* CSV Export — up to 60 days for efficiency evaluation */}
      <ExportCSVSection getToken={getToken} authError={!!authError} />
    </div>
  );
}

export default function PortalOverviewPage() {
  return (
    <Suspense fallback={<div className="p-6 md:p-8 max-w-6xl mx-auto animate-pulse text-white/50">Loading…</div>}>
      <PortalOverviewContent />
    </Suspense>
  );
}
