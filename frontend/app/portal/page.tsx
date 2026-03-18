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

// Match telemetry heartbeat (5s) so UI refreshes on each new packet
const STATS_POLL_MS = 5_000;
const PULSE_STORAGE_KEY = "COOLEDAI_PULSE_DATA";
const PULSE_STORAGE_VERSION = 3; // bump to invalidate stale data
const MAX_STORED_POINTS = 60_480; // 7 days at 10s interval
const EFFICIENCY_DELTA_WINDOW_MS = 30_000; // Instantaneous: last 30 seconds

type PulseRange = "1h" | "6h" | "24h";

const RANGE_MS: Record<PulseRange, number> = {
  "1h": 60 * 60 * 1000,
  "6h": 6 * 60 * 60 * 1000,
  "24h": 24 * 60 * 60 * 1000,
};

type LiveStats = {
  efficiency_gain_pct: number;
  last_telemetry_at: number | null;
  power_reclaimed_kwh: number;
  power_reclaimed_watts: number;
  estimated_annual_savings_usd: number;
  has_live_data: boolean;
  uptime_hours: number;
  pilot_node: { node_id: string; fan_rpm: number; fan_power_watts: number; temp_c: number | null; last_seen_s_ago: number | null };
  baseline_node: { node_id: string; fan_rpm: number; fan_power_watts: number; temp_c: number | null; source: string; last_seen_s_ago: number | null };
};

type PulsePoint = { time: string; pilot: number; baseline: number; delta?: number; ts: number };

function PortalOverviewContent() {
  const searchParams = useSearchParams();
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);

  const [efficiencyGain, setEfficiencyGain] = useState<number | null>(null);
  const [efficiencyLoading, setEfficiencyLoading] = useState(true);
  const [powerReclaimed, setPowerReclaimed] = useState(0);
  const [annualSavings, setAnnualSavings] = useState(0);
  const [hasLiveData, setHasLiveData] = useState(false);
  const [pilotWatts, setPilotWatts] = useState(0);
  const [baselineWatts, setBaselineWatts] = useState(0);
  const [pilotNodeId, setPilotNodeId] = useState("");
  const [baselineNodeId, setBaselineNodeId] = useState("");
  const [pulseRange, setPulseRange] = useState<PulseRange>("1h");
  const [aggregatedData, setAggregatedData] = useState<PulsePoint[]>([]);
  const [aggregatedLoading, setAggregatedLoading] = useState(false);
  const [pulseData, setPulseData] = useState<PulsePoint[]>([]);

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
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const res = await api.getStats();
      if (!res.ok) {
        console.warn("[CooledAI] /api/v1/stats returned", res.status);
        return;
      }
      const data: LiveStats = await res.json();
      console.log("[CooledAI] stats response:", JSON.stringify(data, null, 2));

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
      setPilotNodeId(data.pilot_node.node_id || "");
      setBaselineNodeId(data.baseline_node.node_id || "");

      // Only add point when BOTH temps are present — avoid defaulting to 0 (which produces bogus deltas)
      if (pilotTemp != null && baselineTemp != null) {
        const now = Date.now();
        const point: PulsePoint = {
          time: new Date(now).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          pilot: pilotTemp,
          baseline: baselineTemp,
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
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, STATS_POLL_MS);
    return () => clearInterval(id);
  }, [fetchStats]);

  // Fetch aggregated hourly data when 6H or 24H selected
  const fetchAggregated = useCallback(async (hours: 6 | 24) => {
    setAggregatedLoading(true);
    try {
      const res = await api.getThermalHistory(hours);
      if (!res.ok) {
        setAggregatedData([]);
        setAggregatedLoading(false);
        return;
      }
      const data = await res.json();
      const buckets = (data.buckets || []) as { ts: number; hour_label: string; pilot: number; baseline: number }[];
      setAggregatedData(
        buckets.map((b) => ({
          time: b.hour_label,
          pilot: b.pilot,
          baseline: b.baseline,
          delta: b.baseline - b.pilot,
          ts: b.ts,
        }))
      );
    } catch {
      setAggregatedData([]);
    } finally {
      setAggregatedLoading(false);
    }
  }, []);

  useEffect(() => {
    if (pulseRange === "6h") fetchAggregated(6);
    else if (pulseRange === "24h") fetchAggregated(24);
  }, [pulseRange, fetchAggregated]);

  useEffect(() => {
    if (searchParams.get("success") === "true") {
      setShowSuccessBanner(true);
      const t = setTimeout(() => setShowSuccessBanner(false), 8000);
      return () => clearTimeout(t);
    }
  }, [searchParams]);

  const now = Date.now();
  const rangeMs = RANGE_MS[pulseRange];
  const displayedPulseData =
    pulseRange === "1h"
      ? pulseData.filter((p) => now - p.ts <= rangeMs)
      : aggregatedData;

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
            {efficiencyLoading ? (
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
            {hasLiveData
              ? `${pilotWatts.toFixed(0)}W CooledAI vs ${baselineWatts.toFixed(0)}W Control`
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

      {/* Thermal Chart — 1H Live Pulse, 6H/24H aggregated hourly */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="rounded-xl border border-white/10 bg-[#141414] p-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="text-base font-semibold text-white">Thermal Chart</h2>
            <p className="text-sm text-white/50 mt-0.5">
              {pilotNodeId || "CooledAI (Predictive)"} vs {baselineNodeId || "Control (Traditional)"}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex gap-1 rounded-lg border border-white/10 bg-white/5 p-1">
              {(["1h", "6h", "24h"] as const).map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setPulseRange(r)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                    pulseRange === r ? "bg-accent-cyan/20 text-accent-cyan" : "text-white/60 hover:text-white/80"
                  }`}
                >
                  {r === "1h" ? "1H" : r === "6h" ? "6H" : "24H"}
                </button>
              ))}
            </div>
            <span className="inline-flex items-center gap-2 text-xs text-white/60">
              <span className="w-2 h-2 rounded-full bg-[#22c55e]" />
              {hasLiveData && (
                <span className="inline-flex items-center gap-1.5">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22c55e] opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#22c55e]" />
                  </span>
                  <span className="text-[#22c55e]/90 font-medium">Active</span>
                </span>
              )}
              CooledAI (Predictive)
            </span>
            <span className="inline-flex items-center gap-2 text-xs text-white/60">
              <span className="w-2 h-2 rounded-full bg-[#ef4444]" /> Control (Traditional)
            </span>
          </div>
        </div>
        {aggregatedLoading && pulseRange !== "1h" ? (
          <div className="h-[320px] flex items-center justify-center text-white/40 text-sm">
            <span className="inline-flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Loading aggregated data…
            </span>
          </div>
        ) : displayedPulseData.length === 0 ? (
          <div className="h-[320px] flex items-center justify-center text-white/40 text-sm">
            {pulseRange === "1h"
              ? (pulseData.length === 0 ? "Waiting for telemetry…" : "No data in selected range. Try a shorter range.")
              : "No aggregated data yet. Telemetry will build history over time."}
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
                domain={[0, "dataMax + 10"]}
                tickFormatter={(v) => `${v}°C`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1a1a",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                }}
                labelStyle={{ color: "rgba(255,255,255,0.8)" }}
                formatter={(value, name) => {
                  if (name === "delta") return [`${value != null ? value : 0}°C Δ`, "Efficiency Delta"];
                  return [
                    `${value != null ? value : 0}°C`,
                    name === "pilot" ? "CooledAI (Predictive)" : "Control (Traditional)",
                  ];
                }}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <ReferenceLine y={65} stroke="rgba(234,179,8,0.5)" strokeDasharray="4 4" />
              <ReferenceLine y={85} stroke="rgba(239,68,68,0.5)" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="pilot"
                name="pilot"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="baseline"
                name="baseline"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
              />
              <Legend
                wrapperStyle={{ fontSize: 12 }}
                formatter={(value) => (value === "pilot" ? "CooledAI (Predictive)" : "Control (Traditional)")}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        )}
        <p className="text-xs text-white/40 mt-3">
          Yellow line: warning (65°C). Red line: critical (85°C).
        </p>
      </motion.section>
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
