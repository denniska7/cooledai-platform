"use client";

import { Suspense, useState, useEffect, useCallback } from "react";
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

const STATS_POLL_MS = 10_000;
const MAX_PULSE_POINTS = 30;

type LiveStats = {
  efficiency_gain_pct: number;
  power_reclaimed_kwh: number;
  power_reclaimed_watts: number;
  estimated_annual_savings_usd: number;
  has_live_data: boolean;
  uptime_hours: number;
  pilot_node: { node_id: string; fan_rpm: number; fan_power_watts: number; temp_c: number | null; last_seen_s_ago: number | null };
  baseline_node: { node_id: string; fan_rpm: number; fan_power_watts: number; temp_c: number | null; source: string; last_seen_s_ago: number | null };
};

type PulsePoint = { time: string; pilot: number; baseline: number };

function PortalOverviewContent() {
  const searchParams = useSearchParams();
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);

  const [efficiencyGain, setEfficiencyGain] = useState(0);
  const [powerReclaimed, setPowerReclaimed] = useState(0);
  const [annualSavings, setAnnualSavings] = useState(0);
  const [hasLiveData, setHasLiveData] = useState(false);
  const [pilotWatts, setPilotWatts] = useState(0);
  const [baselineWatts, setBaselineWatts] = useState(0);
  const [pulseData, setPulseData] = useState<PulsePoint[]>([]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await api.getStats();
      if (!res.ok) return;
      const data: LiveStats = await res.json();
      setEfficiencyGain(data.efficiency_gain_pct);
      setPowerReclaimed(data.power_reclaimed_kwh);
      setAnnualSavings(data.estimated_annual_savings_usd);
      setHasLiveData(data.has_live_data);
      setPilotWatts(data.pilot_node.fan_power_watts);
      setBaselineWatts(data.baseline_node.fan_power_watts);

      if (data.has_live_data && data.pilot_node.temp_c != null) {
        const now = new Date();
        const point: PulsePoint = {
          time: now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          pilot: data.pilot_node.temp_c,
          baseline: data.baseline_node.temp_c ?? 0,
        };
        setPulseData((prev) => [...prev.slice(-(MAX_PULSE_POINTS - 1)), point]);
      }
    } catch {
      /* API unreachable — keep last known values */
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, STATS_POLL_MS);
    return () => clearInterval(id);
  }, [fetchStats]);

  useEffect(() => {
    if (searchParams.get("success") === "true") {
      setShowSuccessBanner(true);
      const t = setTimeout(() => setShowSuccessBanner(false), 8000);
      return () => clearTimeout(t);
    }
  }, [searchParams]);

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
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
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
            {efficiencyGain.toFixed(1)}%
          </p>
          <p className="text-xs text-white/40 mt-1">
            {hasLiveData
              ? `${pilotWatts.toFixed(0)}W pilot vs ${baselineWatts.toFixed(0)}W baseline`
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
      </div>

      {/* Live System Pulse — Ambient vs Chip Temp */}
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="rounded-xl border border-white/10 bg-[#141414] p-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h2 className="text-base font-semibold text-white">Live System Pulse</h2>
            <p className="text-sm text-white/50 mt-0.5">Pilot (CooledAI) vs Baseline (BIOS Auto) — live</p>
          </div>
          <div className="flex gap-3">
            <span className="inline-flex items-center gap-2 text-xs text-white/60">
              <span className="w-2 h-2 rounded-full bg-[#22c55e]" /> Pilot
            </span>
            <span className="inline-flex items-center gap-2 text-xs text-white/60">
              <span className="w-2 h-2 rounded-full bg-[#ef4444]" /> Baseline
            </span>
          </div>
        </div>
        {pulseData.length === 0 ? (
          <div className="h-[320px] flex items-center justify-center text-white/40 text-sm">
            Waiting for telemetry…
          </div>
        ) : (
        <div className="h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={pulseData}
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
                formatter={(value, name) => [
                  `${value != null ? value : 0}°C`,
                  name === "pilot" ? "Pilot (CooledAI)" : "Baseline (BIOS Auto)",
                ]}
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
                formatter={(value) => (value === "pilot" ? "Pilot (CooledAI)" : "Baseline (BIOS Auto)")}
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
