"use client";

import { Suspense, useState, useMemo, useEffect } from "react";
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

// Simulated live data for demo; replace with API later
function useSimulatedPulse() {
  const [now] = useState(() => Date.now());
  const data = useMemo(() => {
    const points = 24;
    const d = [];
    for (let i = 0; i < points; i++) {
      const t = new Date(now - (points - 1 - i) * 60 * 1000);
      const ambient = 22 + Math.sin(i * 0.4) * 2 + (i / points) * 0.5;
      const chip = Math.min(72, 38 + Math.sin(i * 0.3) * 8 + (i / points) * 6 + (i % 3) * 1.2);
      d.push({
        time: t.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        ambient: Math.round(ambient * 10) / 10,
        chip: Math.round(chip * 10) / 10,
      });
    }
    return d;
  }, [now]);
  return data;
}

function PortalOverviewContent() {
  const searchParams = useSearchParams();
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);
  const pulseData = useSimulatedPulse();
  const [efficiencyGain] = useState(12.4);
  const [powerReclaimed] = useState(8470);
  const [annualSavings] = useState(28400);

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

      {/* Stat cards */}
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
          <p className="text-3xl font-bold text-[#22c55e] tabular-nums">{efficiencyGain}%</p>
          <p className="text-xs text-white/40 mt-1">vs. baseline cooling</p>
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
          <p className="text-3xl font-bold text-[#22c55e] tabular-nums">{powerReclaimed.toLocaleString()} kWh</p>
          <p className="text-xs text-white/40 mt-1">cumulative</p>
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
            ${annualSavings.toLocaleString()}
          </p>
          <p className="text-xs text-white/40 mt-1">projected</p>
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
            <p className="text-sm text-white/50 mt-0.5">Ambient vs chip temperature (last 24 min)</p>
          </div>
          <div className="flex gap-3">
            <span className="inline-flex items-center gap-2 text-xs text-white/60">
              <span className="w-2 h-2 rounded-full bg-[#3b82f6]" /> Ambient
            </span>
            <span className="inline-flex items-center gap-2 text-xs text-white/60">
              <span className="w-2 h-2 rounded-full bg-[#22c55e]" /> Chip
            </span>
          </div>
        </div>
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
                yAxisId="ambient"
                orientation="left"
                tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
                domain={["dataMin - 1", "dataMax + 1"]}
                tickFormatter={(v) => `${v}°C`}
              />
              <YAxis
                yAxisId="chip"
                orientation="right"
                tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
                domain={[30, 85]}
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
                  name === "ambient" ? "Ambient" : "Chip",
                ]}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <ReferenceLine yAxisId="chip" y={65} stroke="rgba(234,179,8,0.5)" strokeDasharray="4 4" />
              <ReferenceLine yAxisId="chip" y={85} stroke="rgba(239,68,68,0.5)" strokeDasharray="4 4" />
              <Line
                yAxisId="ambient"
                type="monotone"
                dataKey="ambient"
                name="ambient"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="chip"
                type="monotone"
                dataKey="chip"
                name="chip"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
              />
              <Legend
                wrapperStyle={{ fontSize: 12 }}
                formatter={(value) => (value === "ambient" ? "Ambient Temp" : "Chip Temp")}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
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
