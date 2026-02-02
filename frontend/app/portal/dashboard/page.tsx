"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PortalSidebar } from "./components/PortalSidebar";
import { PredictiveGauge } from "./components/PredictiveGauge";
import { LiveTelemetry } from "./components/LiveTelemetry";
import { SafetyShieldToggle } from "./components/SafetyShieldToggle";
import { SavingsTicker } from "./components/SavingsTicker";
import { SystemLog } from "./components/SystemLog";
import { PowerReclaimedChart } from "./components/PowerReclaimedChart";
import { EfficiencyReportModal } from "./components/EfficiencyReportModal";
import { EfficiencyScore } from "./components/EfficiencyScore";
import { SevenDayTestProgress } from "./components/SevenDayTestProgress";
import { useInterval } from "../../../lib/useInterval";

type Metrics = {
  efficiency_score: number;
  thermal_lag_seconds: number;
  state: string;
  nodes_active: number;
  cooling_delta: number;
  cpu_temp_avg?: number;
  delta_t_inlet?: number;
  delta_t_outlet?: number;
  power_draw_kw?: number;
  predicted_load_t10?: number;
  current_capacity?: number;
  carbon_offset_kg?: number;
  opex_reclaimed_usd?: number;
  simulation_critical?: boolean;
} | null;

type LogEntry = { entry: string; timestamp: number; critical?: boolean } | null;

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

async function fetchAiLog(): Promise<LogEntry> {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) return null;
  try {
    const base = url.replace(/\/$/, "");
    const res = await fetch(`${base}/simulated-metrics/log`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    return { entry: data.entry, timestamp: data.timestamp, critical: data.critical };
  } catch {
    return null;
  }
}

export default function PortalDashboardPage() {
  const [metrics, setMetrics] = useState<Metrics>(null);
  const [aiLog, setAiLog] = useState<LogEntry>(null);
  const [safetyShield, setSafetyShield] = useState(true);
  const [hasSeenSimulation, setHasSeenSimulation] = useState(false);
  const [reportModalOpen, setReportModalOpen] = useState(false);

  useEffect(() => {
    if (aiLog?.critical) setHasSeenSimulation(true);
  }, [aiLog?.critical]);
  useEffect(() => {
    if (metrics && (metrics.efficiency_score ?? metrics.opex_reclaimed_usd ?? metrics.carbon_offset_kg) != null) {
      setHasSeenSimulation(true);
    }
  }, [metrics]);

  const poll = useCallback(async () => {
    const [m, log] = await Promise.all([fetchMetrics(), fetchAiLog()]);
    if (m) setMetrics(m);
    if (log) setAiLog(log);
  }, []);

  useInterval(poll, 2000);

  const predictedLoad = metrics?.predicted_load_t10 ?? 85;
  const currentCapacity = metrics?.current_capacity ?? 100;
  const carbonOffset = metrics?.carbon_offset_kg ?? 1240;
  const opexReclaimed = metrics?.opex_reclaimed_usd ?? 12400;

  return (
    <div
      className={`min-h-screen bg-transparent flex flex-col md:flex-row min-h-[100dvh] transition-all duration-500 ${
        safetyShield
          ? "ring-2 ring-[#00FFCC] ring-inset animate-shield-pulse"
          : "ring-2 ring-white ring-inset"
      }`}
    >
      <PortalSidebar />
      <main className="flex-1 overflow-auto p-6 md:p-8">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <h1 className="text-2xl font-medium tracking-tight text-white">
              Mission Control
            </h1>
            <div className="flex items-center gap-4">
              {hasSeenSimulation && (
                <>
                  <button
                    onClick={() => setReportModalOpen(true)}
                    className="rounded border border-[#00FFCC] bg-[#00FFCC]/10 px-4 py-2 text-sm font-medium tracking-tight text-[#00FFCC] transition-opacity hover:bg-[#00FFCC]/20"
                  >
                    Download Efficiency Audit
                  </button>
                </>
              )}
              <SafetyShieldToggle enabled={safetyShield} onToggle={setSafetyShield} />
            </div>
          </div>

          {/* Efficiency Score */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
          >
            <EfficiencyScore
              score={metrics?.efficiency_score ?? 94}
              reclaimedPowerKw={Math.round(70 + (metrics?.efficiency_score ?? 94) * 0.8) || 142}
            />
          </motion.section>

          {/* 7-Day Test Progress */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.02 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
          >
            <SevenDayTestProgress
              currentDay={Math.min(7, Math.max(1, Math.floor((metrics?.efficiency_score ?? 30) / 15) + 1))}
            />
          </motion.section>

          {/* Predictive Gauge */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-8"
          >
            <h2 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-6">
              Predicted Thermal Load (T+10 min) vs Current Capacity
            </h2>
            <div className="flex justify-center">
              <PredictiveGauge
                predictedLoad={predictedLoad}
                currentCapacity={currentCapacity}
                size={220}
              />
            </div>
          </motion.section>

          {/* Live Telemetry */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.05 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
          >
            <h2 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-4">
              Live Telemetry
            </h2>
            <LiveTelemetry data={metrics} />
          </motion.section>

          {/* Savings Ticker */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
          >
            <h2 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-4">
              Savings Ticker
            </h2>
            <SavingsTicker
              carbonOffsetKg={carbonOffset}
              opexReclaimedUsd={opexReclaimed}
            />
          </motion.section>

          {/* Power Reclaimed Chart */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.15 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
          >
            <h2 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-4">
              Power Reclaimed
            </h2>
            <PowerReclaimedChart />
          </motion.section>

          {/* System Log */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
          >
            <h2 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-4">
              System Log
            </h2>
            <AnimatePresence mode="wait">
              {aiLog ? (
                <motion.div
                  key={aiLog.timestamp}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <SystemLog entry={aiLog.entry} timestamp={aiLog.timestamp} critical={aiLog.critical} />
                </motion.div>
              ) : (
                <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-4 font-mono text-sm text-white/50">
                  Waiting for AI decision stream...
                </div>
              )}
            </AnimatePresence>
          </motion.section>
        </div>
      </main>

      <EfficiencyReportModal
        isOpen={reportModalOpen}
        onClose={() => setReportModalOpen(false)}
        metrics={metrics}
      />
    </div>
  );
}
