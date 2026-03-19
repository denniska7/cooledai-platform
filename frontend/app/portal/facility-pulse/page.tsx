"use client";

import { motion } from "framer-motion";
import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@clerk/nextjs";

function LeafIcon({ className }: { className?: string }) {
  return (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
    </svg>
  );
}

type TrackedNode = { node_id: string; predictive: boolean; status: string; last_seen_s_ago: number | null; temp_c: number | null };

export default function FacilityPulsePage() {
  const [efficiencyDelta, setEfficiencyDelta] = useState<number | null>(null);
  const [efficiencyGain, setEfficiencyGain] = useState<number | null>(null);
  const [nodes, setNodes] = useState<TrackedNode[]>([]);
  const [hourlyMetrics, setHourlyMetrics] = useState<{
    pilot_pue?: number | null;
    control_pue?: number | null;
    pilot_acoustic_db?: number | null;
    control_acoustic_db?: number | null;
  } | null>(null);
  const [showServers, setShowServers] = useState(false);

  const { getToken } = useAuth();

  const fetchStats = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;

      const res = await api.getStats(token);
      if (!res.ok) {
        if (res.status === 401) {
          setEfficiencyDelta(null);
          setEfficiencyGain(null);
        }
        return;
      }
      const data = await res.json();
      const pilotTemp = data.pilot_node?.temp_c;
      const baselineTemp = data.baseline_node?.temp_c;
      if (pilotTemp != null && baselineTemp != null) {
        setEfficiencyDelta(baselineTemp - pilotTemp);
      }
      setEfficiencyGain(data.efficiency_gain_pct ?? null);
    } catch {
      // ignore
    }
  }, [getToken]);

  const fetchNodesAndHourly = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;

      const [nodesRes, hourlyRes] = await Promise.all([
        api.getNodesList(token),
        api.getHourlyFacilityMetrics(token),
      ]);
      if (nodesRes.ok) {
        const d = await nodesRes.json();
        setNodes(d.nodes || []);
      }
      if (hourlyRes.ok) {
        const d = await hourlyRes.json();
        setHourlyMetrics(d);
      }
    } catch {
      // ignore
    }
  }, [getToken]);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 5000);
    return () => clearInterval(id);
  }, [fetchStats]);

  useEffect(() => {
    fetchNodesAndHourly();
    const id = setInterval(fetchNodesAndHourly, 5 * 60 * 1000);
    return () => clearInterval(id);
  }, [fetchNodesAndHourly]);

  const co2SavedKg = efficiencyDelta != null && efficiencyGain != null
    ? Math.round((efficiencyGain / 100) * 12 * 0.5)
    : null;

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h1 className="text-2xl font-semibold tracking-tight text-white">Facility Pulse</h1>
        <p className="text-sm text-white/50 mt-0.5">Real-time facility health and thermal metrics</p>
      </motion.div>

      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* PUE — hourly average */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.05 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-3">PUE (hourly avg)</p>
          <div className="flex items-baseline gap-4">
            <div>
              <p className="text-2xl font-bold text-[#22c55e]">{hourlyMetrics?.pilot_pue ?? "—"}</p>
              <p className="text-xs text-white/50 mt-0.5">Pilot</p>
            </div>
            <span className="text-white/30">vs</span>
            <div>
              <p className="text-2xl font-bold text-[#ef4444]">{hourlyMetrics?.control_pue ?? "—"}</p>
              <p className="text-xs text-white/50 mt-0.5">Control</p>
            </div>
          </div>
          <p className="text-xs text-white/40 mt-2">Overall average of units (last hour)</p>
        </motion.div>

        {/* Acoustic Load — hourly average */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-3">Acoustic Load (hourly avg)</p>
          <div className="flex items-baseline gap-4">
            <div>
              <p className="text-2xl font-bold text-[#22c55e]">{hourlyMetrics?.pilot_acoustic_db != null ? `${hourlyMetrics.pilot_acoustic_db} dB` : "—"}</p>
              <p className="text-xs text-white/50 mt-0.5">Pilot</p>
            </div>
            <span className="text-white/30">vs</span>
            <div>
              <p className="text-2xl font-bold text-[#ef4444]">{hourlyMetrics?.control_acoustic_db != null ? `${hourlyMetrics.control_acoustic_db} dB` : "—"}</p>
              <p className="text-xs text-white/50 mt-0.5">Control</p>
            </div>
          </div>
          <p className="text-xs text-white/40 mt-2">From fan RPM (last hour)</p>
        </motion.div>

        {/* Carbon Offset */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.15 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-3">Carbon Offset</p>
          <div className="flex items-center gap-3">
            <LeafIcon className="w-8 h-8 text-[#22c55e]" />
            <div>
              <p className="text-2xl font-bold text-[#22c55e]">
                {co2SavedKg != null ? `${co2SavedKg} kg` : "—"}
              </p>
              <p className="text-xs text-white/50 mt-0.5">CO₂ Saved (today)</p>
            </div>
          </div>
          <p className="text-xs text-white/40 mt-2">Based on efficiency delta vs baseline</p>
        </motion.div>
      </div>

      {/* Tracked servers — predictive vs traditional */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="mt-8 rounded-xl border border-white/10 bg-[#141414] p-6"
      >
        <button
          type="button"
          onClick={() => setShowServers(!showServers)}
          className="flex items-center gap-2 text-sm font-medium text-white/80 hover:text-white"
        >
          {showServers ? "Hide" : "Show"} tracked servers
          <svg className={`w-4 h-4 transition-transform ${showServers ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showServers && (
          <div className="mt-4 space-y-2">
            {nodes.length === 0 ? (
              <p className="text-sm text-white/50">No servers reporting yet.</p>
            ) : (
              nodes.map((n) => (
                <div
                  key={n.node_id}
                  className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-4 py-2"
                >
                  <span className="text-sm text-white">{n.node_id}</span>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded ${n.predictive ? "bg-[#22c55e]/20 text-[#22c55e]" : "bg-[#ef4444]/20 text-[#ef4444]"}`}>
                    {n.predictive ? "Predictive" : "Traditional"}
                  </span>
                  <span className={`text-xs ${n.status === "ok" ? "text-[#22c55e]" : "text-white/50"}`}>
                    {n.status === "ok" ? "Live" : "Stale"}
                  </span>
                </div>
              ))
            )}
          </div>
        )}
      </motion.div>
    </div>
  );
}
