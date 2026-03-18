"use client";

import { motion } from "framer-motion";
import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";

function LeafIcon({ className }: { className?: string }) {
  return (
    <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
    </svg>
  );
}

export default function FacilityPulsePage() {
  const [efficiencyDelta, setEfficiencyDelta] = useState<number | null>(null);
  const [efficiencyGain, setEfficiencyGain] = useState<number | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await api.getStats();
      if (!res.ok) return;
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
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 5000);
    return () => clearInterval(id);
  }, [fetchStats]);

  // CO2 saved (kg) — rough: ~0.5 kg CO2 per kWh saved, efficiency delta contributes to power savings
  const co2SavedKg = efficiencyDelta != null && efficiencyGain != null
    ? Math.round((efficiencyGain / 100) * 12 * 0.5) // ~12 kWh/day baseline * efficiency * 0.5 kg/kWh
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
        {/* PUE */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.05 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-3">PUE (Power Usage Effectiveness)</p>
          <div className="flex items-baseline gap-4">
            <div>
              <p className="text-2xl font-bold text-[#22c55e]">1.14</p>
              <p className="text-xs text-white/50 mt-0.5">Pilot</p>
            </div>
            <span className="text-white/30">vs</span>
            <div>
              <p className="text-2xl font-bold text-[#ef4444]">1.28</p>
              <p className="text-xs text-white/50 mt-0.5">Control</p>
            </div>
          </div>
        </motion.div>

        {/* Acoustic Load */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="rounded-xl border border-white/10 bg-[#141414] p-6"
        >
          <p className="text-xs font-medium uppercase tracking-wider text-white/50 mb-3">Acoustic Load</p>
          <div className="flex items-baseline gap-4">
            <div>
              <p className="text-2xl font-bold text-[#22c55e]">42 dB</p>
              <p className="text-xs text-[#22c55e]/80 mt-0.5">Quiet</p>
              <p className="text-xs text-white/50 mt-0.5">Pilot</p>
            </div>
            <span className="text-white/30">vs</span>
            <div>
              <p className="text-2xl font-bold text-[#ef4444]">61 dB</p>
              <p className="text-xs text-[#ef4444]/80 mt-0.5">Loud</p>
              <p className="text-xs text-white/50 mt-0.5">Control</p>
            </div>
          </div>
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
    </div>
  );
}
