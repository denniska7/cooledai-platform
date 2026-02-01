"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

type Metrics = {
  efficiency_score: number;
  thermal_lag_seconds: number;
  state: string;
  nodes_active: number;
  cooling_delta: number;
} | null;

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

function useSimulatedMetrics(): { temp: number; efficiency: number; powerSaved: number } {
  const [metrics, setMetrics] = useState<Metrics>(null);
  const [simulated, setSimulated] = useState({ temp: 42.3, efficiency: 87.2, powerSaved: 124 });

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      const data = await fetchMetrics();
      if (mounted && data) {
        setMetrics(data);
        setSimulated({
          temp: 38 + data.efficiency_score * 0.1 + Math.random() * 2,
          efficiency: data.efficiency_score,
          powerSaved: Math.round(100 + data.efficiency_score * 2 + Math.random() * 20),
        });
      } else if (mounted) {
        setSimulated((s) => ({
          temp: s.temp + (Math.random() - 0.5) * 0.5,
          efficiency: Math.min(95, Math.max(80, s.efficiency + (Math.random() - 0.5) * 2)),
          powerSaved: s.powerSaved + Math.round((Math.random() - 0.5) * 10),
        }));
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  return simulated;
}

export function LiveSystemPulse() {
  const { temp, efficiency, powerSaved } = useSimulatedMetrics();

  const items = [
    { label: "Current Temp", value: `${temp.toFixed(1)}°C`, unit: "" },
    { label: "Efficiency Gain", value: `${efficiency.toFixed(1)}`, unit: "%" },
    { label: "Power Saved", value: powerSaved.toString(), unit: " kW" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="rounded border border-white/20 bg-black/50 p-8"
    >
      <div className="mb-6 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-accent-cyan animate-pulse" />
        <span className="text-xs font-medium uppercase tracking-widest text-accent-cyan">
          Live System Pulse
        </span>
      </div>
      <div className="grid grid-cols-3 gap-8">
        {items.map((item, i) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + i * 0.1, duration: 0.4 }}
            className="border-l border-white/20 pl-6"
          >
            <p className="text-xs text-white/50 uppercase tracking-wider">{item.label}</p>
            <p className="mt-1 text-2xl font-medium tracking-tight text-white">
              {item.value}
              {item.unit}
            </p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
