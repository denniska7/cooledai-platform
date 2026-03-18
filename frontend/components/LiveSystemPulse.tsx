"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import { useAuth } from "@clerk/nextjs";

export function LiveSystemPulse() {
  const [temp, setTemp] = useState<number | null>(null);
  const [powerSaved, setPowerSaved] = useState(0);
  const [isLive, setIsLive] = useState(false);

  const { getToken } = useAuth();

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const token = await getToken();
        if (!token) return;

        const res = await api.getStats(token);
        if (!res.ok || !mounted) {
          if (res.status === 401) {
            setIsLive(false);
            setTemp(null);
            setPowerSaved(0);
          }
          return;
        }
        const data = await res.json();
        console.log("[CooledAI Pulse] stats:", data.has_live_data, "pilot_temp:", data.pilot_node?.temp_c, "watts:", data.power_reclaimed_watts);
        const t = data.pilot_node?.temp_c;
        if (t != null) {
          setTemp(t);
          setPowerSaved(Math.round(data.power_reclaimed_watts ?? 0));
          setIsLive(true);
        }
      } catch (err) {
        console.error("[CooledAI Pulse] fetch failed:", err);
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [getToken]);

  const items = [
    { label: "Current Temp", value: temp != null ? `${temp.toFixed(1)}°C` : "—", unit: "" },
    { label: "Energy savings", value: isLive ? "Active" : "Connecting…", unit: "" },
    { label: "Power saved", value: `${powerSaved}`, unit: " W" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="rounded border border-white/20 bg-black/50 p-5 sm:p-8"
    >
      <div className="mb-6 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-accent-cyan animate-pulse" />
        <span className="text-xs font-medium uppercase tracking-widest text-accent-cyan">
          Live System Pulse
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 sm:gap-8">
        {items.map((item, i) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + i * 0.1, duration: 0.4 }}
            className="border-l border-white/20 pl-4 sm:pl-6"
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
