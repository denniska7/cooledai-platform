"use client";

import { motion } from "framer-motion";
import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@clerk/nextjs";

interface ModeState {
  mode: string;
  confidence_pct: number;
  data_hours: number;
  can_activate: boolean;
  activation_blockers: string[];
}

interface Projection {
  avg_power_reduction_watts: number;
  projected_annual_usd_saved: number;
  projected_monthly_usd_saved: number;
  projected_co2_avoided_kg: number;
  recommendations_with_savings: number;
  total_data_points: number;
  data_hours: number;
}

type Phase = {
  id: string;
  title: string;
  desc: string;
  status: "done" | "current" | "pending";
  detail?: string;
};

export default function SavingsRoadmapPage() {
  const { getToken } = useAuth();
  const [modeState, setModeState] = useState<ModeState | null>(null);
  const [projection, setProjection] = useState<Projection | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const token = await getToken();
      const [mRes, pRes] = await Promise.all([
        api.getGatewayMode(token),
        api.getSavingsProjection(token),
      ]);
      if (mRes.ok) setModeState(await mRes.json());
      if (pRes.ok) setProjection(await pRes.json());
    } catch { /* ignore */ }
  }, [getToken]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 30_000);
    return () => clearInterval(id);
  }, [fetchData]);

  const confidence = modeState?.confidence_pct ?? 0;
  const dataHours = modeState?.data_hours ?? 0;
  const mode = modeState?.mode ?? "shadow";

  const phases: Phase[] = [
    {
      id: "install",
      title: "Installation",
      desc: "Hardware connected, telemetry flowing",
      status: "done",
      detail: `${projection?.total_data_points ?? 0} telemetry points received`,
    },
    {
      id: "calibrate",
      title: "Calibration",
      desc: "Learning your thermal patterns",
      status: confidence >= 80 ? "done" : "current",
      detail: `Confidence: ${confidence}% | Data: ${dataHours.toFixed(1)}h of 1.0h required`,
    },
    {
      id: "shadow",
      title: "Shadow Mode",
      desc: "Observing and recommending, not yet controlling",
      status: mode === "shadow" && confidence >= 80 ? "current" : confidence >= 80 ? "done" : "pending",
      detail: projection
        ? `${projection.recommendations_with_savings} recommendations made | Projected: $${projection.projected_annual_usd_saved.toFixed(0)}/yr if activated`
        : undefined,
    },
    {
      id: "supervised",
      title: "Supervised Mode",
      desc: "Conservative optimization with safety limits",
      status: mode === "supervised" ? "current" : mode === "active" ? "done" : "pending",
      detail: modeState?.can_activate
        ? "Ready to activate"
        : modeState?.activation_blockers?.[0] || "Prerequisites not yet met",
    },
    {
      id: "active",
      title: "Active Mode",
      desc: "Full AI-driven optimization",
      status: mode === "active" ? "current" : "pending",
      detail: projection
        ? `Estimated savings: $${projection.projected_annual_usd_saved.toFixed(0)}/yr | ${projection.projected_co2_avoided_kg.toFixed(0)} kg CO\u2082 avoided`
        : undefined,
    },
  ];

  const statusIcon = (s: Phase["status"]) =>
    s === "done" ? "\u2705" : s === "current" ? "\uD83D\uDD04" : "\u23F3";

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-semibold tracking-tight text-white">Savings Roadmap</h1>
        <p className="text-sm text-white/50 mt-0.5">Your data-driven path to full AI optimization</p>
      </motion.div>

      {/* Projected Savings Card */}
      {projection && projection.projected_annual_usd_saved > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="mt-6 rounded-xl border border-[#00FFCC]/20 bg-[#00FFCC]/5 p-6"
        >
          <p className="text-sm text-[#00FFCC] font-medium mb-1">
            {mode === "active"
              ? "CooledAI is actively saving you money"
              : "If CooledAI were in Active mode today:"}
          </p>
          <p className="text-3xl font-bold text-white">
            ${projection.projected_annual_usd_saved.toFixed(0)}
            <span className="text-base text-white/50 font-normal">/year estimated</span>
          </p>
          <p className="text-xs text-white/40 mt-1">
            {projection.avg_power_reduction_watts.toFixed(1)}W avg reduction | {projection.projected_co2_avoided_kg.toFixed(0)} kg CO&#x2082; avoided annually
          </p>
        </motion.div>
      )}

      {/* Data Collection Progress */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-6 rounded-xl border border-white/10 bg-[#141414] p-5"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-white/70">Data Collection Progress</span>
          <span className="text-xs text-white/40">
            {dataHours.toFixed(1)}h / 1.0h required
          </span>
        </div>
        <div className="h-2 rounded-full bg-white/10 overflow-hidden">
          <div
            className="h-full rounded-full bg-[#00FFCC] transition-all duration-500"
            style={{ width: `${Math.min(100, dataHours * 100)}%` }}
          />
        </div>
        <p className="text-xs text-white/30 mt-2">
          {projection?.total_data_points ?? 0} telemetry points | Confidence: {confidence}%
        </p>
      </motion.div>

      {/* Journey Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-6 space-y-0"
      >
        {phases.map((phase, i) => (
          <div key={phase.id} className="flex gap-4">
            {/* Timeline connector */}
            <div className="flex flex-col items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm border ${
                  phase.status === "done"
                    ? "border-emerald-500/50 bg-emerald-500/10"
                    : phase.status === "current"
                    ? "border-[#00FFCC]/50 bg-[#00FFCC]/10 animate-pulse"
                    : "border-white/10 bg-white/5"
                }`}
              >
                {statusIcon(phase.status)}
              </div>
              {i < phases.length - 1 && (
                <div
                  className={`w-0.5 flex-1 my-1 ${
                    phase.status === "done" ? "bg-emerald-500/30" : "bg-white/10"
                  }`}
                />
              )}
            </div>
            {/* Content */}
            <div className="pb-6 flex-1">
              <div className="rounded-xl border border-white/10 bg-[#141414] p-4">
                <h3 className="text-sm font-semibold text-white">{phase.title}</h3>
                <p className="text-xs text-white/50 mt-0.5">{phase.desc}</p>
                {phase.detail && (
                  <p className="text-xs text-white/40 mt-2">{phase.detail}</p>
                )}
              </div>
            </div>
          </div>
        ))}
      </motion.div>
    </div>
  );
}
