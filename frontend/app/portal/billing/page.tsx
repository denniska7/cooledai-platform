"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useAuth } from "@clerk/nextjs";
import { api } from "@/lib/api";
import { LeadFormModal } from "../../../components/LeadFormModal";

const tiers = [
  {
    name: "Audit",
    price: "$0",
    period: "7-day shadow period",
    description: "Full visibility into your facility’s thermal and power profile. No commitment.",
    features: [
      "7-day shadow audit",
      "Baseline PUE & thermal map",
      "Savings opportunity report",
      "No credit card required",
    ],
    cta: "Current plan",
    ctaAction: "none" as const,
    active: true,
    highlight: true,
  },
  {
    name: "Optimizer Pro",
    price: "Custom",
    period: "per site / month",
    description: "Single-site AI optimization. Setpoints, sequencing, and reclaim margins.",
    features: [
      "Everything in Audit",
      "AI-driven setpoint optimization",
      "Real-time thermal predictions",
      "Single-site license",
    ],
    cta: "Contact Sales",
    ctaAction: "contact" as const,
    active: false,
    highlight: false,
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: "multi-site",
    description: "Fleet-wide optimization, SLAs, and dedicated support for large deployments.",
    features: [
      "Everything in Optimizer Pro",
      "Multi-site fleet management",
      "Dedicated success manager",
      "Custom SLAs & reporting",
    ],
    cta: "Request Shadow Audit",
    ctaAction: "audit" as const,
    active: false,
    highlight: false,
  },
];

export default function BillingPage() {
  const { getToken } = useAuth();
  const [leadModalOpen, setLeadModalOpen] = useState(false);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [projection, setProjection] = useState<Record<string, unknown> | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const token = await getToken();
      const [sRes, pRes] = await Promise.all([
        api.getStats(token),
        api.getSavingsProjection(token),
      ]);
      if (sRes.ok) setStats(await sRes.json());
      if (pRes.ok) setProjection(await pRes.json());
    } catch { /* ignore */ }
  }, [getToken]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const nodesMonitored = 2;
  const uptimeH = (stats?.uptime_hours as number) ?? 0;
  const savedKwh = (stats?.power_reclaimed_kwh as number) ?? 0;
  const annualCo2 = (projection?.projected_co2_avoided_kg as number) ?? 0;
  const annualUsd = (projection?.projected_annual_usd_saved as number) ?? 0;

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h1 className="text-2xl font-semibold tracking-tight text-white">Plan Selection</h1>
        <p className="text-sm text-white/50 mt-0.5">Choose the tier that fits your facility</p>
      </motion.div>

      {/* Usage Summary */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4"
      >
        {[
          { label: "Nodes Monitored", value: String(nodesMonitored) },
          { label: "Uptime", value: `${uptimeH.toFixed(1)}h` },
          { label: "Energy Saved", value: `${savedKwh.toFixed(2)} kWh` },
          { label: "CO\u2082 Avoided", value: `${annualCo2.toFixed(1)} kg/yr` },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-white/10 bg-[#141414] p-4">
            <p className="text-xs text-white/50 uppercase tracking-wider">{s.label}</p>
            <p className="text-lg font-bold text-white mt-1">{s.value}</p>
          </div>
        ))}
      </motion.div>

      {annualUsd > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mt-4 rounded-xl border border-[#00FFCC]/20 bg-[#00FFCC]/5 p-4 text-sm text-white/80"
        >
          Projected savings: <span className="font-semibold text-[#00FFCC]">${annualUsd.toFixed(0)}/year</span> at current rates ($0.12/kWh)
        </motion.div>
      )}

      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        {tiers.map((tier, i) => (
          <motion.div
            key={tier.name}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.05 * i }}
            className={`rounded-xl border flex flex-col overflow-hidden ${
              tier.highlight
                ? "border-[#22c55e]/50 bg-[#141414] ring-1 ring-[#22c55e]/20"
                : "border-white/10 bg-[#141414]"
            }`}
          >
            {tier.active && (
              <div className="bg-[#22c55e]/15 border-b border-[#22c55e]/30 px-6 py-2 text-center">
                <span className="text-xs font-semibold uppercase tracking-wider text-[#22c55e]">
                  Active
                </span>
              </div>
            )}
            <div className="p-6 flex flex-col flex-1">
              <h2 className="text-lg font-semibold text-white">{tier.name}</h2>
              <div className="mt-2 flex items-baseline gap-1">
                <span className="text-2xl font-bold text-white">{tier.price}</span>
                <span className="text-sm text-white/50">{tier.period}</span>
              </div>
              <p className="text-sm text-white/60 mt-3">{tier.description}</p>
              <ul className="mt-4 space-y-2 flex-1">
                {tier.features.map((f) => (
                  <li key={f} className="flex items-start gap-2 text-sm text-white/80">
                    <span className="text-[#22c55e] mt-0.5 shrink-0" aria-hidden>
                      ✓
                    </span>
                    {f}
                  </li>
                ))}
              </ul>
              <div className="mt-6 pt-4 border-t border-white/10">
                {tier.ctaAction === "audit" ? (
                  <Link
                    href="/audit-request"
                    className="block w-full rounded-lg border border-white/20 bg-white/5 px-4 py-2.5 text-center text-sm font-medium text-white/80 hover:bg-white/10 hover:text-white transition-colors"
                  >
                    {tier.cta}
                  </Link>
                ) : tier.ctaAction === "contact" ? (
                  <button
                    type="button"
                    onClick={() => setLeadModalOpen(true)}
                    className="w-full rounded-lg border border-white/20 bg-white/5 px-4 py-2.5 text-sm font-medium text-white/80 hover:bg-white/10 hover:text-white transition-colors"
                  >
                    {tier.cta}
                  </button>
                ) : (
                  <span
                    className={`inline-block w-full rounded-lg border px-4 py-2.5 text-center text-sm font-medium ${
                      tier.active
                        ? "border-[#22c55e]/40 bg-[#22c55e]/10 text-[#22c55e]"
                        : "border-white/20 bg-white/5 text-white/80"
                    }`}
                  >
                    {tier.cta}
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="mt-8 rounded-xl border border-[#22c55e]/20 bg-[#22c55e]/5 p-6"
      >
        <p className="text-sm text-white/90">
          <span className="font-semibold text-[#22c55e]">Note:</span> Subscription costs are
          typically covered by the first 14 days of recovered energy margins.
        </p>
      </motion.div>

      <LeadFormModal
        isOpen={leadModalOpen}
        onClose={() => setLeadModalOpen(false)}
        title="Get in touch"
      />
    </div>
  );
}
