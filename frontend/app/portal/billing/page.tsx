"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
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
  const [leadModalOpen, setLeadModalOpen] = useState(false);

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
