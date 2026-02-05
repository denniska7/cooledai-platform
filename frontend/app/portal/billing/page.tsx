"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { SubscribeButton } from "../../../components/SubscribeButton";
import { LemonSqueezyCheckoutOverlay } from "../dashboard/components/LemonSqueezyCheckoutOverlay";

export default function BillingPage() {
  const [manageOpen, setManageOpen] = useState(false);

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h1 className="text-2xl font-semibold tracking-tight text-white">Billing</h1>
        <p className="text-sm text-white/50 mt-0.5">Manage your subscription and invoices</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
        className="mt-8 rounded-xl border border-white/10 bg-[#141414] p-8"
      >
        <div className="flex flex-col gap-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-white">Fleet Optimization</h2>
              <p className="text-sm text-white/50 mt-0.5">7-Day Shadow Audit + AI Optimization</p>
            </div>
            <SubscribeButton />
          </div>
          <div className="border-t border-white/10 pt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-white">Current plan</h2>
              <p className="text-sm text-white/50 mt-0.5">CooledAI Pro</p>
            </div>
            <button
              type="button"
              onClick={() => setManageOpen(true)}
              className="shrink-0 rounded-lg border border-white/20 bg-white/5 px-5 py-2.5 text-sm font-medium text-white/80 hover:bg-white/10 hover:text-white transition-colors"
            >
              Manage Subscription
            </button>
          </div>
        </div>
        <p className="text-xs text-white/40 mt-4">
          Activate Fleet Optimization via Lemon Squeezy checkout. Manage payment and invoices in the overlay.
        </p>
      </motion.div>

      <LemonSqueezyCheckoutOverlay
        isOpen={manageOpen}
        onClose={() => setManageOpen(false)}
      />
    </div>
  );
}
