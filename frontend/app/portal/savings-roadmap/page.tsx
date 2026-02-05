"use client";

import { motion } from "framer-motion";

export default function SavingsRoadmapPage() {
  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h1 className="text-2xl font-semibold tracking-tight text-white">Savings Roadmap</h1>
        <p className="text-sm text-white/50 mt-0.5">Your data-driven path to full AI optimization</p>
      </motion.div>
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className="mt-8 rounded-xl border border-white/10 bg-[#141414] p-8 text-center"
      >
        <p className="text-white/50">Savings Roadmap content coming soon.</p>
        <p className="text-sm text-white/40 mt-2">Capacity reclaimed, efficiency gains, and next steps.</p>
      </motion.div>
    </div>
  );
}
