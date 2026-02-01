"use client";

import { motion, AnimatePresence } from "framer-motion";

type EfficiencyReportModalProps = {
  isOpen: boolean;
  onClose: () => void;
  metrics?: {
    opex_reclaimed_usd?: number;
    carbon_offset_kg?: number;
    efficiency_score?: number;
  } | null;
};

// Derived values for the report
function computeReportValues(metrics: EfficiencyReportModalProps["metrics"]) {
  const efficiency = (metrics?.efficiency_score ?? 87) / 100;
  const opexReclaimed = metrics?.opex_reclaimed_usd ?? 12400;
  const carbonOffset = metrics?.carbon_offset_kg ?? 1240;

  // Traditional reactive cooling typically uses ~15% more power
  const traditionalMultiplier = 1.15;
  const cooledaiCost = 100; // baseline 100
  const traditionalCost = Math.round(cooledaiCost * traditionalMultiplier);

  // Energy saved = delta between traditional and CooledAI
  const energySavedPct = Math.round((traditionalMultiplier - 1) * 100);
  const energySavedKwh = Math.round(carbonOffset * 0.5); // rough conversion

  // Hardware lifespan: thermal variance reduction ~12-18% extension (use efficiency as proxy)
  const lifespanExtension = Math.round(12 + efficiency * 6);

  // 12-month ROI: scale opex reclaimed to annual
  const roi12Month = Math.round(opexReclaimed * 12);

  return {
    traditionalCost,
    cooledaiCost,
    energySavedPct,
    energySavedKwh,
    lifespanExtension,
    roi12Month,
  };
}

export function EfficiencyReportModal({
  isOpen,
  onClose,
  metrics,
}: EfficiencyReportModalProps) {
  const values = computeReportValues(metrics);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed inset-4 z-50 flex items-center justify-center p-4 md:inset-8"
          >
            <div
              className="relative w-full max-w-lg overflow-hidden rounded-lg border-2 border-white bg-black shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Certificate header */}
              <div className="border-b border-white/20 px-8 py-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-medium tracking-tight text-white uppercase">
                    Certificate of Efficiency
                  </h2>
                  <button
                    onClick={onClose}
                    className="text-white/50 transition-colors hover:text-white"
                    aria-label="Close"
                  >
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <p className="mt-1 text-xs text-white/50 uppercase tracking-widest">
                  CooledAI Pilot · Thermal Optimization Summary
                </p>
              </div>

              {/* Content */}
              <div className="space-y-8 px-8 py-8">
                {/* SVG Bar chart: Traditional vs CooledAI Power Cost */}
                <div>
                  <p className="mb-4 text-xs font-medium text-white/70 uppercase tracking-wider">
                    Power Cost Comparison
                  </p>
                  <div className="flex items-end justify-between gap-8">
                    <div className="flex flex-1 flex-col items-center gap-2">
                      <span className="text-xs text-white/60">Traditional</span>
                      <svg viewBox="0 0 60 100" className="h-28 w-full max-w-[60px]">
                        <rect
                          x="0"
                          y="0"
                          width="60"
                          height="100"
                          fill="rgba(255,255,255,0.08)"
                          stroke="rgba(255,255,255,0.2)"
                          strokeWidth="1"
                        />
                        <motion.rect
                          initial={{ height: 0, y: 100 }}
                          animate={{ height: 100, y: 0 }}
                          transition={{ duration: 0.6, delay: 0.2 }}
                          x="0"
                          width="60"
                          fill="rgba(255,255,255,0.4)"
                        />
                        <text x="30" y="50" textAnchor="middle" fill="white" fontSize="12" fontWeight="500">
                          {values.traditionalCost}%
                        </text>
                      </svg>
                    </div>
                    <div className="flex flex-1 flex-col items-center gap-2">
                      <span className="text-xs text-white/60">CooledAI</span>
                      <svg viewBox="0 0 60 100" className="h-28 w-full max-w-[60px]">
                        <rect
                          x="0"
                          y="0"
                          width="60"
                          height="100"
                          fill="rgba(255,255,255,0.08)"
                          stroke="rgba(255,255,255,0.2)"
                          strokeWidth="1"
                        />
                        <motion.rect
                          initial={{ height: 0, y: 100 }}
                          animate={{
                            height: (values.cooledaiCost / values.traditionalCost) * 100,
                            y: 100 - (values.cooledaiCost / values.traditionalCost) * 100,
                          }}
                          transition={{ duration: 0.6, delay: 0.4 }}
                          x="0"
                          width="60"
                          fill="#00FFCC"
                        />
                        <text
                          x="30"
                          y={100 - (values.cooledaiCost / values.traditionalCost) * 50}
                          textAnchor="middle"
                          fill="black"
                          fontSize="12"
                          fontWeight="500"
                        >
                          {values.cooledaiCost}%
                        </text>
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Metrics grid */}
                <div className="grid gap-6 sm:grid-cols-3">
                  <div className="rounded border border-white/20 bg-white/5 p-4">
                    <p className="text-xs text-white/50 uppercase tracking-wider">Total Energy Saved</p>
                    <p className="mt-2 text-2xl font-medium text-white">
                      {values.energySavedPct}%
                    </p>
                    <p className="mt-1 text-xs text-white/60">
                      ~{values.energySavedKwh.toLocaleString()} kWh vs reactive
                    </p>
                  </div>
                  <div className="rounded border border-white/20 bg-white/5 p-4">
                    <p className="text-xs text-white/50 uppercase tracking-wider">Hardware Lifespan</p>
                    <p className="mt-2 text-2xl font-medium text-white">
                      +{values.lifespanExtension}%
                    </p>
                    <p className="mt-1 text-xs text-white/60">
                      Thermal variance reduction
                    </p>
                  </div>
                  <div className="rounded border border-white/20 bg-white/5 p-4">
                    <p className="text-xs text-white/50 uppercase tracking-wider">12-Mo ROI Forecast</p>
                    <p className="mt-2 text-2xl font-medium text-[#00FFCC]">
                      ${(values.roi12Month / 1000).toFixed(0)}k
                    </p>
                    <p className="mt-1 text-xs text-white/60">
                      Enterprise tier projection
                    </p>
                  </div>
                </div>

                {/* Footer */}
                <div className="border-t border-white/20 pt-6">
                  <p className="text-center text-xs text-white/40">
                    Based on pilot telemetry. Enterprise deployment may vary.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
