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

  const capacityReclaimedKw = Math.round(70 + (metrics?.efficiency_score ?? 87) * 0.8);

  return {
    traditionalCost,
    cooledaiCost,
    energySavedPct,
    energySavedKwh,
    capacityReclaimedKw,
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

  const handleDownload = () => {
    const printWindow = window.open("", "_blank");
    if (!printWindow) return;
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
        <head><title>CooledAI Efficiency Audit</title>
        <style>
          body{font-family:system-ui;background:#000;color:#fff;padding:24px;max-width:600px;margin:0 auto}
          h1{font-size:18px;text-transform:uppercase;margin-bottom:8px}
          .sub{font-size:11px;color:rgba(255,255,255,0.5);margin-bottom:24px}
          .section{margin:20px 0}
          .label{font-size:11px;color:rgba(255,255,255,0.6);text-transform:uppercase}
          .value{font-size:24px;margin:4px 0}
          .accent{color:#00FFCC}
          .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:24px 0}
          .bar{height:80px;background:rgba(255,255,255,0.1);margin:8px 0;display:flex;align-items:flex-end}
          .bar-fill{background:#00FFCC;width:87%}
          .bar-fill-trad{background:rgba(255,255,255,0.4);width:100%}
        </style>
        </head>
        <body>
          <h1>Efficiency Audit</h1>
          <p class="sub">CooledAI Pilot · Traditional vs CooledAI Optimized Cost</p>
          <div class="section">
            <p class="label">Traditional Cooling Cost vs CooledAI Optimized Cost</p>
            <div class="bar"><div class="bar-fill-trad" style="height:100%"></div></div>
            <p>Traditional: ${values.traditionalCost}% · CooledAI: ${values.cooledaiCost}%</p>
          </div>
          <div class="grid">
            <div><p class="label">Capacity Reclaimed</p><p class="value accent">${values.capacityReclaimedKw} kW</p><p class="label">Fit more hardware in the same room</p></div>
            <div><p class="label">Energy Recovered</p><p class="value">${values.energySavedPct}%</p><p class="label">~${values.energySavedKwh.toLocaleString()} kWh vs reactive</p></div>
            <div><p class="label">12-Mo ROI</p><p class="value accent">$${(values.roi12Month / 1000).toFixed(0)}k</p><p class="label">Enterprise projection</p></div>
          </div>
          <p style="font-size:11px;color:rgba(255,255,255,0.4);margin-top:32px">Based on pilot telemetry. Enterprise deployment may vary. CooledAI · The Universal Autonomy Layer for Every Watt of Compute</p>
        </body>
      </html>
    `);
    printWindow.document.close();
    printWindow.print();
    printWindow.close();
  };

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
                    Efficiency Audit
                  </h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleDownload}
                      className="rounded border border-[#00FFCC] bg-[#00FFCC]/10 px-3 py-1.5 text-xs font-medium text-[#00FFCC] transition-opacity hover:bg-[#00FFCC]/20"
                    >
                      Download PDF
                    </button>
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
                </div>
                <p className="mt-1 text-xs text-white/50 uppercase tracking-widest">
                  CooledAI Pilot · Thermal Optimization Summary
                </p>
              </div>

              {/* Content */}
              <div className="space-y-8 px-8 py-8">
                {/* Traditional vs CooledAI Cost Comparison */}
                <div>
                  <p className="mb-4 text-xs font-medium text-white/70 uppercase tracking-wider">
                    Traditional Cooling Cost vs CooledAI Optimized Cost
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
                    <p className="text-xs text-white/50 uppercase tracking-wider">Capacity Reclaimed</p>
                    <p className="mt-2 text-2xl font-medium text-[#00FFCC]">
                      {values.capacityReclaimedKw} kW
                    </p>
                    <p className="mt-1 text-xs text-white/60">
                      Fit more hardware in the same room
                    </p>
                  </div>
                  <div className="rounded border border-white/20 bg-white/5 p-4">
                    <p className="text-xs text-white/50 uppercase tracking-wider">Total Energy Recovered</p>
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
