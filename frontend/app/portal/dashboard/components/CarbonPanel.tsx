"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import type { DashboardSummary } from "@/lib/types/dashboard";

export function CarbonPanel({
  data,
  loading,
}: {
  data: DashboardSummary | null;
  loading: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [exporting, setExporting] = useState(false);

  const savedKwh =
    data && data.electricity_rate > 0
      ? data.saved_this_month_usd / data.electricity_rate
      : 0;
  const co2AvoidedKg = savedKwh * (data?.carbon_intensity_kg_per_kwh ?? 0);
  const annualCo2Kg = co2AvoidedKg * 12;

  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await api.getThermalExport({ days: 30 });
      if (res.ok) {
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition");
        const match = disposition?.match(/filename="?([^";\n]+)"?/);
        const filename = match?.[1] || "cooledai_esg_export.csv";
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {
      // ignore
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414]">
      {/* Header toggle */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between p-6 text-left"
      >
        <h2 className="text-xl font-semibold text-white">
          Carbon & ESG Impact
        </h2>
        <svg
          width={20}
          height={20}
          viewBox="0 0 24 24"
          fill="none"
          stroke="white"
          strokeWidth={2}
          className={`transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-6">
              {loading ? (
                <div className="space-y-3 animate-pulse">
                  <div className="h-6 w-48 rounded bg-white/10" />
                  <div className="h-4 w-32 rounded bg-white/5" />
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="rounded-xl border border-white/10 bg-[#0a0a0a] p-4">
                    <p className="text-sm text-white/50 mb-1">
                      CO2 Avoided This Month
                    </p>
                    <p className="text-2xl font-bold text-[#22c55e] tabular-nums">
                      {co2AvoidedKg.toFixed(1)} kg
                    </p>
                    <p className="text-xs text-white/40 mt-1">
                      Based on {savedKwh.toFixed(1)} kWh saved &times;{" "}
                      {data?.carbon_intensity_kg_per_kwh ?? 0} kg/kWh
                    </p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-[#0a0a0a] p-4">
                    <p className="text-sm text-white/50 mb-1">
                      Projected Annual CO2 Reduction
                    </p>
                    <p className="text-2xl font-bold text-[#00FFCC] tabular-nums">
                      {annualCo2Kg >= 1000
                        ? `${(annualCo2Kg / 1000).toFixed(1)} tonnes`
                        : `${annualCo2Kg.toFixed(0)} kg`}
                    </p>
                  </div>
                </div>
              )}
              <div className="mt-4">
                <button
                  type="button"
                  onClick={handleExport}
                  disabled={exporting}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-white/70 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50"
                >
                  {exporting ? (
                    <svg
                      className="animate-spin h-3.5 w-3.5"
                      viewBox="0 0 24 24"
                      fill="none"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                  ) : (
                    <svg
                      width={14}
                      height={14}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  )}
                  Export ESG Report
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
