"use client";

import type { DashboardSummary } from "@/lib/types/dashboard";

function CardShell({
  children,
  loading,
}: {
  children: React.ReactNode;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="rounded-xl border border-white/10 bg-[#141414] p-4 animate-pulse">
        <div className="h-4 w-24 rounded bg-white/10 mb-2" />
        <div className="h-3 w-40 rounded bg-white/5" />
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-4">
      {children}
    </div>
  );
}

export function HardwareIntelligencePanel({
  data,
  loading,
}: {
  data: DashboardSummary | null;
  loading: boolean;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-6">
      <h2 className="text-xl font-semibold text-white mb-4">
        Your Server Health
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* 1. Thermal Trends */}
        <CardShell loading={loading}>
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#f97316"
                strokeWidth={1.5}
              >
                <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">Thermal Trends</p>
              <p className="text-xs text-white/50 mt-1">
                7-day avg: {data?.thermal_7d_avg_c?.toFixed(1) ?? "--"}&deg;C vs
                30-day: {data?.thermal_30d_avg_c?.toFixed(1) ?? "--"}&deg;C
              </p>
              {data?.thermal_trend_alert && (
                <div className="flex items-center gap-1.5 mt-2">
                  <svg
                    width={14}
                    height={14}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth={2}
                  >
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                  <span className="text-xs text-amber-400">
                    Temps trending higher than 30-day baseline
                  </span>
                </div>
              )}
            </div>
          </div>
        </CardShell>

        {/* 2. Fan Bearing Health */}
        <CardShell loading={loading}>
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#3b82f6"
                strokeWidth={1.5}
              >
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">
                Fan Bearing Health
              </p>
              <div className="mt-1.5">
                {data?.fan_bearing_status === "healthy" && (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    Fan vibration: Normal
                  </span>
                )}
                {data?.fan_bearing_status === "monitor" && (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs bg-amber-500/20 text-amber-400 border border-amber-500/30">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    Fan vibration: Elevated
                  </span>
                )}
                {data?.fan_bearing_status === "alert" && (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs bg-red-500/20 text-red-400 border border-red-500/30">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                    Fan vibration: High
                  </span>
                )}
                {!data && (
                  <span className="text-xs text-white/40">--</span>
                )}
              </div>
            </div>
          </div>
        </CardShell>

        {/* 3. Calibration Status */}
        <CardShell loading={loading}>
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke={
                  data &&
                  (data.calibration_confidence_pct == null ||
                    data.calibration_confidence_pct < 70)
                    ? "#f59e0b"
                    : "#22c55e"
                }
                strokeWidth={1.5}
              >
                <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">
                Calibration Status
              </p>
              <p className="text-xs text-white/50 mt-1">
                Last updated:{" "}
                {data?.calibration_last_updated
                  ? new Date(data.calibration_last_updated).toLocaleDateString()
                  : "Never"}
              </p>
              <p className="text-xs text-white/50 mt-0.5">
                Confidence:{" "}
                <span
                  className={
                    data &&
                    (data.calibration_confidence_pct == null ||
                      data.calibration_confidence_pct < 70)
                      ? "text-amber-400"
                      : "text-emerald-400"
                  }
                >
                  {data?.calibration_confidence_pct != null
                    ? `${data.calibration_confidence_pct}%`
                    : "N/A"}
                </span>
              </p>
            </div>
          </div>
        </CardShell>

        {/* 4. Efficiency Gate */}
        <CardShell loading={loading}>
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#8b5cf6"
                strokeWidth={1.5}
              >
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">Efficiency Gate</p>
              <p className="text-xs text-white/50 mt-1">
                {data?.efficiency_gate_events_24h ?? 0} thermal protection events
                today
              </p>
              <p className="text-xs text-white/40 mt-0.5" title="The efficiency gate temporarily pauses optimization when GPU load ramps up, ensuring zero performance impact.">
                Prevents optimization during load ramps
              </p>
            </div>
          </div>
        </CardShell>

        {/* 5. Idle Optimization */}
        <CardShell loading={loading}>
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#22c55e"
                strokeWidth={1.5}
              >
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">
                Idle Optimization
              </p>
              <p className="text-xs text-white/50 mt-1">
                Captured {data?.idle_optimization_periods_24h ?? 0} idle windows,
                saving {data?.idle_optimization_wh_saved_24h ?? 0} Wh
              </p>
            </div>
          </div>
        </CardShell>
      </div>
    </div>
  );
}
