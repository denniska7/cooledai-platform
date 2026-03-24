"use client";

import type { EfficiencyTrend } from "@/lib/types/dashboard";
import { TrendArrow } from "./TrendArrow";

export function MetricCard({
  label,
  value,
  sublabel,
  trend,
  valueColor,
  loading,
}: {
  label: string;
  value: string;
  sublabel?: string;
  trend?: EfficiencyTrend;
  valueColor?: string;
  loading?: boolean;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-6">
      <p className="text-sm text-white/50 uppercase tracking-wider mb-1">
        {label}
      </p>
      {loading ? (
        <div className="space-y-2 animate-pulse">
          <div className="h-9 w-32 rounded bg-white/10" />
          {sublabel !== undefined && (
            <div className="h-4 w-48 rounded bg-white/5" />
          )}
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2">
            <p
              className="text-3xl font-bold tabular-nums"
              style={{ color: valueColor || "white" }}
            >
              {value}
            </p>
            {trend && <TrendArrow direction={trend} />}
          </div>
          {sublabel && (
            <p className="text-xs text-white/40 mt-1">{sublabel}</p>
          )}
        </>
      )}
    </div>
  );
}
