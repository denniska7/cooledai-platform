"use client";

import type { DashboardSummary } from "@/lib/types/dashboard";
import { MetricCard } from "./MetricCard";
import { StatusPill } from "./StatusPill";

function formatUSD(n: number): string {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
}

export function HeroMetrics({
  data,
  loading,
  onStatusClick,
}: {
  data: DashboardSummary | null;
  loading: boolean;
  onStatusClick: () => void;
}) {
  const isShadow = data?.operational_mode === "shadow";

  // Card 1: Saved This Month (with projected in shadow mode)
  let savedValue = "--";
  let savedSublabel: string | undefined;
  if (data) {
    if (data.saved_this_month_usd > 0) {
      if (data.savings_confidence === "HIGH") {
        savedValue = formatUSD(data.saved_this_month_usd);
      } else {
        const low = Math.round(data.saved_this_month_usd * 0.8);
        const high = Math.round(data.saved_this_month_usd * 1.2);
        savedValue = `${formatUSD(low)} \u2013 ${formatUSD(high)}`;
      }
    } else if (isShadow && data.projected_monthly_usd && data.projected_monthly_usd > 0) {
      savedValue = "$0";
      savedSublabel = `Projected if active: ~${formatUSD(data.projected_monthly_usd)}/mo`;
    } else {
      savedValue = "$0";
      savedSublabel = isShadow ? "Switch to Active to start saving" : undefined;
    }
  }

  // Card 2: System Efficiency (show calibration progress in shadow)
  let efficiencyValue = "--";
  if (data) {
    if (data.system_efficiency_today_pct > 0) {
      efficiencyValue = `${data.system_efficiency_today_pct.toFixed(1)}%`;
    } else if (isShadow) {
      efficiencyValue = `Calibrating (${data.calibration_confidence_pct ?? 0}%)`;
    } else {
      efficiencyValue = "0.0%";
    }
  }

  // Card 3: Projected Annual Savings
  let annualValue = "--";
  if (data) {
    if (data.projected_annual_savings_low_usd > 0) {
      annualValue = `${formatUSD(data.projected_annual_savings_low_usd)} \u2013 ${formatUSD(data.projected_annual_savings_high_usd)}`;
    } else if (isShadow && data.projected_annual_usd && data.projected_annual_usd > 0) {
      annualValue = `~${formatUSD(data.projected_annual_usd)}`;
    } else {
      annualValue = "$0";
    }
  }

  return (
    <div className="relative">
      {/* Status pill top-right */}
      <div className="flex justify-end mb-3">
        <StatusPill
          status={data?.system_status ?? "green"}
          message={data?.system_status_message}
          onClick={onStatusClick}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="$ Saved This Month"
          value={savedValue}
          sublabel={savedSublabel}
          valueColor="#22c55e"
          loading={loading}
        />
        <MetricCard
          label="System Efficiency Today"
          value={efficiencyValue}
          trend={data?.efficiency_trend}
          loading={loading}
        />
        <MetricCard
          label="Projected Annual Savings"
          value={annualValue}
          valueColor="#00FFCC"
          loading={loading}
        />
      </div>
    </div>
  );
}
