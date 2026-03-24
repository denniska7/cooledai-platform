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
  // Card 1: Saved This Month
  let savedValue = "--";
  let savedSublabel: string | undefined;
  if (data) {
    if (data.savings_confidence === "HIGH") {
      savedValue = formatUSD(data.saved_this_month_usd);
    } else if (data.savings_confidence === "MEDIUM") {
      const low = Math.round(data.saved_this_month_usd * 0.8);
      const high = Math.round(data.saved_this_month_usd * 1.2);
      savedValue = `${formatUSD(low)} \u2013 ${formatUSD(high)}`;
    } else {
      const low = Math.round(data.saved_this_month_usd * 0.5);
      const high = Math.round(data.saved_this_month_usd * 1.5);
      savedValue = `${formatUSD(low)} \u2013 ${formatUSD(high)}`;
      savedSublabel =
        "Workload shifted \u2014 estimate based on fan savings only";
    }
  }

  // Card 2: System Efficiency
  const efficiencyValue = data
    ? `${data.system_efficiency_today_pct.toFixed(1)}%`
    : "--";

  // Card 3: Projected Annual Savings
  const annualValue = data
    ? `${formatUSD(data.projected_annual_savings_low_usd)} \u2013 ${formatUSD(data.projected_annual_savings_high_usd)}`
    : "--";

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
