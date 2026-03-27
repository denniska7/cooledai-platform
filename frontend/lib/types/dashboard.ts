export type SystemStatus = "green" | "amber" | "red";
export type EfficiencyTrend = "up" | "down" | "flat";
export type SavingsConfidence = "HIGH" | "MEDIUM" | "LOW";
export type BearingStatus = "healthy" | "monitor" | "alert";

export type DashboardSummary = {
  saved_this_month_usd: number;
  savings_confidence: SavingsConfidence;
  savings_confidence_reason: string;
  system_efficiency_today_pct: number;
  efficiency_trend: EfficiencyTrend;
  projected_annual_savings_low_usd: number;
  projected_annual_savings_high_usd: number;
  system_status: SystemStatus;
  system_status_message: string;
  current_saving_watts: number;
  fan_energy_reduction_pct: number;
  gpu_energy_reduction_pct: number;
  temperature_advantage_c: number;
  thermal_7d_avg_c: number;
  thermal_30d_avg_c: number;
  thermal_trend_alert: boolean;
  fan_rpm_std_dev: number;
  fan_bearing_status: BearingStatus;
  calibration_last_updated: string | null;
  calibration_confidence_pct: number | null;
  efficiency_gate_events_24h: number;
  idle_optimization_periods_24h: number;
  idle_optimization_wh_saved_24h: number;
  electricity_rate: number;
  currency: string;
  carbon_intensity_kg_per_kwh: number;
  operational_mode?: "shadow" | "supervised" | "active";
  projected_monthly_usd?: number;
  projected_annual_usd?: number;
};

export type SavingsChartPoint = {
  ts: number;
  optimized_w: number;
  baseline_w: number;
  delta_w: number;
  gpu_pilot_c?: number;
  gpu_baseline_c?: number;
  cpu_pilot_c?: number;
  cpu_baseline_c?: number;
};

export type Alert = {
  id: string;
  category: "action" | "awareness" | "achievement";
  message: string;
  detail?: string;
  timestamp: number;
  dismissed?: boolean;
};
