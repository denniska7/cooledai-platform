"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { apiFetch } from "@/lib/api";
import type { DashboardSummary, SavingsChartPoint } from "@/lib/types/dashboard";

const CHART_POLL_MS = 60_000;

export function SavingsProofPanel({
  data,
  loading,
}: {
  data: DashboardSummary | null;
  loading: boolean;
}) {
  const [chartData, setChartData] = useState<SavingsChartPoint[]>([]);
  const [chartLoading, setChartLoading] = useState(true);
  const mountedRef = useRef(true);

  const fetchChart = useCallback(async () => {
    try {
      const res = await apiFetch("/api/v1/savings-chart?hours=24");
      if (!mountedRef.current) return;
      if (res.ok) {
        const json = await res.json();
        setChartData(json.points || json || []);
      }
    } catch {
      // ignore
    } finally {
      if (mountedRef.current) setChartLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchChart();
    const id = setInterval(fetchChart, CHART_POLL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [fetchChart]);

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-6">
      <h2 className="text-xl font-semibold text-white mb-4">
        What CooledAI Is Doing Right Now
      </h2>

      {/* Chart */}
      <div className="relative">
        {chartLoading ? (
          <div className="h-[300px] flex items-center justify-center">
            <div className="animate-spin h-6 w-6 border-2 border-white/20 border-t-[#22c55e] rounded-full" />
          </div>
        ) : (
          <div className="relative">
            {/* Floating label */}
            {data && data.current_saving_watts > 0 && (
              <div className="absolute top-2 right-2 z-10 bg-[#1a1a1a]/90 border border-white/10 rounded-lg px-3 py-1.5">
                <span className="text-sm font-medium text-[#22c55e]">
                  Saving {data.current_saving_watts}W right now
                </span>
              </div>
            )}
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData}>
                <defs>
                  <linearGradient id="savingsGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  stroke="rgba(255,255,255,0.05)"
                  strokeDasharray="3 3"
                />
                <XAxis
                  dataKey="ts"
                  tickFormatter={formatTime}
                  tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                  tickLine={false}
                />
                <YAxis
                  yAxisId="watts"
                  tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                  tickLine={false}
                  label={{
                    value: "Watts",
                    angle: -90,
                    position: "insideLeft",
                    fill: "rgba(255,255,255,0.3)",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  yAxisId="temp"
                  orientation="right"
                  tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                  tickLine={false}
                  domain={["dataMin - 5", "dataMax + 5"]}
                  label={{
                    value: "\u00B0C",
                    angle: 90,
                    position: "insideRight",
                    fill: "rgba(255,255,255,0.3)",
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1a1a1a",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "8px",
                    color: "white",
                    fontSize: 12,
                  }}
                  labelFormatter={formatTime}
                />
                <Area
                  yAxisId="watts"
                  type="monotone"
                  dataKey="baseline_w"
                  fill="url(#savingsGradient)"
                  stroke="none"
                />
                <Line
                  yAxisId="watts"
                  type="monotone"
                  dataKey="baseline_w"
                  stroke="#6b7280"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  name="Baseline Fan W"
                />
                <Line
                  yAxisId="watts"
                  type="monotone"
                  dataKey="optimized_w"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="CooledAI Fan W"
                />
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="cpu_pilot_c"
                  stroke="#f97316"
                  strokeWidth={1.5}
                  dot={false}
                  name="CooledAI CPU \u00B0C"
                />
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="cpu_baseline_c"
                  stroke="#f97316"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  name="Control CPU \u00B0C"
                  opacity={0.5}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Sub-cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
        {/* Fan Energy */}
        <div className="rounded-xl border border-white/10 bg-[#0a0a0a] p-4">
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#3b82f6"
                strokeWidth={1.5}
                className="opacity-80"
              >
                <path d="M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" />
                <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">Fan Energy</p>
              {loading ? (
                <div className="h-4 w-40 rounded bg-white/10 animate-pulse mt-1" />
              ) : (
                <p className="text-xs text-white/50 mt-1">
                  Running at {data?.fan_energy_reduction_pct ?? 0}% less energy
                  than standard cooling
                </p>
              )}
            </div>
          </div>
        </div>

        {/* GPU Energy */}
        <div className="rounded-xl border border-white/10 bg-[#0a0a0a] p-4">
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#a855f7"
                strokeWidth={1.5}
                className="opacity-80"
              >
                <rect x="4" y="4" width="16" height="16" rx="2" />
                <rect x="8" y="8" width="8" height="8" rx="1" />
                <path d="M2 9h2M2 15h2M20 9h2M20 15h2M9 2v2M15 2v2M9 20v2M15 20v2" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">GPU Energy</p>
              {loading ? (
                <div className="h-4 w-40 rounded bg-white/10 animate-pulse mt-1" />
              ) : (
                <>
                  <p className="text-xs text-white/50 mt-1">
                    {data?.gpu_energy_reduction_pct ?? 0}% reduction during idle
                    periods
                  </p>
                  <p className="text-xs text-white/40 mt-0.5">
                    Full performance maintained during active compute
                  </p>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Temperature */}
        <div className="rounded-xl border border-white/10 bg-[#0a0a0a] p-4">
          <div className="flex items-start gap-3">
            <div className="shrink-0 mt-0.5">
              <svg
                width={20}
                height={20}
                viewBox="0 0 24 24"
                fill="none"
                stroke="#f97316"
                strokeWidth={1.5}
                className="opacity-80"
              >
                <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-white">Temperature</p>
              {loading ? (
                <div className="h-4 w-40 rounded bg-white/10 animate-pulse mt-1" />
              ) : (
                <>
                  <p className="text-xs text-white/50 mt-1">
                    GPUs run {data?.temperature_advantage_c ?? 0}&deg;C cooler
                  </p>
                  <p className="text-xs text-white/40 mt-0.5">
                    Lower temps = longer hardware lifespan
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
