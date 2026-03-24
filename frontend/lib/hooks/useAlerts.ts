"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { DashboardSummary, Alert } from "@/lib/types/dashboard";

const DISMISSED_KEY = "COOLEDAI_DISMISSED_ALERTS";
const ACHIEVEMENT_AUTO_DISMISS_S = 86400;

function loadDismissed(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = localStorage.getItem(DISMISSED_KEY);
    if (!raw) return new Set();
    return new Set(JSON.parse(raw) as string[]);
  } catch {
    return new Set();
  }
}

function saveDismissed(ids: Set<string>) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(DISMISSED_KEY, JSON.stringify([...ids]));
  } catch {
    // localStorage full or unavailable
  }
}

export function useAlerts(data: DashboardSummary | null) {
  const [dismissed, setDismissed] = useState<Set<string>>(() => loadDismissed());
  const prevDataRef = useRef<DashboardSummary | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);

  const deriveAlerts = useCallback(
    (summary: DashboardSummary): Alert[] => {
      const now = Date.now();
      const result: Alert[] = [];

      if (summary.thermal_trend_alert) {
        result.push({
          id: "thermal_trend",
          category: "awareness",
          message: "GPU running warmer than 30-day average",
          detail: `7-day avg ${summary.thermal_7d_avg_c.toFixed(1)}\u00B0C vs 30-day ${summary.thermal_30d_avg_c.toFixed(1)}\u00B0C`,
          timestamp: now,
        });
      }

      if (summary.fan_bearing_status === "alert") {
        result.push({
          id: "fan_bearing_alert",
          category: "action",
          message: "Fan showing wear pattern \u2014 schedule inspection",
          detail: `Fan RPM std dev: ${summary.fan_rpm_std_dev.toFixed(1)}`,
          timestamp: now,
        });
      } else if (summary.fan_bearing_status === "monitor") {
        result.push({
          id: "fan_bearing_monitor",
          category: "awareness",
          message: "Fan bearing monitoring \u2014 early wear detected",
          detail: `Fan RPM std dev: ${summary.fan_rpm_std_dev.toFixed(1)}`,
          timestamp: now,
        });
      }

      if (summary.system_status === "red") {
        result.push({
          id: "system_red",
          category: "action",
          message: summary.system_status_message || "System requires attention",
          timestamp: now,
        });
      }

      if (
        summary.calibration_confidence_pct != null &&
        summary.calibration_confidence_pct < 50
      ) {
        result.push({
          id: "calibration_low",
          category: "awareness",
          message: "Model confidence low \u2014 recalibrating",
          detail: `Current confidence: ${summary.calibration_confidence_pct}%`,
          timestamp: now,
        });
      }

      if (summary.efficiency_gate_events_24h > 10) {
        result.push({
          id: "efficiency_gate",
          category: "awareness",
          message: "Efficiency gate active \u2014 protecting performance",
          detail: `${summary.efficiency_gate_events_24h} events in last 24h`,
          timestamp: now,
        });
      }

      // Filter out dismissed alerts
      return result.filter((a) => !dismissed.has(a.id));
    },
    [dismissed]
  );

  useEffect(() => {
    if (!data) return;
    setAlerts(deriveAlerts(data));
    prevDataRef.current = data;
  }, [data, deriveAlerts]);

  // Auto-dismiss achievements after 24h
  useEffect(() => {
    const achievements = alerts.filter((a) => a.category === "achievement");
    if (achievements.length === 0) return;

    const timers = achievements.map((a) => {
      const age = (Date.now() - a.timestamp) / 1000;
      const remaining = Math.max(0, ACHIEVEMENT_AUTO_DISMISS_S - age);
      return setTimeout(() => {
        dismissAlert(a.id);
      }, remaining * 1000);
    });

    return () => timers.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alerts]);

  const dismissAlert = useCallback((id: string) => {
    setDismissed((prev) => {
      const next = new Set(prev);
      next.add(id);
      saveDismissed(next);
      return next;
    });
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }, []);

  return { alerts, dismissAlert };
}
