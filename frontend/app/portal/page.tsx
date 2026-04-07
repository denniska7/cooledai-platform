"use client";

import { useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useAlerts } from "@/lib/hooks/useAlerts";
import { HeroMetrics } from "./dashboard/components/HeroMetrics";
import { SavingsProofPanel } from "./dashboard/components/SavingsProofPanel";
import { HardwareIntelligencePanel } from "./dashboard/components/HardwareIntelligencePanel";
import { LivePerformancePanel } from "./dashboard/components/LivePerformancePanel";
import { CarbonPanel } from "./dashboard/components/CarbonPanel";
import { AlertDrawer } from "./dashboard/components/AlertDrawer";
import { ModeToggle } from "./dashboard/components/ModeToggle";
import { ActivityFeed } from "./dashboard/components/ActivityFeed";

export default function PortalPage() {
  const { getToken } = useAuth();
  const { data, loading, error } = useDashboardData(getToken);
  const { alerts, dismissAlert } = useAlerts(data);
  const [drawerOpen, setDrawerOpen] = useState(false);

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-6">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="mb-6">
        <HeroMetrics
          data={data}
          loading={loading}
          onStatusClick={() => setDrawerOpen(true)}
        />
      </div>

      <div className="mb-6">
        <ModeToggle getToken={getToken} />
      </div>

      {/* Mode-aware status banner */}
      {data?.operational_mode === "shadow" && (
        <div className="mb-4 rounded-xl border border-blue-500/20 bg-blue-500/5 px-5 py-3 flex items-center justify-between">
          <div>
            <span className="text-sm text-blue-400 font-medium">CooledAI is in Shadow Mode</span>
            <span className="text-xs text-white/40 ml-2">Currently observing thermal patterns. Switch to Active to start saving energy.</span>
          </div>
        </div>
      )}
      {data?.operational_mode === "active" && (
        <div className="mb-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-5 py-3">
          <span className="text-sm text-emerald-400 font-medium">CooledAI Active — Optimizing fan speeds</span>
          <span className="text-xs text-white/40 ml-2">Auto-fallback enabled (85°C threshold)</span>
        </div>
      )}
      {data?.operational_mode === "supervised" && (
        <div className="mb-4 rounded-xl border border-amber-500/20 bg-amber-500/5 px-5 py-3">
          <span className="text-sm text-amber-400 font-medium">CooledAI Supervised — Conservative fan control</span>
          <span className="text-xs text-white/40 ml-2">Max 10% reduction from baseline</span>
        </div>
      )}

      <div className="mb-6">
        <SavingsProofPanel data={data} loading={loading} />
      </div>

      <div className="mb-6">
        <ActivityFeed />
      </div>

      <div className="mb-6">
        <HardwareIntelligencePanel data={data} loading={loading} />
      </div>

      <div className="mb-6">
        <LivePerformancePanel loading={loading} />
      </div>

      <div className="mb-6">
        <CarbonPanel data={data} loading={loading} />
      </div>

      <AlertDrawer
        alerts={alerts}
        onDismiss={dismissAlert}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  );
}
