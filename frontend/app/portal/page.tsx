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
        <SavingsProofPanel data={data} loading={loading} />
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
