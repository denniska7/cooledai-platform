"use client";

import { PortalSidebar } from "../dashboard/components/PortalSidebar";

export default function ThermalMapPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col md:flex-row">
      <PortalSidebar />
      <main className="flex-1 overflow-auto p-6 md:p-8">
        <h1 className="text-2xl font-medium tracking-tight text-white mb-6">
          Thermal Map
        </h1>
        <div className="rounded border border-white/20 bg-black/50 p-16 text-center">
          <p className="text-white/50">Thermal map visualization coming soon.</p>
        </div>
      </main>
    </div>
  );
}
