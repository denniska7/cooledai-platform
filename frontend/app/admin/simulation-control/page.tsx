"use client";

import { useState } from "react";
import Link from "next/link";

const CAUTION_ORANGE = "#d97706";
const CAUTION_ORANGE_HOVER = "#f59e0b";

async function triggerSimulation(mode: string): Promise<{ status: string; message?: string }> {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) throw new Error("NEXT_PUBLIC_API_URL not set");
  const base = url.replace(/\/$/, "");
  const res = await fetch(`${base}/admin/simulation-control/trigger`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export default function SimulationControlPage() {
  const [loading, setLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTrigger = async (mode: string) => {
    setLoading(mode);
    setMessage(null);
    setError(null);
    try {
      const data = await triggerSimulation(mode);
      setMessage(data.message || "Triggered.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="min-h-screen bg-black">
      <header className="border-b border-[rgba(255,255,255,0.1)] px-6 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-lg font-medium tracking-tight text-white">
            CooledAI
          </Link>
          <span className="text-xs text-white/40 uppercase tracking-wider">
            Command Center
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-2xl px-6 py-16">
        <h1 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
          Simulation Control
        </h1>
        <p className="mt-2 text-sm text-white/50">
          Trigger chaos scenarios to demonstrate CooledAI response. Open the
          Client Portal in another tab to observe.
        </p>

        <div className="mt-12 space-y-6">
          <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
            <h2 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-4">
              Chaos Triggers
            </h2>
            <div className="space-y-4">
              <button
                onClick={() => handleTrigger("gpu_spike")}
                disabled={!!loading}
                className="w-full rounded border-2 px-6 py-4 text-sm font-medium tracking-tight transition-opacity hover:opacity-90 disabled:opacity-50"
                style={{
                  borderColor: CAUTION_ORANGE,
                  color: CAUTION_ORANGE,
                  backgroundColor: `${CAUTION_ORANGE}15`,
                }}
              >
                {loading === "gpu_spike" ? "Triggering…" : "Simulate GPU Spike"}
              </button>
              <p className="text-xs text-white/50 -mt-2">
                Forces telemetry from 45°C to 82°C instantly.
              </p>

              <button
                onClick={() => handleTrigger("chiller_failure")}
                disabled={!!loading}
                className="w-full rounded border-2 px-6 py-4 text-sm font-medium tracking-tight transition-opacity hover:opacity-90 disabled:opacity-50"
                style={{
                  borderColor: CAUTION_ORANGE,
                  color: CAUTION_ORANGE,
                  backgroundColor: `${CAUTION_ORANGE}15`,
                }}
              >
                {loading === "chiller_failure" ? "Triggering…" : "Simulate Chiller Failure"}
              </button>
              <p className="text-xs text-white/50 -mt-2">
                Steady climb in ambient temp, power draw constant.
              </p>

              <button
                onClick={() => handleTrigger("reset")}
                disabled={!!loading}
                className="w-full rounded border-2 px-6 py-4 text-sm font-medium tracking-tight transition-opacity hover:opacity-90 disabled:opacity-50"
                style={{
                  borderColor: CAUTION_ORANGE,
                  color: CAUTION_ORANGE,
                  backgroundColor: `${CAUTION_ORANGE}15`,
                }}
              >
                {loading === "reset" ? "Resetting…" : "Reset Environment"}
              </button>
              <p className="text-xs text-white/50 -mt-2">
                Returns all metrics to steady-state green levels.
              </p>
            </div>
          </div>

          {message && (
            <div className="rounded border border-[rgba(255,255,255,0.2)] bg-white/5 px-4 py-3 text-sm text-white/80">
              {message}
            </div>
          )}
          {error && (
            <div className="rounded border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>

        <div className="mt-12 rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
          <h3 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-2">
            Observe
          </h3>
          <p className="text-sm text-white/60 leading-relaxed">
            Open the{" "}
            <Link href="/portal" className="text-white underline">
              Client Portal
            </Link>{" "}
            in another tab. When you trigger a spike or chiller failure, the
            portal will reflect the change within 2 seconds. The System Log will
            show the critical AI response.
          </p>
        </div>
      </main>

      <footer className="border-t border-[rgba(255,255,255,0.1)] py-6">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI Command Center · Internal Use Only
        </div>
      </footer>
    </div>
  );
}
