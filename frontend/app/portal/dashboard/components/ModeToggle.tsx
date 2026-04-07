"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { apiFetch } from "@/lib/api";

type Mode = "shadow" | "supervised" | "active";

interface ModeState {
  mode: Mode;
  available_modes: Mode[];
  can_activate: boolean;
  activation_blockers: string[];
  confidence_pct: number;
  data_hours: number;
  fan_rpm_floor_pct: number;
  thermal_ceiling_c: number;
  changed_at: string;
  changed_by: string;
}

const MODE_LABELS: Record<Mode, { title: string; desc: string }> = {
  shadow: {
    title: "Shadow (Observe Only)",
    desc: "BMC controls fans. CooledAI observes thermal patterns and simulates recommendations.",
  },
  supervised: {
    title: "Supervised (Guided Optimization)",
    desc: "CooledAI controls fans with max 20% reduction from BMC baseline. Generates calibration data.",
  },
  active: {
    title: "Active (Full Optimization)",
    desc: "CooledAI optimizes fan speeds within ASHRAE limits. Rate-of-change monitoring + 85\u00B0C ceiling.",
  },
};

const CONFIRM_MSG: Record<string, string> = {
  "shadow->supervised":
    "CooledAI will begin controlling fan speeds (max 20% below BMC baseline). Rate-of-change safety active. Continue?",
  "shadow->active":
    "CooledAI will fully optimize fan speeds. Rate-of-change monitoring + auto-fallback at 85\u00B0C. Continue?",
  "supervised->active":
    "CooledAI will fully optimize fan speeds. Rate-of-change monitoring + auto-fallback at 85\u00B0C. Continue?",
  "active->shadow":
    "CooledAI will stop controlling fans. BMC resumes full control. Continue?",
  "supervised->shadow":
    "CooledAI will stop controlling fans. BMC resumes full control. Continue?",
  "active->supervised":
    "CooledAI will limit fan reductions to max 20% below BMC baseline. Continue?",
};

export function ModeToggle({
  token,
  getToken,
}: {
  token?: string | null;
  getToken?: () => Promise<string | null>;
}) {
  const [state, setState] = useState<ModeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [selected, setSelected] = useState<Mode>("shadow");
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);
  const mountedRef = useRef(true);

  /** Resolve a fresh Clerk JWT (or use the static token prop). */
  const resolveToken = useCallback(async (): Promise<string | null> => {
    if (getToken) {
      try { return await getToken(); } catch { return null; }
    }
    return token ?? null;
  }, [getToken, token]);

  const fetchMode = useCallback(async () => {
    try {
      const t = await resolveToken();
      const res = await apiFetch("/api/v1/gateway/mode", {
        headers: t ? { Authorization: `Bearer ${t}` } : undefined,
      });
      if (!mountedRef.current) return;
      if (res.ok) {
        const data = await res.json();
        setState(data);
        setSelected(data.mode);
      }
    } catch {
      // ignore
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [resolveToken]);

  useEffect(() => {
    mountedRef.current = true;
    fetchMode();
    const id = setInterval(fetchMode, 30_000);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [fetchMode]);

  const applyMode = async () => {
    if (!state || selected === state.mode) return;
    const key = `${state.mode}->${selected}`;
    const msg = CONFIRM_MSG[key] || `Switch to ${selected} mode?`;
    if (!confirm(msg)) return;

    setSaving(true);
    try {
      const t = await resolveToken();
      const res = await apiFetch("/api/v1/gateway/mode", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(t ? { Authorization: `Bearer ${t}` } : {}),
        },
        body: JSON.stringify({ mode: selected }),
      });
      const data = await res.json();
      if (res.ok) {
        setToast({ msg: data.message || `Mode changed to ${selected}`, ok: true });
        await fetchMode();
      } else {
        setToast({ msg: data.detail || "Failed to change mode", ok: false });
      }
    } catch {
      setToast({ msg: "Network error", ok: false });
    } finally {
      setSaving(false);
    }
    setTimeout(() => setToast(null), 5000);
  };

  if (loading) {
    return (
      <div className="rounded-xl border border-white/10 bg-[#141414] p-5">
        <div className="h-24 animate-pulse rounded bg-white/5" />
      </div>
    );
  }

  const isDisabled = (m: Mode) =>
    m !== "shadow" && state && !state.can_activate;

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-base font-semibold text-white">Operational Mode</h3>
        {state && (
          <span className="text-xs text-white/40">
            Since{" "}
            {new Date(state.changed_at).toLocaleString("en-US", {
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        )}
      </div>

      <div className="space-y-2 mb-4">
        {(["shadow", "supervised", "active"] as Mode[]).map((m) => (
          <label
            key={m}
            className={`flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors ${
              selected === m
                ? "border-[#00FFCC]/50 bg-[#00FFCC]/5"
                : isDisabled(m)
                ? "border-white/5 bg-white/[0.02] opacity-50 cursor-not-allowed"
                : "border-white/10 bg-white/[0.02] hover:border-white/20"
            }`}
          >
            <input
              type="radio"
              name="mode"
              value={m}
              checked={selected === m}
              disabled={isDisabled(m) ?? false}
              onChange={() => setSelected(m)}
              className="mt-1 accent-[#00FFCC]"
            />
            <div>
              <div className="text-sm font-medium text-white">
                {MODE_LABELS[m].title}
              </div>
              <div className="text-xs text-white/50">{MODE_LABELS[m].desc}</div>
              {isDisabled(m) && state?.activation_blockers?.length ? (
                <div className="mt-1 text-xs text-amber-400/80">
                  {state.activation_blockers[0]}
                </div>
              ) : null}
            </div>
          </label>
        ))}
      </div>

      {/* Prerequisites */}
      {state && (
        <div className="flex gap-4 text-xs text-white/50 mb-3">
          <span>
            Confidence:{" "}
            <span className={state.confidence_pct >= 80 ? "text-emerald-400" : "text-amber-400"}>
              {state.confidence_pct}%{state.confidence_pct >= 80 ? " \u2713" : ""}
            </span>
          </span>
          <span>
            Data:{" "}
            <span className={state.data_hours >= 1 ? "text-emerald-400" : "text-amber-400"}>
              {state.data_hours}h{state.data_hours >= 1 ? " \u2713" : ""}
            </span>
          </span>
        </div>
      )}

      {/* Auto-fallback note */}
      <div className="text-xs text-white/30 mb-3">
        Auto-fallback: If GPU &gt; {state?.thermal_ceiling_c ?? 85}&deg;C, reverts to Shadow automatically.
      </div>

      {/* Apply button */}
      {selected !== state?.mode && (
        <button
          onClick={applyMode}
          disabled={saving || (isDisabled(selected) ?? false)}
          className="w-full rounded-lg bg-[#00FFCC]/10 border border-[#00FFCC]/30 py-2 text-sm font-medium text-[#00FFCC] hover:bg-[#00FFCC]/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {saving ? "Applying..." : `Switch to ${MODE_LABELS[selected].title.split(" (")[0]}`}
        </button>
      )}

      {/* Toast */}
      {toast && (
        <div
          className={`mt-3 rounded-lg px-3 py-2 text-xs ${
            toast.ok
              ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400"
              : "bg-red-500/10 border border-red-500/20 text-red-400"
          }`}
        >
          {toast.msg}
        </div>
      )}
    </div>
  );
}
