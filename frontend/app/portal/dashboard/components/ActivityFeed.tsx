"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { apiFetch } from "@/lib/api";

interface FeedEntry {
  ts: number;
  type: string;
  node_id: string;
  message: string;
  gpu_temp_c?: number;
  fan_rpm?: number;
  target_rpm?: number;
  saving_w?: number;
  source?: string;
}

export function ActivityFeed({ token }: { token?: string | null }) {
  const [entries, setEntries] = useState<FeedEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);

  const fetchFeed = useCallback(async () => {
    try {
      const res = await apiFetch("/api/v1/activity-feed?limit=15", {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (!mountedRef.current) return;
      if (res.ok) {
        const data = await res.json();
        setEntries(data.entries || []);
      }
    } catch { /* ignore */ }
    finally { if (mountedRef.current) setLoading(false); }
  }, [token]);

  useEffect(() => {
    mountedRef.current = true;
    fetchFeed();
    const id = setInterval(fetchFeed, 10_000);
    return () => { mountedRef.current = false; clearInterval(id); };
  }, [fetchFeed]);

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };

  return (
    <div className="rounded-xl border border-white/10 bg-[#141414] p-5">
      <h3 className="text-base font-semibold text-white mb-3">Brain Activity</h3>
      {loading ? (
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-6 rounded bg-white/5 animate-pulse" />
          ))}
        </div>
      ) : entries.length === 0 ? (
        <p className="text-sm text-white/40">Waiting for brain decisions...</p>
      ) : (
        <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
          {entries.map((e, i) => (
            <div
              key={`${e.ts}-${i}`}
              className="flex items-start gap-2 text-xs"
            >
              <span className="text-white/30 shrink-0 w-[72px] font-mono">
                {formatTime(e.ts)}
              </span>
              <span
                className={`shrink-0 w-1.5 h-1.5 rounded-full mt-1.5 ${
                  e.type === "optimization" && e.saving_w && e.saving_w > 0
                    ? "bg-emerald-400"
                    : "bg-white/20"
                }`}
              />
              <span className="text-white/70">{e.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
