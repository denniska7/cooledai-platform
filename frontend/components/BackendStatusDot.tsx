"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Status = "loading" | "live" | "offline";

async function checkHealth(): Promise<boolean> {
  try {
    const res = await api.health();
    return res.ok;
  } catch {
    return false;
  }
}

export function BackendStatusDot() {
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let mounted = true;
    checkHealth().then((ok) => {
      if (mounted) setStatus(ok ? "live" : "offline");
    });
    return () => {
      mounted = false;
    };
  }, []);

  const dotColor =
    status === "live"
      ? "bg-accent-lime"
      : status === "offline"
        ? "bg-red-500"
        : "bg-white/40 animate-pulse";

  const label =
    status === "live" ? "Live" : status === "offline" ? "Offline" : "Checking…";

  return (
    <span
      className={`h-2 w-2 rounded-full ${dotColor} border border-white/20`}
      title={`System: ${label}`}
      aria-label={`Backend status: ${label}`}
    />
  );
}
