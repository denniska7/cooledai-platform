"use client";

import { useEffect, useState } from "react";

type Status = "loading" | "live" | "offline";

async function checkHealth(): Promise<boolean> {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) return false;
  try {
    const base = url.replace(/\/$/, "");
    const res = await fetch(`${base}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

export function BackendStatus() {
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
      ? "bg-green-500"
      : status === "offline"
        ? "bg-red-500"
        : "bg-gray-400 animate-pulse";

  const label =
    status === "live" ? "Live" : status === "offline" ? "Offline" : "Checking…";

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
      <span
        className={`h-2 w-2 rounded-full ${dotColor}`}
        title={`Backend: ${label}`}
        aria-hidden
      />
      Backend Status: {label}
    </span>
  );
}
