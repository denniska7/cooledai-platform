/**
 * CooledAI API client
 *
 * API URL resolved from NEXT_PUBLIC_API_URL env var with a hardcoded
 * fallback to the Railway production deployment so the demo works even
 * if Vercel env-var propagation is delayed.
 *
 * The master key is used as the default so the dashboard sees all
 * tenants' telemetry (the agents post under the master key too).
 */

const RAILWAY_API_URL = "https://proactive-creativity-production.up.railway.app";
const FALLBACK_API_KEY = "***REDACTED_API_KEY***";

const getApiUrl = (): string => {
  const url = process.env.NEXT_PUBLIC_API_URL;
  // Never fall back to localhost. If NEXT_PUBLIC_API_URL isn't set yet,
  // fall back to the known Railway deployment.
  const resolved = url || RAILWAY_API_URL;
  return resolved.replace(/\/$/, "");
};

export const apiUrl = (): string => getApiUrl();

const authHeaders = (): Record<string, string> => {
  const key = process.env.NEXT_PUBLIC_COOLEDAI_API_KEY || FALLBACK_API_KEY;
  return { "X-API-Key": key };
};

/**
 * Merge caller-supplied headers with the auth header.
 * Caller headers win on conflict (e.g. Content-Type).
 */
const mergeHeaders = (
  extra?: HeadersInit
): Record<string, string> => {
  const base = authHeaders();
  if (!extra) return base;
  // HeadersInit can be Record, Headers, or string[][]
  if (extra instanceof Headers) {
    extra.forEach((v, k) => {
      base[k] = v;
    });
  } else if (Array.isArray(extra)) {
    extra.forEach(([k, v]) => {
      base[k] = v;
    });
  } else {
    Object.assign(base, extra);
  }
  return base;
};

/**
 * Fetch wrapper that auto-injects the API base URL and X-API-Key header.
 */
export const apiFetch = async (
  path: string,
  options?: RequestInit
): Promise<Response> => {
  const base = getApiUrl();
  const url = path.startsWith("/") ? `${base}${path}` : `${base}/${path}`;
  const hdrs = mergeHeaders(options?.headers);
  console.log(`[CooledAI] ${options?.method ?? "GET"} ${url}  key=${hdrs["X-API-Key"]?.slice(0, 12)}…`);
  const res = await fetch(url, { ...options, headers: hdrs, cache: "no-store" });
  return res;
};

// Convenience methods for CooledAI endpoints
export const api = {
  // --- Unauthenticated (no API key needed) ---
  health: () => apiFetch("/health"),
  getSimulatedMetrics: () => apiFetch("/simulated-metrics"),

  // --- Read (API key sent but not required by server for GET) ---
  getOptimize: () => apiFetch("/optimize"),
  getState: () => apiFetch("/state"),
  getStats: (token?: string | null) =>
    apiFetch("/api/v1/stats", {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    }),
  getThermalHistory: (hours: number, token?: string | null) =>
    apiFetch(`/api/v1/thermal-history?hours=${hours}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    }),
  getThermalHistoryRaw: (hours: number, token?: string | null) =>
    apiFetch(`/api/v1/thermal-history?hours=${hours}&mode=raw`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    }),
  getNodesList: (token?: string | null) =>
    apiFetch("/api/v1/nodes/list", {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    }),
  getHourlyFacilityMetrics: (token?: string | null) =>
    apiFetch("/api/v1/facility/hourly-metrics", {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    }),
  getThermalExport: (
    params: { start_date?: string; end_date?: string; days?: number },
    token?: string | null
  ) => {
    const sp = new URLSearchParams();
    if (params.start_date) sp.set("start_date", params.start_date);
    if (params.end_date) sp.set("end_date", params.end_date);
    if (params.days != null) sp.set("days", String(params.days));
    return apiFetch(`/api/v1/thermal-history/export?${sp}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
  },

  // --- Write (API key required) ---
  postOptimize: (body: unknown) =>
    apiFetch("/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  postIngestJson: (body: unknown) =>
    apiFetch("/ingest/json", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  setState: (state: string) =>
    apiFetch("/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state }),
    }),
  apply: () =>
    apiFetch("/apply", { method: "POST" }),

  // --- Admin (admin key required) ---
  triggerSimulation: (mode: string) =>
    apiFetch("/admin/simulation-control/trigger", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    }),
  simulationStatus: () =>
    apiFetch("/admin/simulation-control/status"),
  setShadowMode: (enabled: boolean) =>
    apiFetch("/shadow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }),
};
