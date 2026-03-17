/**
 * CooledAI API client
 *
 * API URL resolved from NEXT_PUBLIC_API_URL env var with a hardcoded
 * fallback to the Railway production deployment so the demo works even
 * if Vercel env-var propagation is delayed.
 */

const RAILWAY_API_URL = "https://proactive-creativity-production.up.railway.app";
const FALLBACK_API_KEY = "***REDACTED_API_KEY***";

const getApiUrl = (): string => {
  const url = process.env.NEXT_PUBLIC_API_URL;
  return (url || RAILWAY_API_URL).replace(/\/$/, "");
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
  return fetch(url, {
    ...options,
    headers: mergeHeaders(options?.headers),
  });
};

// Convenience methods for CooledAI endpoints
export const api = {
  // --- Unauthenticated (no API key needed) ---
  health: () => apiFetch("/health"),
  getSimulatedMetrics: () => apiFetch("/simulated-metrics"),

  // --- Read (API key sent but not required by server for GET) ---
  getOptimize: () => apiFetch("/optimize"),
  getState: () => apiFetch("/state"),
  getStats: () => apiFetch("/api/v1/stats"),

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
