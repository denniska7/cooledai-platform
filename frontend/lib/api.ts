/**
 * CooledAI API client
 * All API calls use process.env.NEXT_PUBLIC_API_URL
 */

const getApiUrl = (): string => {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) {
    throw new Error("NEXT_PUBLIC_API_URL is not set");
  }
  return url.replace(/\/$/, ""); // strip trailing slash
};

export const apiUrl = (): string => getApiUrl();

export const apiFetch = async (
  path: string,
  options?: RequestInit
): Promise<Response> => {
  const base = getApiUrl();
  const url = path.startsWith("/") ? `${base}${path}` : `${base}/${path}`;
  return fetch(url, options);
};

// Convenience methods for CooledAI endpoints
export const api = {
  health: () => apiFetch("/health"),
  getSimulatedMetrics: () => apiFetch("/simulated-metrics"),
  getOptimize: () => apiFetch("/optimize"),
  getState: () => apiFetch("/state"),
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
};
