"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { api } from "@/lib/api";
import type { DashboardSummary } from "@/lib/types/dashboard";

const HAS_CLERK = !!(
  process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY &&
  process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY !== "pk_test_cooledai_bypass"
);

const POLL_MS = 5_000;

export function useDashboardData(getToken?: () => Promise<string | null>) {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    try {
      let token: string | null = null;
      if (HAS_CLERK && getToken) {
        token = await getToken();
        if (!token) {
          setError("Sign in required.");
          setLoading(false);
          return;
        }
      }

      const res = await api.getDashboardSummary(token);
      if (!mountedRef.current) return;

      if (!res.ok) {
        if (res.status === 401) {
          setError("Sign in required.");
        } else {
          setError(`Server error (${res.status})`);
        }
        setLoading(false);
        return;
      }

      const json: DashboardSummary = await res.json();
      if (!mountedRef.current) return;

      setData(json);
      setError(null);
      setLoading(false);
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : "Failed to fetch dashboard data");
      setLoading(false);
    }
  }, [getToken]);

  useEffect(() => {
    mountedRef.current = true;
    fetchData();
    const id = setInterval(fetchData, POLL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [fetchData]);

  return { data, loading, error };
}
