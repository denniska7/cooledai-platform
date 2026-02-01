"use client";

import { useEffect, useRef } from "react";

/**
 * Polls a callback at a fixed interval (e.g., every 2 seconds for telemetry).
 * Use for Railway backend /simulated-metrics polling.
 */
export function useInterval(
  callback: () => void | Promise<void>,
  delayMs: number
) {
  const savedCallback = useRef(callback);
  const idRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delayMs <= 0) return;

    const tick = () => {
      const result = savedCallback.current();
      if (result instanceof Promise) {
        result.catch(() => {});
      }
    };

    tick();
    idRef.current = setInterval(tick, delayMs);
    return () => {
      if (idRef.current) {
        clearInterval(idRef.current);
        idRef.current = null;
      }
    };
  }, [delayMs]);
}
