"use client";

import type { EfficiencyTrend } from "@/lib/types/dashboard";

export function TrendArrow({
  direction,
  className = "",
}: {
  direction: EfficiencyTrend;
  className?: string;
}) {
  if (direction === "up") {
    return (
      <svg
        width={16}
        height={16}
        viewBox="0 0 16 16"
        fill="none"
        className={`inline-block text-[#22c55e] ${className}`}
        aria-label="Trending up"
      >
        <path
          d="M8 3v10M8 3l-3.5 3.5M8 3l3.5 3.5"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  if (direction === "down") {
    return (
      <svg
        width={16}
        height={16}
        viewBox="0 0 16 16"
        fill="none"
        className={`inline-block text-red-400 ${className}`}
        aria-label="Trending down"
      >
        <path
          d="M8 13V3M8 13l-3.5-3.5M8 13l3.5-3.5"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  // flat
  return (
    <svg
      width={16}
      height={16}
      viewBox="0 0 16 16"
      fill="none"
      className={`inline-block text-white/50 ${className}`}
      aria-label="Flat trend"
    >
      <path
        d="M3 8h10"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
      />
    </svg>
  );
}
