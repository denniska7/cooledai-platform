"use client";

interface SevenDayTestProgressProps {
  /** 1-7 */
  currentDay?: number;
  /** e.g., "Establishing Baseline", "Collecting Data", "Validating Model" */
  phase?: string;
}

const PHASES: Record<number, string> = {
  1: "Initializing",
  2: "Collecting Data",
  3: "Establishing Baseline",
  4: "Building Model",
  5: "Validating Predictions",
  6: "Final Calibration",
  7: "Complete",
};

export function SevenDayTestProgress({
  currentDay = 3,
  phase,
}: SevenDayTestProgressProps) {
  const day = Math.min(7, Math.max(1, currentDay ?? 3));
  const label = phase ?? PHASES[day] ?? "In Progress";
  const progress = (day / 7) * 100;

  return (
    <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
      <p className="text-xs text-white/50 uppercase tracking-wider mb-3">
        The 7-Day Efficiency Test
      </p>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-white">
          Day {day}/7: {label}
        </span>
        <span className="text-xs text-white/50">{Math.round(progress)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full bg-[#00FFCC] transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
