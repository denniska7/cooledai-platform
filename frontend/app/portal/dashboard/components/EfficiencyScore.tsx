"use client";

interface EfficiencyScoreProps {
  score?: number;
  reclaimedPowerKw?: number;
}

export function EfficiencyScore({
  score = 94,
  reclaimedPowerKw = 142,
}: EfficiencyScoreProps) {
  return (
    <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
      <p className="text-xs text-white/50 uppercase tracking-wider mb-2">
        Efficiency Score
      </p>
      <p className="text-4xl font-medium tracking-tight text-white tabular-nums">
        {Math.round(score ?? 0)}%
      </p>
      <p className="mt-1 text-xs text-white/60">
        How close to Perfect Efficiency
      </p>
      <p className="mt-4 text-sm font-medium text-[#00FFCC] tabular-nums">
        Reclaimed Power: {reclaimedPowerKw.toLocaleString()} kW
      </p>
      <p className="mt-0.5 text-xs text-white/50">
        Power you can now use for more servers
      </p>
    </div>
  );
}
