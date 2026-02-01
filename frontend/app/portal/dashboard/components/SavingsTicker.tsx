"use client";

interface SavingsTickerProps {
  carbonOffsetKg: number;
  opexReclaimedUsd: number;
}

export function SavingsTicker({ carbonOffsetKg, opexReclaimedUsd }: SavingsTickerProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
        <p className="text-xs text-white/50 uppercase tracking-wider">
          Total Carbon Offset
        </p>
        <p className="mt-2 text-2xl font-medium tracking-tight text-white tabular-nums">
          {carbonOffsetKg.toLocaleString()} kg CO₂e
        </p>
      </div>
      <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
        <p className="text-xs text-white/50 uppercase tracking-wider">
          Est. OpEx Reclaimed
        </p>
        <p className="mt-2 text-2xl font-medium tracking-tight text-[#00FFCC] tabular-nums">
          ${opexReclaimedUsd.toLocaleString()} USD
        </p>
      </div>
    </div>
  );
}
