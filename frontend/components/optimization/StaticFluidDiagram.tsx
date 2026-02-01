"use client";

export function StaticFluidDiagram() {
  // Left: Competitor A - jagged reactive line (threshold-based)
  // Right: CooledAI - smooth predictive curve (inference-based)
  const competitorPath = "M 20 40 L 50 38 L 80 65 L 110 35 L 140 70 L 170 40 L 200 55";
  const cooledaiPath = "M 20 50 Q 60 45 100 48 T 180 50";

  return (
    <div className="grid grid-cols-2 gap-8 rounded border border-white/20 bg-black p-8 md:gap-12">
      {/* Competitor A - Left */}
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white/50">Competitor A</span>
        </div>
        <div className="rounded border border-white/10 bg-black p-6">
          <svg viewBox="0 0 220 90" className="w-full" preserveAspectRatio="xMidYMid meet">
            <line x1="10" y1="45" x2="210" y2="45" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" strokeDasharray="4" />
            <path
              d={competitorPath}
              fill="none"
              stroke="rgba(255,255,255,0.5)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <text x="160" y="25" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui">
              Threshold-based
            </text>
          </svg>
        </div>
        <p className="text-xs text-white/50">
          Jagged, reactive. If temp &gt; 40°C, turn on fans.
        </p>
      </div>

      {/* CooledAI - Right */}
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white">CooledAI</span>
        </div>
        <div className="rounded border border-white/10 bg-black p-6">
          <svg viewBox="0 0 220 90" className="w-full" preserveAspectRatio="xMidYMid meet">
            <line x1="10" y1="45" x2="210" y2="45" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" strokeDasharray="4" />
            <path
              d={cooledaiPath}
              fill="none"
              stroke="#FFFFFF"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <text x="140" y="65" fill="rgba(255,255,255,0.8)" fontSize="8" fontFamily="system-ui">
              Inference-based
            </text>
          </svg>
        </div>
        <p className="text-xs text-white/70">
          Smooth, predictive. Anticipate before the spike.
        </p>
      </div>
    </div>
  );
}
