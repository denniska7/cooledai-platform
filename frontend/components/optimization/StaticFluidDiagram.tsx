"use client";

export function StaticFluidDiagram() {
  const traditionalPath = "M 30 55 L 55 52 L 80 68 L 105 38 L 130 72 L 155 42 L 180 65 L 205 48 L 230 58 L 255 45 L 280 62 L 305 50 L 330 55 L 355 52";
  const cooledaiPath = "M 30 52 Q 90 50 150 52 T 270 52 T 355 51";

  return (
    <div className="grid grid-cols-2 gap-8 rounded border border-white/20 bg-black p-8 md:gap-12">
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white/50">Traditional Cooling</span>
        </div>
        <div className="rounded border border-white/10 bg-black p-6">
          <svg viewBox="0 0 380 130" className="w-full min-h-[140px]" preserveAspectRatio="xMidYMid meet">
            <rect x="35" y="52" width="330" height="16" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.12)" strokeWidth="0.5" strokeDasharray="4" />
            <text x="200" y="48" fill="rgba(255,255,255,0.35)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Setpoint 35–40°C</text>
            <text x="200" y="118" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Time</text>
            <text x="22" y="60" fill="rgba(255,255,255,0.4)" fontSize="7" fontFamily="system-ui" transform="rotate(-90 22 60)">Temp (°C)</text>
            <path
              d={traditionalPath}
              fill="none"
              stroke="rgba(255,255,255,0.5)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              pathLength={1}
              className="animate-draw-10s"
            />
            <text x="130" y="28" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui">Spike</text>
            <text x="250" y="28" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui">Spike</text>
          </svg>
        </div>
        <p className="text-xs text-white/50">
          Jagged, reactive. If temp &gt; 40°C, turn on fans.
        </p>
      </div>

      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white">CooledAI</span>
        </div>
        <div className="rounded border border-white/10 bg-black p-6">
          <svg viewBox="0 0 380 130" className="w-full min-h-[140px]" preserveAspectRatio="xMidYMid meet">
            <rect x="35" y="48" width="330" height="12" fill="rgba(0,255,204,0.06)" stroke="rgba(0,255,204,0.2)" strokeWidth="0.5" strokeDasharray="4" />
            <text x="200" y="44" fill="rgba(0,255,204,0.5)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Target 38°C ±1</text>
            <text x="200" y="118" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Time</text>
            <text x="22" y="60" fill="rgba(255,255,255,0.4)" fontSize="7" fontFamily="system-ui" transform="rotate(-90 22 60)">Temp (°C)</text>
            <path
              d={cooledaiPath}
              fill="none"
              stroke="#00FFCC"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              pathLength={1}
              className="animate-draw-10s"
            />
            <text x="200" y="28" fill="rgba(0,255,204,0.8)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Predictive · No bounce</text>
          </svg>
        </div>
        <p className="text-xs text-white/70">
          Smooth, predictive. Anticipate before the spike.
        </p>
      </div>
    </div>
  );
}
