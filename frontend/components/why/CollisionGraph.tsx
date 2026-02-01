"use client";

export function CollisionGraph() {
  // SVG: y increases downward. Top = low value, Bottom = high value.
  // AI Thermal Density: steep upward curve (Moore's Law) = starts at top, ends at bottom
  // Traditional Cooling Capacity: flat (infrastructure lags) = stays near top
  const aiPath = "M 45 95 Q 120 75 195 55 T 340 25 T 420 12";
  const traditionalPath = "M 45 88 L 420 82";

  return (
    <div className="rounded border border-white/20 bg-black p-8">
      <svg
        viewBox="0 0 460 140"
        className="w-full max-w-4xl min-h-[180px] mx-auto"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid - horizontal */}
        {[25, 50, 75, 100, 115].map((y) => (
          <line
            key={y}
            x1="40"
            y1={y}
            x2="435"
            y2={y}
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="0.5"
            strokeDasharray="4"
          />
        ))}
        {/* Grid - vertical */}
        {[100, 175, 250, 325, 400].map((x) => (
          <line
            key={x}
            x1={x}
            y1="15"
            x2={x}
            y2="120"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="0.5"
            strokeDasharray="4"
          />
        ))}
        {/* Y-axis */}
        <line x1="45" y1="15" x2="45" y2="120" stroke="rgba(255,255,255,0.2)" strokeWidth="0.5" />
        {/* X-axis */}
        <line x1="45" y1="120" x2="435" y2="120" stroke="rgba(255,255,255,0.2)" strokeWidth="0.5" />
        {/* Y-axis labels */}
        <text x="38" y="22" fill="rgba(255,255,255,0.45)" fontSize="8" fontFamily="system-ui" textAnchor="end">High</text>
        <text x="38" y="68" fill="rgba(255,255,255,0.45)" fontSize="8" fontFamily="system-ui" textAnchor="end">—</text>
        <text x="38" y="118" fill="rgba(255,255,255,0.45)" fontSize="8" fontFamily="system-ui" textAnchor="end">Low</text>
        {/* X-axis labels */}
        <text x="95" y="135" fill="rgba(255,255,255,0.4)" fontSize="7" fontFamily="system-ui" textAnchor="middle">2022</text>
        <text x="240" y="135" fill="rgba(255,255,255,0.4)" fontSize="7" fontFamily="system-ui" textAnchor="middle">2026</text>
        <text x="385" y="135" fill="rgba(255,255,255,0.4)" fontSize="7" fontFamily="system-ui" textAnchor="middle">2030</text>
        {/* Traditional Cooling Capacity - flat gray */}
        <path
          d={traditionalPath}
          fill="none"
          stroke="rgba(255,255,255,0.4)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="6 4"
        />
        {/* AI Thermal Density - steep white/cyan */}
        <path
          d={aiPath}
          fill="none"
          stroke="#00FFCC"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Collision point marker */}
        <circle cx="280" cy="55" r="5" fill="none" stroke="#ef4444" strokeWidth="1.5" strokeOpacity="0.9" />
        <text x="290" y="52" fill="rgba(239,68,68,0.9)" fontSize="8" fontFamily="system-ui" fontWeight="500">Thermal Wall</text>
        {/* Legend */}
        <rect x="300" y="18" width="130" height="36" fill="rgba(0,0,0,0.5)" stroke="rgba(255,255,255,0.15)" strokeWidth="0.5" rx="2" />
        <line x1="315" y1="30" x2="345" y2="30" stroke="#00FFCC" strokeWidth="2" strokeLinecap="round" />
        <text x="355" y="33" fill="rgba(255,255,255,0.9)" fontSize="8" fontFamily="system-ui">AI Thermal Density</text>
        <line x1="315" y1="44" x2="345" y2="44" stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" strokeDasharray="4 3" strokeLinecap="round" />
        <text x="355" y="47" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui">Traditional Cooling</text>
      </svg>
      <p className="mt-4 text-xs text-white/45 text-center">
        Silicon doubles every 18–24 months. Cooling infrastructure upgrades take years.
      </p>
    </div>
  );
}
