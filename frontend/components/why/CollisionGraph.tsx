"use client";

export function CollisionGraph() {
  // SVG: y increases downward. Top = low value, Bottom = high value.
  // AI Thermal Density: steep upward curve (starts low, ends high) = starts at top, ends at bottom
  // Legacy Cooling Capacity: flat (barely increases) = stays near top
  const aiPath = "M 20 75 Q 70 60 120 45 T 200 15";
  const legacyPath = "M 20 72 L 200 68";

  return (
    <div className="rounded border border-white/20 bg-black p-8">
      <svg
        viewBox="0 0 220 100"
        className="w-full max-w-2xl"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid */}
        {[25, 50, 75].map((y) => (
          <line
            key={y}
            x1="15"
            y1={y}
            x2="205"
            y2={y}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="0.5"
          />
        ))}
        {[55, 110, 165].map((x) => (
          <line
            key={x}
            x1={x}
            y1="10"
            x2={x}
            y2="90"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="0.5"
          />
        ))}
        {/* Legacy Cooling Capacity - flat gray */}
        <path
          d={legacyPath}
          fill="none"
          stroke="rgba(255,255,255,0.35)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        {/* AI Thermal Density - steep white */}
        <path
          d={aiPath}
          fill="none"
          stroke="#FFFFFF"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Labels */}
        <text x="140" y="72" fill="rgba(255,255,255,0.35)" fontSize="9" fontFamily="system-ui">
          Legacy Cooling Capacity
        </text>
        <text x="140" y="22" fill="#FFFFFF" fontSize="9" fontFamily="system-ui">
          AI Thermal Density
        </text>
      </svg>
    </div>
  );
}
