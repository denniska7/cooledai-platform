"use client";

export function NeuralMapDiagram() {
  return (
    <div className="rounded border border-white/20 bg-black p-8">
      <svg viewBox="0 0 400 180" className="w-full max-w-2xl mx-auto" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="rgba(255,255,255,0.6)" />
          </marker>
        </defs>
        <rect x="20" y="60" width="100" height="60" fill="none" stroke="#FFFFFF" strokeWidth="1" strokeOpacity="0.8" />
        <text x="70" y="88" fill="#FFFFFF" fontSize="10" fontFamily="system-ui" textAnchor="middle">
          Rack Telemetry
        </text>
        <text x="70" y="102" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui" textAnchor="middle">
          Temp · Power · Load
        </text>
        <line x1="120" y1="90" x2="160" y2="90" stroke="rgba(255,255,255,0.5)" strokeWidth="1" markerEnd="url(#arrow)" />
        <rect x="160" y="50" width="80" height="80" fill="none" stroke="#FFFFFF" strokeWidth="1.5" />
        <text x="200" y="85" fill="#FFFFFF" fontSize="10" fontFamily="system-ui" textAnchor="middle">
          Predictive
        </text>
        <text x="200" y="98" fill="#FFFFFF" fontSize="10" fontFamily="system-ui" textAnchor="middle">
          Engine
        </text>
        <text x="200" y="115" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui" textAnchor="middle">
          RL Model
        </text>
        <line x1="240" y1="90" x2="280" y2="90" stroke="rgba(255,255,255,0.5)" strokeWidth="1" markerEnd="url(#arrow)" />
        <rect x="280" y="60" width="100" height="60" fill="none" stroke="#FFFFFF" strokeWidth="1" strokeOpacity="0.8" />
        <text x="330" y="88" fill="#FFFFFF" fontSize="10" fontFamily="system-ui" textAnchor="middle">
          Fan/Chiller
        </text>
        <text x="330" y="102" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui" textAnchor="middle">
          Control
        </text>
        <circle cx="200" cy="30" r="4" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        <circle cx="180" cy="40" r="3" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        <circle cx="220" cy="40" r="3" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" />
        <line x1="180" y1="40" x2="200" y2="30" stroke="rgba(255,255,255,0.2)" strokeWidth="0.3" />
        <line x1="220" y1="40" x2="200" y2="30" stroke="rgba(255,255,255,0.2)" strokeWidth="0.3" />
      </svg>
    </div>
  );
}
