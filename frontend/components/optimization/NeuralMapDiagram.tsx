"use client";

export function NeuralMapDiagram() {
  return (
    <div className="rounded border border-white/20 bg-black p-8">
      <svg viewBox="0 0 560 200" className="w-full max-w-4xl mx-auto min-h-[200px]" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="rgba(255,255,255,0.6)" />
          </marker>
        </defs>
        <rect x="20" y="70" width="95" height="60" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="1" />
        <text x="67" y="95" fill="#FFFFFF" fontSize="10" fontFamily="system-ui" textAnchor="middle">Rack Telemetry</text>
        <text x="67" y="110" fill="rgba(255,255,255,0.5)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Temp · Power · Load</text>
        <text x="67" y="122" fill="rgba(255,255,255,0.35)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Power · Room temp</text>

        <line x1="115" y1="100" x2="155" y2="100" stroke="rgba(255,255,255,0.5)" strokeWidth="1" markerEnd="url(#arrow)" pathLength={1} className="animate-draw-10s" />

        {/* Predictive Engine */}
        <rect x="155" y="55" width="100" height="90" fill="none" stroke="#00FFCC" strokeWidth="1.5" strokeOpacity="0.9" />
        <text x="205" y="85" fill="#FFFFFF" fontSize="11" fontFamily="system-ui" textAnchor="middle">Predictive Engine</text>
        <text x="205" y="100" fill="#00FFCC" fontSize="9" fontFamily="system-ui" textAnchor="middle">AI Model</text>
        <text x="205" y="115" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Heat prediction</text>
        <text x="205" y="128" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui" textAnchor="middle">model</text>

        {/* Training data inputs */}
        <line x1="180" y1="55" x2="180" y2="35" stroke="rgba(0,255,204,0.3)" strokeWidth="0.8" strokeDasharray="3" />
        <line x1="230" y1="55" x2="230" y2="35" stroke="rgba(0,255,204,0.3)" strokeWidth="0.8" strokeDasharray="3" />
        <rect x="140" y="8" width="130" height="22" fill="rgba(0,255,204,0.06)" stroke="rgba(0,255,204,0.25)" strokeWidth="0.5" strokeDasharray="4" />
        <text x="205" y="23" fill="rgba(0,255,204,0.7)" fontSize="8" fontFamily="system-ui" textAnchor="middle">500K+ failure hours · H100 · EPYC</text>

        {/* Workload Schedule as secondary input */}
        <rect x="270" y="8" width="100" height="28" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="0.8" />
        <text x="320" y="23" fill="rgba(255,255,255,0.7)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Workload Schedule</text>
        <text x="320" y="32" fill="rgba(255,255,255,0.45)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Job schedule · Pre-cool</text>
        <line x1="320" y1="36" x2="255" y2="55" stroke="rgba(255,255,255,0.25)" strokeWidth="0.5" strokeDasharray="2" markerEnd="url(#arrow)" />

        <line x1="255" y1="100" x2="365" y2="100" stroke="rgba(255,255,255,0.5)" strokeWidth="1" markerEnd="url(#arrow)" pathLength={1} className="animate-draw-10s" />

        {/* Output: Fan/Chiller Control */}
        <rect x="375" y="70" width="95" height="60" fill="none" stroke="#FFFFFF" strokeWidth="1" strokeOpacity="0.8" />
        <text x="422" y="95" fill="#FFFFFF" fontSize="10" fontFamily="system-ui" textAnchor="middle">Fan/Chiller</text>
        <text x="422" y="110" fill="rgba(255,255,255,0.6)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Control</text>
        <text x="422" y="122" fill="rgba(255,255,255,0.4)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Pre-cool · Ramp</text>

        {/* Latency badge */}
        <rect x="382" y="145" width="80" height="18" fill="rgba(0,255,204,0.08)" stroke="rgba(0,255,204,0.3)" strokeWidth="0.5" />
        <text x="422" y="157" fill="rgba(0,255,204,0.8)" fontSize="8" fontFamily="system-ui" textAnchor="middle">&lt;1ms on-site</text>
      </svg>
    </div>
  );
}
