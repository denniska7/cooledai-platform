"use client";

export function NeuralMapDiagram() {
  const vbW = 640;
  const vbH = 300;

  return (
    <div className="rounded border border-white/20 bg-black p-8">
      <svg viewBox={`0 0 ${vbW} ${vbH}`} className="w-full max-w-5xl mx-auto min-h-[300px]" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow-nm" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto">
            <polygon points="0 0, 12 4, 0 8" fill="rgba(255,255,255,0.6)" />
          </marker>
          <marker id="arrow-cyan" markerWidth="10" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="rgba(0,255,204,0.6)" />
          </marker>
        </defs>

        {/* Subtle grid */}
        {[0.25, 0.5, 0.75].map((t) => (
          <line key={t} x1={t * vbW} y1={0} x2={t * vbW} y2={vbH} stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
        ))}
        {[0.33, 0.66].map((t) => (
          <line key={t} x1={0} y1={t * vbH} x2={vbW} y2={t * vbH} stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
        ))}

        {/* Step label */}
        <text x="52" y="118" fill="rgba(255,255,255,0.35)" fontSize="10" fontFamily="system-ui" textAnchor="middle">1</text>
        <rect x="28" y="125" width="110" height="88" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.6)" strokeWidth="1.2" rx="4" />
        <text x="83" y="158" fill="#FFFFFF" fontSize="13" fontFamily="system-ui" textAnchor="middle" fontWeight="500">Rack Telemetry</text>
        <text x="83" y="178" fill="rgba(255,255,255,0.55)" fontSize="10" fontFamily="system-ui" textAnchor="middle">Temp · Power · Load</text>
        <text x="83" y="195" fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Ambient · Room temp · Flow</text>

        <line x1="138" y1="169" x2="198" y2="169" stroke="rgba(255,255,255,0.5)" strokeWidth="1.2" markerEnd="url(#arrow-nm)" pathLength={1} className="animate-draw-10s" />

        {/* Predictive Engine */}
        <text x="328" y="118" fill="rgba(255,255,255,0.35)" fontSize="10" fontFamily="system-ui" textAnchor="middle">2</text>
        <rect x="198" y="125" width="260" height="88" fill="rgba(0,255,204,0.03)" stroke="#00FFCC" strokeWidth="1.5" strokeOpacity="0.9" rx="4" />
        <text x="328" y="158" fill="#FFFFFF" fontSize="14" fontFamily="system-ui" textAnchor="middle" fontWeight="500">Predictive Engine</text>
        <text x="328" y="178" fill="#00FFCC" fontSize="11" fontFamily="system-ui" textAnchor="middle">AI Model · Heat prediction</text>
        <text x="328" y="195" fill="rgba(255,255,255,0.55)" fontSize="10" fontFamily="system-ui" textAnchor="middle">Thermal runaway · Time-to-failure</text>

        {/* Training data input */}
        <line x1="268" y1="125" x2="268" y2="85" stroke="rgba(0,255,204,0.35)" strokeWidth="1" strokeDasharray="4" />
        <line x1="388" y1="125" x2="388" y2="85" stroke="rgba(0,255,204,0.35)" strokeWidth="1" strokeDasharray="4" />
        <rect x="218" y="28" width="220" height="52" fill="rgba(0,255,204,0.06)" stroke="rgba(0,255,204,0.28)" strokeWidth="0.8" strokeDasharray="4" rx="2" />
        <text x="328" y="50" fill="rgba(0,255,204,0.8)" fontSize="11" fontFamily="system-ui" textAnchor="middle">500K+ thermal failure hours</text>
        <text x="328" y="68" fill="rgba(0,255,204,0.55)" fontSize="9" fontFamily="system-ui" textAnchor="middle">NVIDIA H100 · AMD EPYC · Blackwell</text>

        {/* Workload Schedule */}
        <rect x="478" y="28" width="134" height="52" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.4)" strokeWidth="1" rx="2" />
        <text x="545" y="50" fill="rgba(255,255,255,0.75)" fontSize="11" fontFamily="system-ui" textAnchor="middle">Workload Schedule</text>
        <text x="545" y="68" fill="rgba(255,255,255,0.5)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Job schedule · Pre-cool triggers</text>
        <line x1="545" y1="80" x2="418" y2="125" stroke="rgba(255,255,255,0.3)" strokeWidth="0.8" strokeDasharray="3" markerEnd="url(#arrow-cyan)" />

        <line x1="458" y1="169" x2="538" y2="169" stroke="rgba(255,255,255,0.5)" strokeWidth="1.2" markerEnd="url(#arrow-nm)" pathLength={1} className="animate-draw-10s" />

        {/* Output: Fan/Chiller Control */}
        <text x="602" y="118" fill="rgba(255,255,255,0.35)" fontSize="10" fontFamily="system-ui" textAnchor="middle">3</text>
        <rect x="538" y="125" width="110" height="88" fill="rgba(255,255,255,0.02)" stroke="#FFFFFF" strokeWidth="1.2" strokeOpacity="0.85" rx="4" />
        <text x="593" y="158" fill="#FFFFFF" fontSize="13" fontFamily="system-ui" textAnchor="middle" fontWeight="500">Fan / Chiller</text>
        <text x="593" y="178" fill="rgba(255,255,255,0.6)" fontSize="10" fontFamily="system-ui" textAnchor="middle">Control</text>
        <text x="593" y="195" fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Pre-cool · Ramp · Throttle</text>

        {/* Latency badge */}
        <rect x="548" y="228" width="90" height="24" fill="rgba(0,255,204,0.08)" stroke="rgba(0,255,204,0.35)" strokeWidth="0.6" rx="3" />
        <text x="593" y="244" fill="rgba(0,255,204,0.9)" fontSize="10" fontFamily="system-ui" textAnchor="middle" fontWeight="500">&lt;1ms on-site</text>
      </svg>
    </div>
  );
}
