"use client";

export function StaticFluidDiagram() {
  const viewW = 400;
  const viewH = 200;
  const padL = 44;
  const padR = 24;
  const padT = 32;
  const padB = 36;
  const chartW = viewW - padL - padR;
  const chartH = viewH - padT - padB;
  const yMin = 30;
  const yMax = 50;
  const yToY = (v: number) => padT + chartH - ((v - yMin) / (yMax - yMin)) * chartH;
  const gridYs = [30, 35, 40, 45, 50];
  const gridXs = [0, 2, 4, 6, 8, 10];

  // Paths in chart coordinates: x from padL to padL+chartW, y = yToY(temp)
  const pts = (temps: number[]) => temps.map((t, i) => `${i === 0 ? "M" : "L"} ${padL + (i / (temps.length - 1)) * chartW} ${yToY(t)}`).join(" ");
  const traditionalPath = pts([38, 36, 48, 35, 46, 34, 45, 37, 42, 35, 40, 36]);
  const cooledaiPath = `M ${padL} ${yToY(37.5)} Q ${padL + chartW * 0.35} ${yToY(37.8)} ${padL + chartW * 0.5} ${yToY(38)} Q ${padL + chartW * 0.65} ${yToY(37.9)} ${padL + chartW} ${yToY(38)}`;

  return (
    <div className="grid grid-cols-2 gap-8 rounded border border-white/20 bg-black p-8 md:gap-12">
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white/50">Traditional Cooling</span>
        </div>
        <div className="rounded border border-white/10 bg-black p-6 min-h-[280px] flex items-center">
          <svg viewBox={`0 0 ${viewW} ${viewH}`} className="w-full min-h-[240px]" preserveAspectRatio="xMidYMid meet">
            {/* Y-axis grid + labels */}
            {gridYs.map((y) => (
              <g key={y}>
                <line x1={padL} y1={yToY(y)} x2={viewW - padR} y2={yToY(y)} stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" strokeDasharray="3" />
                <text x={padL - 6} y={yToY(y) + 3} fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="end">{y}°</text>
              </g>
            ))}
            {/* X-axis grid + labels */}
            {gridXs.map((x, i) => {
              const px = padL + (i / (gridXs.length - 1)) * chartW;
              return (
                <g key={x}>
                  <line x1={px} y1={padT} x2={px} y2={viewH - padB} stroke="rgba(255,255,255,0.06)" strokeWidth="0.5" strokeDasharray="2" />
                  <text x={px} y={viewH - 12} fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui" textAnchor="middle">{x}m</text>
                </g>
              );
            })}
            <rect x={padL} y={yToY(40)} width={chartW} height={yToY(35) - yToY(40)} fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.12)" strokeWidth="0.5" strokeDasharray="4" />
            <text x={padL + chartW / 2} y={yToY(40) - 8} fill="rgba(255,255,255,0.35)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Setpoint 35–40°C</text>
            <text x={padL + chartW / 2} y={viewH - 2} fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Time (minutes)</text>
            <text x="14" y={padT + chartH / 2} fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="middle" transform="rotate(-90 14 100)">Temp (°C)</text>
            <path
              d={traditionalPath}
              fill="none"
              stroke="rgba(255,255,255,0.55)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              pathLength={1}
              className="animate-draw-10s"
            />
            <text x={130} y={padT - 6} fill="rgba(255,255,255,0.5)" fontSize="9" fontFamily="system-ui">Spike</text>
            <text x={265} y={padT - 6} fill="rgba(255,255,255,0.5)" fontSize="9" fontFamily="system-ui">Spike</text>
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
        <div className="rounded border border-white/10 bg-black p-6 min-h-[280px] flex items-center">
          <svg viewBox={`0 0 ${viewW} ${viewH}`} className="w-full min-h-[240px]" preserveAspectRatio="xMidYMid meet">
            {gridYs.map((y) => (
              <g key={y}>
                <line x1={padL} y1={yToY(y)} x2={viewW - padR} y2={yToY(y)} stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" strokeDasharray="3" />
                <text x={padL - 6} y={yToY(y) + 3} fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="end">{y}°</text>
              </g>
            ))}
            {gridXs.map((x, i) => {
              const px = padL + (i / (gridXs.length - 1)) * chartW;
              return (
                <g key={x}>
                  <line x1={px} y1={padT} x2={px} y2={viewH - padB} stroke="rgba(255,255,255,0.06)" strokeWidth="0.5" strokeDasharray="2" />
                  <text x={px} y={viewH - 12} fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="system-ui" textAnchor="middle">{x}m</text>
                </g>
              );
            })}
            <rect x={padL} y={yToY(39)} width={chartW} height={yToY(37) - yToY(39)} fill="rgba(0,255,204,0.06)" stroke="rgba(0,255,204,0.2)" strokeWidth="0.5" strokeDasharray="4" />
            <text x={padL + chartW / 2} y={yToY(39) - 8} fill="rgba(0,255,204,0.5)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Target 38°C ±1</text>
            <text x={padL + chartW / 2} y={viewH - 2} fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Time (minutes)</text>
            <text x="14" y={padT + chartH / 2} fill="rgba(255,255,255,0.45)" fontSize="9" fontFamily="system-ui" textAnchor="middle" transform="rotate(-90 14 100)">Temp (°C)</text>
            <path
              d={cooledaiPath}
              fill="none"
              stroke="#00FFCC"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              pathLength={1}
              className="animate-draw-10s"
            />
            <text x={padL + chartW / 2} y={padT - 6} fill="rgba(0,255,204,0.85)" fontSize="9" fontFamily="system-ui" textAnchor="middle">Predictive · No bounce</text>
          </svg>
        </div>
        <p className="text-xs text-white/70">
          Smooth, predictive. Anticipate before the spike.
        </p>
      </div>
    </div>
  );
}
