"use client";

interface PredictiveGaugeProps {
  predictedLoad: number; // T+10 min thermal load (%)
  currentCapacity: number; // Current capacity (%)
  size?: number;
}

export function PredictiveGauge({
  predictedLoad,
  currentCapacity,
  size = 220,
}: PredictiveGaugeProps) {
  const strokeWidth = 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const loadOffset = circumference - (predictedLoad / 100) * circumference;
  const capacityOffset = circumference - (currentCapacity / 100) * circumference;

  const headroom = currentCapacity - predictedLoad;
  const statusColor = headroom >= 15 ? "#00FFCC" : headroom >= 5 ? "#FFFFFF" : "#ef4444";

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
        />
        {/* Current Capacity (outer, white) */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.4)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={capacityOffset}
          className="transition-all duration-500"
        />
        {/* Predicted Load T+10 (inner, cyan) */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius - 2}
          fill="none"
          stroke={statusColor}
          strokeWidth={strokeWidth - 2}
          strokeLinecap="round"
          strokeDasharray={2 * Math.PI * (radius - 2)}
          strokeDashoffset={
            2 * Math.PI * (radius - 2) - (predictedLoad / 100) * 2 * Math.PI * (radius - 2)
          }
          className="transition-all duration-500"
        />
      </svg>
      <div className="mt-6 text-center space-y-1">
        <p className="text-2xl font-medium tracking-tight text-white">
          {Math.round(predictedLoad)}% / {Math.round(currentCapacity)}%
        </p>
        <p className="text-xs text-white/50">Predicted Load (T+10) / Capacity</p>
        <p className="text-sm mt-2" style={{ color: statusColor }}>
          {headroom >= 0 ? `${Math.round(headroom)}% headroom` : "Over capacity"}
        </p>
      </div>
    </div>
  );
}
