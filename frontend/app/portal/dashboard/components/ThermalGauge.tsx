"use client";

interface ThermalGaugeProps {
  value: number; // 0-100, health score
  size?: number;
}

export function ThermalGauge({ value, size = 200 }: ThermalGaugeProps) {
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  const getColor = () => {
    if (value >= 80) return "#00FFCC";
    if (value >= 60) return "#FFFFFF";
    if (value >= 40) return "#fbbf24";
    return "#ef4444";
  };

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <span className="mt-4 text-4xl font-medium tracking-tight text-white">
        {Math.round(value)}%
      </span>
      <span className="text-sm text-white/50 mt-1">Global Thermal Health</span>
    </div>
  );
}
