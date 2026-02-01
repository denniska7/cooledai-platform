"use client";

export type TelemetryData = {
  cpu_temp_avg?: number;
  delta_t_inlet?: number;
  delta_t_outlet?: number;
  power_draw_kw?: number;
  nodes_active?: number;
  state?: string;
} | null;

interface LiveTelemetryProps {
  data: TelemetryData;
}

export function LiveTelemetry({ data }: LiveTelemetryProps) {
  const cpuTemp = data?.cpu_temp_avg ?? 42;
  const deltaTInlet = data?.delta_t_inlet ?? 18;
  const deltaTOutlet = data?.delta_t_outlet ?? 24;
  const powerDraw = data?.power_draw_kw ?? 120;

  const items = [
    { label: "CPU Temp (Avg)", value: `${cpuTemp}°C`, unit: "" },
    { label: "Delta-T Inlet", value: `${deltaTInlet}°C`, unit: "" },
    { label: "Delta-T Outlet", value: `${deltaTOutlet}°C`, unit: "" },
    { label: "Power Draw", value: `${powerDraw}`, unit: " kW" },
    { label: "Nodes Active", value: data?.nodes_active ?? "—", unit: "" },
    { label: "State", value: data?.state ?? "—", unit: "" },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-4"
        >
          <p className="text-xs text-white/50 uppercase tracking-wider">
            {item.label}
          </p>
          <p className="mt-1 text-xl font-medium tracking-tight text-white tabular-nums">
            {item.value}
            {item.unit}
          </p>
        </div>
      ))}
    </div>
  );
}
