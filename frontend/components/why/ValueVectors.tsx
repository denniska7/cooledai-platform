"use client";

function RackIcon() {
  return (
    <svg viewBox="0 0 64 64" className="h-12 w-12" fill="none" stroke="currentColor" strokeWidth="1">
      <rect x="8" y="8" width="48" height="48" stroke="white" strokeOpacity="0.8" />
      {[16, 28, 40, 52].map((y) => (
        <line key={y} x1="12" y1={y} x2="52" y2={y} stroke="white" strokeOpacity="0.5" />
      ))}
      {[20, 32, 44].map((x) => (
        <rect key={x} x={x} y="12" width="8" height="8" stroke="white" strokeOpacity="0.6" fill="none" />
      ))}
    </svg>
  );
}

function ChipShieldIcon() {
  return (
    <svg viewBox="0 0 64 64" className="h-12 w-12" fill="none" stroke="currentColor" strokeWidth="1">
      <path d="M32 8 L52 16 L52 32 L32 48 L12 32 L12 16 Z" stroke="white" strokeOpacity="0.8" fill="none" />
      <rect x="24" y="24" width="16" height="16" stroke="white" strokeOpacity="0.6" fill="none" />
      <line x1="28" y1="24" x2="28" y2="40" stroke="white" strokeOpacity="0.4" />
      <line x1="32" y1="24" x2="32" y2="40" stroke="white" strokeOpacity="0.4" />
      <line x1="36" y1="24" x2="36" y2="40" stroke="white" strokeOpacity="0.4" />
    </svg>
  );
}

function OpExIcon() {
  return (
    <svg viewBox="0 0 64 64" className="h-12 w-12" fill="none" stroke="currentColor" strokeWidth="1">
      {/* Power plug prongs + base */}
      <rect x="26" y="16" width="4" height="24" stroke="white" strokeOpacity="0.8" fill="none" />
      <rect x="34" y="16" width="4" height="24" stroke="white" strokeOpacity="0.8" fill="none" />
      <rect x="20" y="38" width="24" height="10" rx="2" stroke="white" strokeOpacity="0.8" fill="none" />
      {/* Dollar S */}
      <path d="M32 22 L32 42 M30 24 Q32 22 34 24 Q36 28 32 30 Q28 32 30 36 Q32 38 34 36" stroke="white" strokeOpacity="0.8" fill="none" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const vectors = [
  {
    icon: RackIcon,
    title: "Pack More In",
    text: "Stop leaving rack space empty due to heat limits. Safely deploy higher-density clusters in existing space by cutting peak heat loads.",
  },
  {
    icon: ChipShieldIcon,
    title: "Hardware Lasts Longer",
    text: "Rapid heating and cooling damages chips. By smoothing out temperature swings, CooledAI reduces hardware failure rates over 3–5 years.",
  },
  {
    icon: OpExIcon,
    title: "Cut Cooling Costs",
    text: "Reduce total cooling energy spend by 8–12%. In multi-megawatt facilities, this translates to millions in savings that go straight to the bottom line.",
  },
];

export function ValueVectors() {
  return (
    <div className="grid gap-12 md:grid-cols-3">
      {vectors.map((v, i) => (
        <div key={i} className="rounded border border-white/20 p-8">
          <div className="mb-6 text-white">
            <v.icon />
          </div>
          <h3 className="mb-4 text-lg font-medium tracking-tight text-white">
            {v.title}
          </h3>
          <p className="text-sm leading-relaxed text-white/70">
            {v.text}
          </p>
        </div>
      ))}
    </div>
  );
}
