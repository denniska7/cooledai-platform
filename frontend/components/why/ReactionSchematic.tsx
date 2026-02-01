"use client";

export function ReactionSchematic() {
  return (
    <div className="grid grid-cols-2 gap-8 rounded border border-white/20 bg-black p-8 md:gap-12">
      {/* Traditional - Left */}
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white/50">Traditional</span>
          <span className="ml-2 text-xs text-white/40">Sense → React</span>
        </div>
        <div className="flex flex-col items-center gap-6">
          {/* Chip turns red */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 56 56" className="h-20 w-20">
              <rect x="10" y="10" width="36" height="36" fill="none" stroke="#ef4444" strokeWidth="1.5" rx="2" />
              <rect x="14" y="14" width="10" height="10" fill="#ef4444" opacity="0.6" rx="1" />
              <rect x="32" y="14" width="10" height="10" fill="#ef4444" opacity="0.6" rx="1" />
              <rect x="14" y="32" width="10" height="10" fill="#ef4444" opacity="0.6" rx="1" />
              <rect x="32" y="32" width="10" height="10" fill="#ef4444" opacity="0.6" rx="1" />
            </svg>
            <span className="text-xs font-medium text-white/60">1. Temp spike</span>
            <span className="text-[10px] text-white/40">40°C → 85°C in ms</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl text-white/30">→</span>
            <span className="text-[10px] text-white/35">2–5s latency</span>
          </div>
          {/* Fan slowly on */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 56 56" className="h-20 w-20">
              <circle cx="28" cy="28" r="22" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="1" />
              <path d="M28 10 L28 46 M10 28 L46 28 M16 16 L40 40 M40 16 L16 40" stroke="rgba(255,255,255,0.25)" strokeWidth="1" />
              <text x="28" y="30" fill="rgba(255,255,255,0.2)" fontSize="8" fontFamily="system-ui" textAnchor="middle">CRAC</text>
            </svg>
            <span className="text-xs font-medium text-white/50">2. Fan response</span>
            <span className="text-[10px] text-white/40">Delayed · Chip slows</span>
          </div>
        </div>
        <div className="rounded border border-white/10 bg-white/5 px-3 py-2">
          <p className="text-[10px] text-white/50 leading-relaxed">
            Sensor must detect heat before cooling ramps. GPU throttles to protect itself.
          </p>
        </div>
      </div>

      {/* CooledAI - Right */}
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white">CooledAI</span>
          <span className="ml-2 text-xs text-white/60">Predict → Pre-cool</span>
        </div>
        <div className="flex flex-col items-center gap-6">
          {/* Neural net predicts */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 56 56" className="h-20 w-20">
              <circle cx="28" cy="10" r="5" fill="none" stroke="#00FFCC" strokeWidth="1" />
              <circle cx="14" cy="28" r="4" fill="none" stroke="#00FFCC" strokeWidth="0.8" />
              <circle cx="42" cy="28" r="4" fill="none" stroke="#00FFCC" strokeWidth="0.8" />
              <circle cx="28" cy="46" r="5" fill="none" stroke="#00FFCC" strokeWidth="1" />
              <line x1="28" y1="15" x2="16" y2="25" stroke="#00FFCC" strokeWidth="0.5" opacity="0.5" />
              <line x1="28" y1="15" x2="40" y2="25" stroke="#00FFCC" strokeWidth="0.5" opacity="0.5" />
              <line x1="14" y1="31" x2="24" y2="40" stroke="#00FFCC" strokeWidth="0.5" opacity="0.5" />
              <line x1="42" y1="31" x2="32" y2="40" stroke="#00FFCC" strokeWidth="0.5" opacity="0.5" />
            </svg>
            <span className="text-xs font-medium text-white/80">1. Predict spike</span>
            <span className="text-[10px] text-white/50">Workload · Power · Room temp</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl text-white/50">→</span>
            <span className="text-[10px] text-white/50">&lt;1ms response</span>
          </div>
          {/* Pre-cool chip (stays cool) */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 56 56" className="h-20 w-20">
              <rect x="10" y="10" width="36" height="36" fill="none" stroke="#00FFCC" strokeWidth="1.5" rx="2" strokeOpacity="0.9" />
              <rect x="14" y="14" width="10" height="10" fill="rgba(0,255,204,0.2)" rx="1" />
              <rect x="32" y="14" width="10" height="10" fill="rgba(0,255,255,0.2)" rx="1" />
              <rect x="14" y="32" width="10" height="10" fill="rgba(0,255,255,0.2)" rx="1" />
              <rect x="32" y="32" width="10" height="10" fill="rgba(0,255,204,0.2)" rx="1" />
            </svg>
            <span className="text-xs font-medium text-white/80">2. Pre-cooled</span>
            <span className="text-[10px] text-white/50">Flat envelope · No throttle</span>
          </div>
        </div>
        <div className="rounded border border-white/20 bg-white/5 px-3 py-2 border-cyan-500/30">
          <p className="text-[10px] text-white/60 leading-relaxed">
            Cooling ramps before heat arrives. Thermal envelope stays flat.
          </p>
        </div>
      </div>
    </div>
  );
}
