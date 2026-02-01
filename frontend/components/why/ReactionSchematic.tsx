"use client";

export function ReactionSchematic() {
  return (
    <div className="grid grid-cols-2 gap-8 rounded border border-white/20 bg-black p-8 md:gap-12">
      {/* Legacy - Left */}
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white/50">Legacy</span>
        </div>
        <div className="flex flex-col items-center gap-6">
          {/* Chip turns red */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 48 48" className="h-16 w-16">
              <rect x="8" y="8" width="32" height="32" fill="none" stroke="#ef4444" strokeWidth="1.5" />
              <rect x="12" y="12" width="8" height="8" fill="#ef4444" opacity="0.5" />
              <rect x="28" y="12" width="8" height="8" fill="#ef4444" opacity="0.5" />
              <rect x="12" y="28" width="8" height="8" fill="#ef4444" opacity="0.5" />
              <rect x="28" y="28" width="8" height="8" fill="#ef4444" opacity="0.5" />
            </svg>
            <span className="text-xs text-white/50">Temp spike</span>
          </div>
          <span className="text-2xl text-white/30">→</span>
          {/* Fan slowly on */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 48 48" className="h-16 w-16">
              <circle cx="24" cy="24" r="20" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
              <path d="M24 8 L24 40 M8 24 L40 24 M14 14 L34 34 M34 14 L14 34" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
            </svg>
            <span className="text-xs text-white/50">Fan response (delayed)</span>
          </div>
        </div>
      </div>

      {/* CooledAI - Right */}
      <div className="space-y-6">
        <div className="border-b border-white/20 pb-2">
          <span className="text-sm font-medium text-white">CooledAI</span>
        </div>
        <div className="flex flex-col items-center gap-6">
          {/* Neural net predicts */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 48 48" className="h-16 w-16">
              <circle cx="24" cy="8" r="4" fill="none" stroke="#FFFFFF" strokeWidth="1" />
              <circle cx="12" cy="24" r="3" fill="none" stroke="#FFFFFF" strokeWidth="1" />
              <circle cx="36" cy="24" r="3" fill="none" stroke="#FFFFFF" strokeWidth="1" />
              <circle cx="24" cy="40" r="4" fill="none" stroke="#FFFFFF" strokeWidth="1" />
              <line x1="24" y1="12" x2="14" y2="22" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.6" />
              <line x1="24" y1="12" x2="34" y2="22" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.6" />
              <line x1="12" y1="27" x2="22" y2="36" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.6" />
              <line x1="36" y1="27" x2="26" y2="36" stroke="#FFFFFF" strokeWidth="0.5" opacity="0.6" />
            </svg>
            <span className="text-xs text-white/70">Predict spike</span>
          </div>
          <span className="text-2xl text-white/50">→</span>
          {/* Pre-cool chip (stays cool) */}
          <div className="flex flex-col items-center gap-2">
            <svg viewBox="0 0 48 48" className="h-16 w-16">
              <rect x="8" y="8" width="32" height="32" fill="none" stroke="#FFFFFF" strokeWidth="1.5" />
              <rect x="12" y="12" width="8" height="8" fill="rgba(255,255,255,0.2)" />
              <rect x="28" y="12" width="8" height="8" fill="rgba(255,255,255,0.2)" />
              <rect x="12" y="28" width="8" height="8" fill="rgba(255,255,255,0.2)" />
              <rect x="28" y="28" width="8" height="8" fill="rgba(255,255,255,0.2)" />
            </svg>
            <span className="text-xs text-white/70">Pre-cooled</span>
          </div>
        </div>
      </div>
    </div>
  );
}
