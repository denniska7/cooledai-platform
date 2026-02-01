"use client";

interface SafetyShieldToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export function SafetyShieldToggle({ enabled, onToggle }: SafetyShieldToggleProps) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <p className="text-sm font-medium text-white">Autonomous Safety Shield</p>
        <p className="text-xs text-white/50 mt-0.5">
          {enabled ? "Autonomous Mode — AI protecting hardware" : "Manual Mode — Human control"}
        </p>
      </div>
      <button
        role="switch"
        aria-checked={enabled}
        onClick={() => onToggle(!enabled)}
        className={`relative w-14 h-8 rounded-full border-2 transition-colors ${
          enabled
            ? "bg-[#00FFCC]/20 border-[#00FFCC]"
            : "bg-black border-white"
        }`}
      >
        <span
          className={`absolute top-1 w-6 h-6 rounded-full transition-transform ${
            enabled ? "left-7 bg-[#00FFCC]" : "left-1 bg-white"
          }`}
        />
      </button>
    </div>
  );
}
