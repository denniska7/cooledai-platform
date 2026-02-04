"use client";

interface SafetyShieldToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export function SafetyShieldToggle({ enabled, onToggle }: SafetyShieldToggleProps) {
  return (
    <div className="flex items-center justify-between gap-6">
      <div className="min-w-0">
        <p className="text-sm font-medium text-white">Autonomous Safety Shield</p>
        <p className="text-xs text-white/50 mt-0.5">
          {enabled ? "Autonomous Mode — AI protecting hardware" : "Manual Mode — Human control"}
        </p>
      </div>
      <button
        role="switch"
        aria-checked={enabled}
        onClick={() => onToggle(!enabled)}
        className={`relative shrink-0 h-8 w-14 rounded-full border-2 transition-colors flex items-center ${
          enabled
            ? "bg-[#00FFCC]/20 border-[#00FFCC] justify-end pl-1 pr-1"
            : "bg-black border-white justify-start pl-1 pr-1"
        }`}
      >
        <span
          className={`block h-5 w-5 shrink-0 rounded-full transition-colors duration-200 ${
            enabled ? "bg-[#00FFCC]" : "bg-white"
          }`}
        />
      </button>
    </div>
  );
}
