"use client";

import { useCallback } from "react";

const variantId = process.env.NEXT_PUBLIC_VARIANT_ID;
const customCheckoutDomain = process.env.NEXT_PUBLIC_LEMON_SQUEEZY_CHECKOUT_DOMAIN ?? "billing.cooledai.com";

function getCheckoutUrl(): string | null {
  if (!variantId) return null;
  return `https://${customCheckoutDomain}/checkout/buy/${variantId}`;
}

type Props = {
  className?: string;
  children?: React.ReactNode;
};

/**
 * Opens the Lemon Squeezy checkout overlay for the configured variant.
 * Uses custom domain (billing.cooledai.com). Requires Lemon.js script and NEXT_PUBLIC_VARIANT_ID.
 */
export function SubscribeButton({ className, children }: Props) {
  const handleClick = useCallback(() => {
    const url = getCheckoutUrl();
    if (!url) {
      console.warn("SubscribeButton: Missing NEXT_PUBLIC_VARIANT_ID");
      return;
    }
    if (typeof window !== "undefined" && window.LemonSqueezy?.Url) {
      window.LemonSqueezy.Url.Open(url);
    } else {
      console.warn("SubscribeButton: Lemon.js not loaded yet. Open checkout manually:", url);
    }
  }, []);

  const url = getCheckoutUrl();
  const disabled = !url;

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled}
      className={
        className ??
        `relative rounded-lg border-2 border-[#22c55e] bg-[#0a0a0a] px-6 py-3.5 text-sm font-semibold tracking-tight text-[#22c55e] shadow-[0_0_20px_rgba(34,197,94,0.25)] transition-all duration-200 hover:border-[#22c55e] hover:bg-[#22c55e]/10 hover:shadow-[0_0_28px_rgba(34,197,94,0.35)] hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-[#22c55e]/50 focus:ring-offset-2 focus:ring-offset-[#0a0a0a] disabled:pointer-events-none disabled:opacity-50`
      }
    >
      {children ?? "Activate Fleet Optimization"}
    </button>
  );
}
