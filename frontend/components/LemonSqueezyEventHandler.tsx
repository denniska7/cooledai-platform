"use client";

import { useEffect } from "react";

const successRedirectUrl =
  process.env.NEXT_PUBLIC_CHECKOUT_SUCCESS_REDIRECT_URL ?? "https://www.cooledai.com/portal?success=true";

/**
 * Sets up Lemon.js Checkout.Success handler to redirect to the main site portal.
 * Uses absolute URL (www.cooledai.com) so redirect works when checkout runs on billing.cooledai.com.
 * Mount once in root layout so the overlay event handler is registered on every page.
 */
export function LemonSqueezyEventHandler() {
  useEffect(() => {
    if (typeof window === "undefined") return;

    const setup = () => {
      if (window.LemonSqueezy?.Setup) {
        window.LemonSqueezy.Setup({
          eventHandler: (event) => {
            if (event.event === "Checkout.Success") {
              window.location.href = successRedirectUrl;
            }
          },
        });
        return true;
      }
      return false;
    };

    if (setup()) return;
    const id = setInterval(() => {
      if (setup()) clearInterval(id);
    }, 100);
    return () => clearInterval(id);
  }, []);

  return null;
}
