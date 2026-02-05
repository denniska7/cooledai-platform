import type { Metadata, Viewport } from "next";
import Script from "next/script";
import { Inter } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { BetaSignupPopup } from "../components/BetaSignupPopup";
import { Footer } from "../components/Footer";
import { LemonSqueezyEventHandler } from "../components/LemonSqueezyEventHandler";

const clerkPubKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CooledAI | The Universal Autonomy Layer for Every Watt of Compute",
  description:
    "Predictive thermal optimization for high-density data centers. Air, liquid, and immersion cooling—unified. Meaningful cooling energy savings while scaling from 100 kW to 100+ MW.",
  icons: {
    icon: "/logo.png",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  viewportFit: "cover",
  themeColor: "#000000",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const content = (
    <>
      <LemonSqueezyEventHandler />
      <div className="flex-1">{children}</div>
      <Footer />
      <BetaSignupPopup />
      <Analytics />
    </>
  );

  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-black text-white flex flex-col">
        <Script
          src="https://app.lemonsqueezy.com/js/lemon.js"
          strategy="afterInteractive"
        />
        {clerkPubKey ? (
          <ClerkProvider publishableKey={clerkPubKey}>{content}</ClerkProvider>
        ) : (
          content
        )}
      </body>
    </html>
  );
}
