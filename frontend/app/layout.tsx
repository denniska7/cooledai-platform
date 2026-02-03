import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { BetaSignupPopup } from "../components/BetaSignupPopup";
import { Footer } from "../components/Footer";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CooledAI | The Universal Autonomy Layer for Every Watt of Compute",
  description:
    "Predictive thermal optimization for high-density data centers. Air, liquid, and immersion cooling—unified. Reduce cooling costs by 12% while scaling from 100 kW to 100+ MW.",
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
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-black text-white flex flex-col">
        <div className="flex-1">{children}</div>
        <Footer />
        <BetaSignupPopup />
        <Analytics />
      </body>
    </html>
  );
}
