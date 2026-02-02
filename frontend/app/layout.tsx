import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { BetaSignupPopup } from "../components/BetaSignupPopup";
import { GlobeDataCenters } from "../components/GlobeDataCenters";

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
      <body className="min-h-screen bg-black text-white">
        <GlobeDataCenters />
        {children}
        <BetaSignupPopup />
      </body>
    </html>
  );
}
