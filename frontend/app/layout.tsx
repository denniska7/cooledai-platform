import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CooledAI | Predictive Thermal Intelligence",
  description:
    "Stop reacting to heat. Start predicting it. CooledAI reduces cooling overhead by up to 12% with industrial-grade fail-safes.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-black text-white">{children}</body>
    </html>
  );
}
