"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export default function ImplementationPage() {
  return (
    <div className="min-h-screen bg-black">
      <NavBar />

      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-6">
          {/* Hero */}
          <section className="py-24">
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-4xl font-medium tracking-tight text-white md:text-5xl"
            >
              From Theory to Autonomy in 30 Days.
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mt-6 max-w-2xl text-lg text-white/70 leading-relaxed"
            >
              CooledAI deploys in phases. Shadow Mode proves ROI before any
              control is handed over. Edge-first architecture ensures
              sub-millisecond fail-safes.
            </motion.p>
          </section>

          {/* Shadow Mode */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              7-Day Shadow Mode
            </h2>
            <div className="mt-8 space-y-6 text-white/80 leading-relaxed">
              <p>
                The AI observes your environment without making changes. We
                ingest real-time telemetry from your existing sensors and build
                a <strong className="text-white">Thermal Digital Twin</strong>{" "}
                of your facility. No control handover. No risk.
              </p>
              <p>
                After 7 days, you receive a Performance Report: predicted
                savings, thermal hotspots, and capacity headroom. ROI is proven
                before a single fan or chiller is touched.
              </p>
            </div>
            <div className="mt-10 rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
              <h3 className="text-sm font-medium text-white/90 uppercase tracking-wider mb-4">
                Shadow Mode Output
              </h3>
              <ul className="space-y-2 text-sm text-white/70">
                <li>• Thermal Digital Twin (3D heat map)</li>
                <li>• Predicted vs. actual cooling efficiency</li>
                <li>• Capacity headroom by zone</li>
                <li>• 7-day Performance Report with ROI projection</li>
              </ul>
            </div>
          </section>

          {/* Hardware Schematic */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl mb-8">
              Architecture
            </h2>
            <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-8">
              <svg viewBox="0 0 480 180" className="w-full max-w-2xl mx-auto" preserveAspectRatio="xMidYMid meet">
                <defs>
                  <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="rgba(255,255,255,0.5)" />
                  </marker>
                </defs>
                <rect x="20" y="60" width="80" height="60" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                <text x="60" y="88" fill="rgba(255,255,255,0.7)" fontSize="10" fontFamily="system-ui" textAnchor="middle">SNMP/BACnet</text>
                <text x="60" y="102" fill="rgba(255,255,255,0.5)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Modbus</text>
                <line x1="100" y1="90" x2="140" y2="90" stroke="rgba(255,255,255,0.4)" strokeWidth="1" markerEnd="url(#arrow)" />
                <rect x="140" y="50" width="100" height="80" fill="none" stroke="#00FFCC" strokeWidth="1.5" strokeOpacity="0.8" />
                <text x="190" y="85" fill="#FFFFFF" fontSize="11" fontFamily="system-ui" textAnchor="middle">Edge Agent</text>
                <text x="190" y="102" fill="#00FFCC" fontSize="9" fontFamily="system-ui" textAnchor="middle">AI Model</text>
                <text x="190" y="118" fill="rgba(255,255,255,0.5)" fontSize="8" fontFamily="system-ui" textAnchor="middle">&lt;1ms fail-safe</text>
                <line x1="240" y1="90" x2="280" y2="90" stroke="rgba(255,255,255,0.4)" strokeWidth="1" markerEnd="url(#arrow)" />
                <rect x="280" y="60" width="80" height="60" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                <text x="320" y="88" fill="rgba(255,255,255,0.7)" fontSize="10" fontFamily="system-ui" textAnchor="middle">CRAC/Chiller</text>
                <text x="320" y="102" fill="rgba(255,255,255,0.5)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Control</text>
                <rect x="380" y="60" width="80" height="60" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                <text x="420" y="88" fill="rgba(255,255,255,0.7)" fontSize="10" fontFamily="system-ui" textAnchor="middle">Rack</text>
                <text x="420" y="102" fill="rgba(255,255,255,0.5)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Telemetry</text>
                <line x1="190" y1="50" x2="190" y2="30" stroke="rgba(0,255,204,0.3)" strokeWidth="0.8" strokeDasharray="3" />
                <text x="190" y="22" fill="rgba(0,255,204,0.7)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Local Network</text>
              </svg>
            </div>
          </section>

          {/* Integration Specs */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Integration Specs
            </h2>
            <div className="mt-8 space-y-6 text-white/80 leading-relaxed">
              <p>
                CooledAI integrates with your existing building and IT
                infrastructure. We support industry-standard protocols:
              </p>
              <ul className="space-y-2 text-white/70">
                <li>
                  <strong className="text-white">SNMP v3</strong> — Network
                  gear, PDUs, BMS
                </li>
                <li>
                  <strong className="text-white">BACnet/IP</strong> — HVAC,
                  chillers, CRAC/CRAH
                </li>
                <li>
                  <strong className="text-white">Modbus TCP</strong> — Industrial
                  chillers, power meters
                </li>
              </ul>
              <p>
                Our <strong className="text-white">Edge-First</strong> philosophy
                means the AI agent runs locally on your network. No round-trip to
                the cloud. Sub-millisecond fail-safes when thermal anomalies are
                detected. The edge deployment is the only architecture that meets
                the real-time demands of high-density AI infrastructure.
              </p>
            </div>
            <div className="mt-10 rounded border border-[rgba(255,255,255,0.1)] bg-black p-6">
              <h3 className="text-sm font-medium text-white/90 uppercase tracking-wider mb-4">
                Edge-First Architecture
              </h3>
              <p className="text-sm text-white/70 leading-relaxed">
                The AI agent sits locally in your data center. Telemetry flows
                in. Optimization commands flow out. No cloud dependency for
                safety-critical control. Fail-safe response &lt;1ms.
              </p>
            </div>
          </section>

          {/* SaaS Pricing */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              SaaS Pricing
            </h2>
            <div className="mt-10 grid gap-6 md:grid-cols-2">
              <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-8">
                <h3 className="text-lg font-medium tracking-tight text-white">
                  Pilot
                </h3>
                <p className="mt-2 text-2xl font-medium text-white">
                  Up to 1 MW
                </p>
                <p className="mt-4 text-sm text-white/70 leading-relaxed">
                  7-day Audit + Performance Report. Shadow Mode only. No control
                  handover. Prove ROI before commitment.
                </p>
                <Link
                  href="/#request-audit"
                  className="mt-6 inline-block border border-white px-6 py-3 text-sm font-medium text-white transition-opacity hover:opacity-90"
                >
                  Request Pilot
                </Link>
              </div>
              <div className="rounded border border-[rgba(255,255,255,0.1)] bg-black p-8">
                <h3 className="text-lg font-medium tracking-tight text-white">
                  Enterprise
                </h3>
                <p className="mt-2 text-2xl font-medium text-white">
                  Per-MW / month
                </p>
                <p className="mt-4 text-sm text-white/70 leading-relaxed">
                  Full Autonomous Control + 24/7 Hardware Monitoring. Thermal
                  Digital Twin. Predictive shutdown. Mission Control Portal.
                </p>
                <Link
                  href="/#request-audit"
                  className="mt-6 inline-block border border-white bg-white px-6 py-3 text-sm font-medium text-black transition-opacity hover:opacity-90"
                >
                  Request Enterprise
                </Link>
              </div>
            </div>
          </section>

          {/* CTA */}
          <section className="py-24 border-t border-[rgba(255,255,255,0.1)]">
            <p className="max-w-2xl text-xl font-medium tracking-tight text-white">
              Ready to prove ROI in 7 days?
            </p>
            <Link
              href="/#request-audit"
              className="mt-6 inline-block border-2 border-white px-8 py-4 text-sm font-medium text-white transition-opacity hover:opacity-90"
            >
              Request Your 2026 Efficiency Blueprint
            </Link>
          </section>
        </div>
      </main>

      <footer className="border-t border-[rgba(255,255,255,0.1)] py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · Energy Freedom for the AI Era
        </div>
      </footer>
    </div>
  );
}
