"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export default function ImplementationPage() {
  return (
    <div className="min-h-screen bg-transparent">
      <NavBar />

      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-6">
          {/* Hero */}
          <section className="py-24">
            <motion.h1
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-4xl font-medium tracking-tight text-white md:text-5xl"
            >
              From Theory to Autonomy in 30 Days.
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-6 max-w-2xl text-lg text-white/70 leading-relaxed"
            >
              CooledAI deploys in phases. The 7-Day Efficiency Test proves ROI before any
              control is handed over. Edge-first architecture ensures
              sub-millisecond fail-safes. Scales from single-rack pilots to
              multi-megawatt fleets.
            </motion.p>
          </section>

          {/* Hybrid Infrastructure Support */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <motion.div
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="rounded border border-[rgba(255,255,255,0.15)] bg-black/50 p-8"
            >
              <h3 className="text-xl font-medium tracking-tight text-white">
                Hybrid Infrastructure Support
              </h3>
              <p className="mt-4 text-white/80 leading-relaxed">
                CooledAI manages Air-Cooled, Liquid-Cooled, and Immersion-Cooled
                environments simultaneously. Whether you run traditional CRAC
                units, direct-to-chip liquid, or full-rack immersion—one
                intelligence layer unifies thermal control across your entire
                fleet.
              </p>
            </motion.div>
          </section>

          {/* Shadow Mode */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              The 7-Day Efficiency Test
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-8 space-y-6 text-white/80 leading-relaxed"
            >
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
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
              className="mt-10 rounded border border-[rgba(255,255,255,0.1)] bg-black p-6"
            >
              <h3 className="text-sm font-medium text-white/90 uppercase tracking-wider mb-4">
                Shadow Mode Output
              </h3>
              <ul className="space-y-2 text-sm text-white/70">
                <li>• Thermal Digital Twin (3D heat map)</li>
                <li>• Predicted vs. actual cooling efficiency</li>
                <li>• Capacity headroom by zone</li>
                <li>• 7-day Performance Report with ROI projection</li>
              </ul>
            </motion.div>
          </section>

          {/* Hardware Schematic */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl mb-8"
            >
              Architecture
            </motion.h2>
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
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Integration Specs
            </motion.h2>
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

          {/* Data Connectivity & Security */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl mb-10"
            >
              Data Connectivity & Security
            </motion.h2>

            <h3 className="text-sm font-medium text-white/70 uppercase tracking-wider mb-6">
              Connectivity Options
            </h3>
            <div className="grid gap-6 md:grid-cols-3">
              {/* Option 1: Edge Agent */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4 }}
                className="rounded border border-white/10 bg-black p-6"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded border border-white/20">
                  <svg viewBox="0 0 24 24" className="h-6 w-6 text-white/80" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="4" y="4" width="16" height="16" rx="2" />
                    <path d="M4 12h16M12 4v16" />
                  </svg>
                </div>
                <h4 className="text-lg font-medium tracking-tight text-white">
                  CooledAI Edge Agent
                </h4>
                <span className="mt-1 inline-block text-xs text-white/50 uppercase tracking-wider">
                  Recommended
                </span>
                <p className="mt-4 text-sm text-white/70 leading-relaxed">
                  A lightweight Docker container deployed on your local VM. Uses Outbound-Only HTTPS (Port 443) to stream telemetry. No inbound firewall changes required.
                </p>
                <p className="mt-4 text-xs text-white/50">
                  <span className="font-medium text-white/70">Best for:</span> Facilities requiring fast deployment and zero-trust security.
                </p>
                <div className="mt-6">
                  <svg viewBox="0 0 200 60" className="w-full" preserveAspectRatio="xMidYMid meet">
                    <defs>
                      <marker id="edge-arrow" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                        <polygon points="0 0, 6 2, 0 4" fill="rgba(255,255,255,0.6)" />
                      </marker>
                    </defs>
                    <rect x="10" y="15" width="50" height="30" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                    <text x="35" y="32" fill="rgba(255,255,255,0.8)" fontSize="8" fontFamily="system-ui" textAnchor="middle">VM</text>
                    <line x1="60" y1="30" x2="130" y2="30" stroke="rgba(255,255,255,0.5)" strokeWidth="1" strokeDasharray="4" markerEnd="url(#edge-arrow)" />
                    <text x="95" y="22" fill="rgba(255,255,255,0.5)" fontSize="7" fontFamily="system-ui" textAnchor="middle">HTTPS 443</text>
                    <rect x="140" y="15" width="50" height="30" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                    <text x="165" y="32" fill="rgba(255,255,255,0.8)" fontSize="8" fontFamily="system-ui" textAnchor="middle">Cloud</text>
                  </svg>
                </div>
              </motion.div>

              {/* Option 2: Site-to-Site VPN */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: 0.05 }}
                className="rounded border border-white/10 bg-black p-6"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded border border-white/20">
                  <svg viewBox="0 0 24 24" className="h-6 w-6 text-white/80" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                </div>
                <h4 className="text-lg font-medium tracking-tight text-white">
                  Secure Site-to-Site VPN
                </h4>
                <p className="mt-4 text-sm text-white/70 leading-relaxed">
                  An encrypted IPsec tunnel between your facility and our sovereign cloud. Provides a dedicated, private network path for all control and monitoring data.
                </p>
                <p className="mt-4 text-xs text-white/50">
                  <span className="font-medium text-white/70">Best for:</span> Enterprise-scale deployments, multi-site management, and global scalability.
                </p>
                <div className="mt-6">
                  <svg viewBox="0 0 200 60" className="w-full" preserveAspectRatio="xMidYMid meet">
                    <rect x="10" y="15" width="45" height="30" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                    <text x="32" y="32" fill="rgba(255,255,255,0.8)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Facility</text>
                    <path d="M55 30 Q100 10 145 30" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="1" strokeDasharray="3" />
                    <text x="100" y="18" fill="rgba(255,255,255,0.5)" fontSize="7" fontFamily="system-ui" textAnchor="middle">IPsec</text>
                    <rect x="145" y="15" width="45" height="30" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                    <text x="167" y="32" fill="rgba(255,255,255,0.8)" fontSize="7" fontFamily="system-ui" textAnchor="middle">Sovereign</text>
                  </svg>
                </div>
              </motion.div>

              {/* Option 3: Cloud-to-Cloud API */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: 0.1 }}
                className="rounded border border-white/10 bg-black p-6"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded border border-white/20">
                  <svg viewBox="0 0 24 24" className="h-6 w-6 text-white/80" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M3 15a4 4 0 0 0 4 4h9a5 5 0 1 0-.1-9.999 5.002 5.002 0 0 0-9.78 2A3.5 3.5 0 0 0 3 15z" />
                  </svg>
                </div>
                <h4 className="text-lg font-medium tracking-tight text-white">
                  Cloud-to-Cloud API Integration
                </h4>
                <p className="mt-4 text-sm text-white/70 leading-relaxed">
                  Direct handshake with your existing DCIM/BMS provider (Schneider, Sunbird, Vertiv). Zero hardware or local software footprint.
                </p>
                <p className="mt-4 text-xs text-white/50">
                  <span className="font-medium text-white/70">Best for:</span> Facilities already utilizing modern cloud-based infrastructure management.
                </p>
                <div className="mt-6">
                  <svg viewBox="0 0 200 60" className="w-full" preserveAspectRatio="xMidYMid meet">
                    <ellipse cx="50" cy="30" rx="35" ry="18" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                    <text x="50" y="33" fill="rgba(255,255,255,0.8)" fontSize="7" fontFamily="system-ui" textAnchor="middle">DCIM/BMS</text>
                    <line x1="85" y1="30" x2="115" y2="30" stroke="rgba(255,255,255,0.5)" strokeWidth="1" />
                    <text x="100" y="22" fill="rgba(255,255,255,0.5)" fontSize="7" fontFamily="system-ui" textAnchor="middle">API</text>
                    <ellipse cx="150" cy="30" rx="35" ry="18" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1" />
                    <text x="150" y="33" fill="rgba(255,255,255,0.8)" fontSize="7" fontFamily="system-ui" textAnchor="middle">CooledAI</text>
                  </svg>
                </div>
              </motion.div>
            </div>

            {/* Sovereign Data Security */}
            <div className="mt-16 rounded border border-white/10 bg-black p-8">
              <h3 className="text-xl font-medium tracking-tight text-white">
                Sovereign Data Security
              </h3>
              <ul className="mt-6 space-y-4 text-sm text-white/80 leading-relaxed">
                <li className="flex gap-3">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-white/60" />
                  All data is encrypted at rest (AES-256) and in transit (TLS 1.3).
                </li>
                <li className="flex gap-3">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-white/60" />
                  Read-Only &apos;7-Day Test&apos; by default. Autonomous control requires multi-factor physical authorization.
                </li>
                <li className="flex gap-3">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-white/60" />
                  SOC2 Type II and ISO 27001 compliance-ready architecture.
                </li>
              </ul>
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

          {/* Beta Access Callout */}
          <section className="py-16 border-t border-[rgba(255,255,255,0.1)]">
            <div className="rounded border border-white/10 bg-black p-8">
              <h3 className="text-lg font-medium tracking-tight text-white">
                Beta Access
              </h3>
              <p className="mt-3 max-w-xl text-sm text-white/70 leading-relaxed">
                We are accepting 5 more high-density facilities for Q1 2026.
                Join the CooledAI Private Beta for early access to The 7-Day Efficiency Test
                and the Mission Control Portal.
              </p>
              <Link
                href="/#request-audit"
                className="mt-6 inline-block border border-white px-6 py-3 text-sm font-medium text-white transition-opacity hover:opacity-90"
              >
                Request Beta Access
              </Link>
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
          CooledAI · The Universal Autonomy Layer for Every Watt of Compute
        </div>
      </footer>
    </div>
  );
}
