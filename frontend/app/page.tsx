"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { useEffect, useState } from "react";
import { NavBar } from "../components/NavBar";
import { LiveSystemPulse } from "../components/LiveSystemPulse";
import { LeadForm } from "../components/LeadForm";

export default function HomePage() {
  const [showStickyCta, setShowStickyCta] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      // Show sticky CTA after user scrolls past hero (~400px)
      setShowStickyCta(window.scrollY > 400);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="relative min-h-screen bg-transparent text-white overflow-hidden">
      <NavBar />

      <main className={`relative pt-20 ${showStickyCta ? "pb-20" : ""}`}>
        {/* Beta signup one-liner */}
        <div className="border-b border-white/20 bg-black/90">
          <div className="mx-auto max-w-6xl px-6 py-3 text-center">
            <p className="text-sm text-white/90 tracking-tight">
              Join the private beta—limited spots for Q1 2026.{" "}
              <Link
                href="/#request-audit"
                className="font-medium text-white underline decoration-white/60 underline-offset-2 hover:decoration-white"
              >
                Sign up for Beta Testing
              </Link>
            </p>
          </div>
        </div>

        {/* Hero */}
        <section className="mx-auto max-w-4xl px-6 py-32">
          <motion.h1
            initial={{ opacity: 0, x: -80, filter: "blur(10px)" }}
            animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-4xl font-medium tracking-tight text-white md:text-5xl lg:text-6xl"
          >
            The Universal Autonomy Layer for Every Watt of Compute.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, x: -60, filter: "blur(10px)" }}
            animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-6 max-w-2xl text-lg text-white/80"
          >
            Reclaim your cooling-constrained power capacity. AI-optimized thermal
            control that scales from one rack to hundreds of megawatts. Stop
            reacting to heat—predict it.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, x: -60, filter: "blur(10px)" }}
            animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
            className="mt-10 flex flex-wrap gap-4"
          >
            <Link
              href="/#request-audit"
              className="rounded border border-white bg-white px-6 py-4 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90"
            >
              Get My Savings Roadmap
            </Link>
            <Link
              href="/optimization"
              className="rounded border border-white bg-transparent px-6 py-4 text-sm font-medium tracking-tight text-white transition-opacity hover:opacity-90"
            >
              See the Science
            </Link>
          </motion.div>
          {/* Explore strip */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.5 }}
            className="mt-16 flex flex-wrap items-center justify-center gap-6 border-t border-white/10 pt-10"
          >
            <span className="text-xs font-medium uppercase tracking-widest text-white/50">Explore</span>
            <Link href="/why" className="text-sm text-white/70 hover:text-white transition-colors underline-offset-4 hover:underline">
              Why CooledAI
            </Link>
            <Link href="/optimization" className="text-sm text-white/70 hover:text-white transition-colors underline-offset-4 hover:underline">
              How it works
            </Link>
            <Link href="/implementation" className="text-sm text-white/70 hover:text-white transition-colors underline-offset-4 hover:underline">
              Implementation
            </Link>
            <Link href="/#request-audit" className="text-sm font-medium text-white underline underline-offset-4">
              Get My Savings Roadmap
            </Link>
          </motion.div>
        </section>

        {/* Security First — icon grid */}
        <section className="mx-auto max-w-5xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl mb-12"
          >
            Security First
          </motion.h2>
          <div className="grid gap-8 md:grid-cols-3">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="flex flex-col items-start rounded border border-white/20 bg-white/[0.02] p-8"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded border border-white/30 bg-black mb-6">
                <svg className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium tracking-tight text-white">Read-Only Integration</h3>
              <p className="mt-2 text-sm text-white/70">SNMP, BACnet, Modbus. We ingest telemetry only—no control commands to your infrastructure during the 7-Day Shadow Audit.</p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex flex-col items-start rounded border border-white/20 bg-white/[0.02] p-8"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded border border-white/30 bg-black mb-6">
                <svg className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0110 0v4" />
                </svg>
              </div>
              <h3 className="text-lg font-medium tracking-tight text-white">AES-256 Encryption</h3>
              <p className="mt-2 text-sm text-white/70">Data in transit (TLS) and at rest (AES-256). Enterprise-grade protection for your facility and telemetry data.</p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="flex flex-col items-start rounded border border-white/20 bg-white/[0.02] p-8"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded border border-white/30 bg-black mb-6">
                <svg className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  <path d="M9 12l2 2 4-4" />
                </svg>
              </div>
              <h3 className="text-lg font-medium tracking-tight text-white">Zero-Downtime Deployment</h3>
              <p className="mt-2 text-sm text-white/70">Edge agent and integrations deploy without taking systems offline. No scheduled outages required.</p>
            </motion.div>
          </div>
        </section>

        {/* The 7-Day Shadow Audit — timeline */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl mb-4"
          >
            The 7-Day Shadow Audit
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.05, ease: "easeOut" }}
            className="text-white/70 text-sm mb-12"
          >
            A clean, low-friction path to your savings roadmap.
          </motion.p>
          <ol className="space-y-8">
            <motion.li
              initial={{ opacity: 0, x: -40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5 }}
              className="flex gap-6"
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/30 bg-black text-sm font-medium text-white">1</span>
              <div>
                <h3 className="text-lg font-medium tracking-tight text-white">Simple Connection</h3>
                <p className="mt-1 text-sm text-white/60">~15 mins</p>
                <p className="mt-2 text-white/80">Connect via your existing SNMP, BACnet, or Modbus TCP. Read-only; no changes to setpoints or equipment.</p>
              </div>
            </motion.li>
            <motion.li
              initial={{ opacity: 0, x: -40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="flex gap-6"
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/30 bg-black text-sm font-medium text-white">2</span>
              <div>
                <h3 className="text-lg font-medium tracking-tight text-white">Passive Monitoring</h3>
                <p className="mt-1 text-sm text-white/60">7 Days</p>
                <p className="mt-2 text-white/80">Our AI observes and builds a thermal digital twin. No control commands; you keep full operational control.</p>
              </div>
            </motion.li>
            <motion.li
              initial={{ opacity: 0, x: -40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex gap-6"
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/30 bg-black text-sm font-medium text-white">3</span>
              <div>
                <h3 className="text-lg font-medium tracking-tight text-white">Results</h3>
                <p className="mt-1 text-sm text-white/60">Your Savings Roadmap</p>
                <p className="mt-2 text-white/80">A data-driven blueprint: capacity reclaimed, efficiency gains, and a clear path to full AI optimization.</p>
              </div>
            </motion.li>
          </ol>
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="mt-10"
          >
            <Link
              href="/#request-audit"
              className="rounded border border-white bg-white px-6 py-3 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90"
            >
              Get My Savings Roadmap
            </Link>
          </motion.div>
        </section>

        {/* Universal Compatibility — marquee */}
        <section className="border-t border-white/20 py-16 overflow-hidden">
          <p className="text-center text-xs font-medium uppercase tracking-widest text-white/50 mb-8">
            Universalist Design: Built for the hardware you already own.
          </p>
          <div className="flex animate-marquee gap-16 whitespace-nowrap text-white/40 text-sm font-medium tracking-tight">
            <span>Vertiv</span>
            <span>Schneider Electric</span>
            <span>Eaton</span>
            <span>HPE</span>
            <span>NVIDIA</span>
            <span className="text-white/20">·</span>
            <span>Vertiv</span>
            <span>Schneider Electric</span>
            <span>Eaton</span>
            <span>HPE</span>
            <span>NVIDIA</span>
          </div>
        </section>

        {/* Sticky CTA: visible after scroll so users can convert without reaching bottom */}
        {showStickyCta && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-0 left-0 right-0 z-40 border-t border-white/20 bg-black/95 backdrop-blur-sm py-3"
          >
            <div className="mx-auto flex max-w-4xl items-center justify-center gap-4 px-6">
              <span className="text-sm text-white/80 hidden sm:inline">Get your custom savings roadmap.</span>
              <Link
                href="/#request-audit"
                className="rounded border border-white bg-white px-5 py-2.5 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90"
              >
                Get My Savings Roadmap
              </Link>
            </div>
          </motion.div>
        )}

        {/* AI-Optimized Energy Saving */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl"
          >
            AI-Optimized Energy Saving
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-6 max-w-3xl space-y-4 text-white/80 leading-relaxed"
          >
            <p>
              CooledAI turns cooling into a predictable, automated layer. Instead
              of manual setpoints and reactive responses, our system learns your
              facility and adjusts in real time—reducing cooling energy use
              while keeping every watt of compute available.
            </p>
            <p>
              <strong className="text-white">Automation</strong> means fewer
              human interventions and fewer errors. Set safety bounds once; the AI
              handles the rest. <strong className="text-white">Scalability</strong>{" "}
              means the same platform runs in a single rack or across multiple
              sites. Add capacity without re-architecting—from pilot to
              enterprise without a new playbook.
            </p>
          </motion.div>
        </section>

        {/* Three Steps: Audit → Shadow → AI Optimization */}
        <section className="mx-auto max-w-5xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl mb-4"
          >
            From Audit to Autonomy
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.05, ease: "easeOut" }}
            className="max-w-2xl text-white/70 text-sm mb-16"
          >
            A clear path from first contact to full AI-driven optimization.
          </motion.p>

          <div className="grid gap-8 md:grid-cols-3 md:gap-12">
            {/* Step 1: Audit */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="relative rounded border border-white/20 bg-white/[0.02] p-8"
            >
              <div className="mb-6 flex h-20 w-20 items-center justify-center rounded border border-white/30 bg-black">
                <svg className="h-10 w-10 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>
              <span className="text-xs font-medium tracking-widest text-white/50 uppercase">Step 1</span>
              <h3 className="mt-2 text-xl font-medium tracking-tight text-white">Audit</h3>
              <p className="mt-3 text-sm text-white/70 leading-relaxed">
                We map your thermal profile and power draw. You get a custom
                efficiency blueprint—no obligation, no control changes yet.
              </p>
            </motion.div>

            {/* Step 2: Shadow (7-Day Test) */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative rounded border border-white/20 bg-white/[0.02] p-8"
            >
              <div className="mb-6 flex h-20 w-20 items-center justify-center rounded border border-white/30 bg-black">
                <svg className="h-10 w-10 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              </div>
              <span className="text-xs font-medium tracking-widest text-white/50 uppercase">Step 2</span>
              <h3 className="mt-2 text-xl font-medium tracking-tight text-white">7-Day Efficiency Test</h3>
              <p className="mt-3 text-sm text-white/70 leading-relaxed">
                Our AI observes your environment in read-only mode. It builds a
                thermal digital twin and proves ROI before any control is
                handed over.
              </p>
            </motion.div>

            {/* Step 3: AI Optimization */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-30px" }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="relative rounded border border-white/20 bg-white/[0.02] p-8"
            >
              <div className="mb-6 flex h-20 w-20 items-center justify-center rounded border border-white/30 bg-black">
                <svg className="h-10 w-10 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42M19.78 4.22l-1.42 1.42M5.64 18.36l-1.42 1.42" />
                </svg>
              </div>
              <span className="text-xs font-medium tracking-widest text-white/50 uppercase">Step 3</span>
              <h3 className="mt-2 text-xl font-medium tracking-tight text-white">AI Optimization</h3>
              <p className="mt-3 text-sm text-white/70 leading-relaxed">
                Full autonomous control. The AI manages setpoints and response
                in real time—sub-millisecond safety, maximum efficiency, at
                scale.
              </p>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-12 flex justify-center"
          >
            <Link
              href="/#request-audit"
              className="rounded border border-white bg-white px-6 py-3 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90"
            >
              Get My Savings Roadmap
            </Link>
          </motion.div>
        </section>

        {/* The Problem - The Thermal Wall */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl"
          >
            The Thermal Wall
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-6 max-w-2xl text-white/80 leading-relaxed"
          >
            AI chips use more power than traditional cooling can handle. Reclaim
            your cooling-constrained power capacity—keep servers at full speed
            without overbuilding cooling.
          </motion.p>
        </section>

        {/* The Lab vs. The Reality */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl"
          >
            The Lab vs. The Reality
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-6 max-w-2xl space-y-4 text-white/80 leading-relaxed"
          >
            <p>
              Hardware is tuned in labs—controlled temperatures, steady loads,
              ideal conditions. But data centers live in chaos: shifting
              workloads, ambient spikes, equipment failures, human error.
            </p>
            <p>
              CooledAI is the intelligence that manages the real-world chaos.
              From single-rack pilots to multi-megawatt fleets, we scale with
              your infrastructure.
            </p>
          </motion.div>
        </section>

        {/* Mission-Critical Infrastructure */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl"
          >
            Mission-Critical Infrastructure
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-8 space-y-6 text-white/80 leading-relaxed"
          >
            <p>
              Whether it’s a regional healthcare node, a high-density mining
              fleet, or a Tier 4 financial data center, CooledAI is built for
              environments where downtime is a non-starter. Our autonomy layer
              is protocol-agnostic, providing a unified intelligence shield across
              disparate hardware, legacy chillers, and next-gen liquid-cooled
              clusters. We meet the world’s strictest SLAs by predicting thermal
              chaos before it impacts your uptime.
            </p>
            <div className="inline-flex items-center gap-2 rounded border border-white/30 bg-white/[0.03] px-4 py-2 mt-4">
              <span className="text-xs font-medium uppercase tracking-widest text-white/50">
                Protocol Agnostic
              </span>
              <span className="text-white/40">·</span>
              <span className="text-sm text-white/70 tracking-tight">
                SNMP · BACnet · Modbus · MQTT
              </span>
            </div>
          </motion.div>
        </section>

        {/* Live System Pulse */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <LiveSystemPulse />
        </section>

        {/* Lead Form */}
        <LeadForm />
      </main>

      <footer className="border-t border-white/20 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · The Universal Autonomy Layer for Every Watt of Compute
        </div>
      </footer>
    </div>
  );
}
