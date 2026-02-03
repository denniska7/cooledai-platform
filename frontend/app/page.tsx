"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { NavBar } from "../components/NavBar";
import { LiveSystemPulse } from "../components/LiveSystemPulse";
import { LeadForm } from "../components/LeadForm";

export default function HomePage() {
  return (
    <div className="relative min-h-screen bg-transparent text-white overflow-hidden">
      <NavBar />

      <main className="relative pt-20">
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
            AI-optimized energy saving that automates thermal control and
            scales from one rack to hundreds of megawatts. Stop reacting to
            heat—predict it.
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
              Request Custom Audit
            </Link>
            <Link
              href="/optimization"
              className="rounded border border-white bg-transparent px-6 py-4 text-sm font-medium tracking-tight text-white transition-opacity hover:opacity-90"
            >
              See the Science
            </Link>
          </motion.div>
        </section>

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
              facility and adjusts in real time—reducing energy use by up to 12%
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
              Start with an Audit
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
            AI chips use more power than traditional cooling can handle. We cut
            cooling costs by 12% while keeping servers running at full speed.
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

        {/* Target Markets */}
        <section className="mx-auto max-w-4xl px-6 py-24 border-t border-white/20">
          <motion.h2
            initial={{ opacity: 0, x: -80 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl"
          >
            Target Markets
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, x: -60 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-8 space-y-6 text-white/80 leading-relaxed"
          >
            <p>
              CooledAI serves hyperscalers, colocation providers, and enterprise
              data centers—from 100 kW pilot deployments to 100+ MW fleets.
              Scalability is built in: add sites, add capacity, add autonomy.
            </p>
            <p>
              <strong className="text-white">High-Criticality Partners:</strong>{" "}
              Financial institutions and healthcare providers rely on 99.99%+
              uptime. CooledAI delivers predictive thermal control that meets
              the strictest SLAs—without sacrificing efficiency.
            </p>
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
