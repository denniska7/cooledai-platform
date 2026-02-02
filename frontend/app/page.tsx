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
            CooledAI predicts heat before it happens for high-density servers.
            Stop reacting to heat—predict it.
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
