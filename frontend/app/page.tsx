"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { NavBar } from "../components/NavBar";
import { BackgroundMesh } from "../components/BackgroundMesh";
import { LiveSystemPulse } from "../components/LiveSystemPulse";
import { LeadForm } from "../components/LeadForm";

export default function HomePage() {
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">
      <BackgroundMesh />
      <NavBar />

      <main className="relative pt-20">
        {/* Hero */}
        <section className="mx-auto max-w-4xl px-6 py-32">
          <motion.h1
            initial={{ opacity: 0, y: 30, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-4xl font-medium tracking-tight text-white md:text-5xl lg:text-6xl"
          >
            Energy Freedom for the AI Era.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-6 max-w-2xl text-lg text-white/80"
          >
            CooledAI predicts heat before it happens for high-density servers.
            Stop reacting to heat—predict it.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
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
            initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
            whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-2xl font-medium tracking-tight text-white md:text-3xl"
          >
            The Thermal Wall
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
            whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            className="mt-6 max-w-2xl text-white/80 leading-relaxed"
          >
            AI chips use more power than traditional cooling can handle. We cut
            cooling costs by 12% while keeping servers running at full speed.
          </motion.p>
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
          CooledAI · Energy Freedom for the AI Era
        </div>
      </footer>
    </div>
  );
}
