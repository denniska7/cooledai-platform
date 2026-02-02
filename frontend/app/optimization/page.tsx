"use client";

import { motion } from "framer-motion";
import { NavBar } from "../../components/NavBar";
import Link from "next/link";
import { StaticFluidDiagram } from "../../components/optimization/StaticFluidDiagram";
import { NeuralMapDiagram } from "../../components/optimization/NeuralMapDiagram";

export default function OptimizationPage() {
  return (
    <div className="min-h-screen bg-black">
      <NavBar />

      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-6">
          {/* Section 1: Static vs Fluid */}
          <section className="py-32">
            <motion.h1
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-3xl font-medium tracking-tight text-white md:text-4xl"
            >
              Traditional Cooling is Static. CooledAI is Fluid.
            </motion.h1>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                Traditional systems use simple rules: if temp &gt; 40°C, turn on
                fans. If temp &lt; 35°C, turn them off. The result is a jagged,
                reactive response—bouncing, overshooting, and wasted energy. The
                system is always chasing the last spike, never anticipating the
                next one.
              </p>
              <p>
                CooledAI uses inference-based logic. We don&apos;t wait for a
                sensor to spike. We predict thermal demand from workload
                scheduling, GPU voltage draw, and ambient conditions. The result
                is a smooth, predictive curve—no oscillation, no overshoot, no
                wasted cycles. Scales from single-rack pilots to multi-megawatt
                fleets.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
              className="mt-16"
            >
              <StaticFluidDiagram />
            </motion.div>
          </section>

          {/* Section 2: Proprietary Training Model */}
          <section className="py-32 border-t border-white/20">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Trained on 500,000+ Thermal Failure Hours.
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                Our model isn&apos;t a generic AI. It&apos;s a specialized model
                trained on high-density server data. It understands how heat
                builds up in specific chips: NVIDIA H100s, AMD EPYC, and the
                next generation of AI accelerators.
              </p>
              <p>
                The training data includes real failure scenarios—thermal
                runaways, cooling outages, workload spikes—from data centers
                running at the edge of capacity. The model learned to predict
                and prevent, not just react.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
              className="mt-16"
            >
              <NeuralMapDiagram />
            </motion.div>
          </section>

          {/* Section 3: Hardware-Aware Intelligence */}
          <section className="py-32 border-t border-white/20">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Knows Your Hardware
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                CooledAI knows the hardware. It understands the heat pattern of
                different AI workloads—training vs. inference, batch vs.
                real-time—and adjusts cooling before the workload even starts.
              </p>
              <p>
                An H100 under full training has a different heat pattern than one
                running inference. Our model has learned these patterns.
                Pre-cooling kicks in when the job is scheduled, not when the
                chip starts to heat.
              </p>
            </motion.div>
          </section>

          {/* Section 4: Zero-Latency Edge Deployment */}
          <section className="py-32 border-t border-white/20">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Zero-Latency Edge Deployment
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                Our optimization doesn&apos;t happen in a slow cloud. It runs as
                a lightweight edge agent—locally in the data center—for
                sub-millisecond safety response. No round-trip to a remote API.
                No network latency. No single point of failure.
              </p>
              <p>
                When a thermal anomaly is detected, the agent responds in
                milliseconds. When a workload spike is predicted, pre-cooling
                ramps before the heat arrives. The edge deployment isn&apos;t
                just faster; it&apos;s the only architecture that can meet the
                real-time demands of high-density AI infrastructure.
                Scales from single sites to global multi-site deployments.
              </p>
            </motion.div>
          </section>

          {/* Footer CTA */}
          <section className="py-32 border-t border-white/20">
            <motion.p
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Experience the Shift in Heat Management.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12"
            >
              <Link
                href="/#request-audit"
                className="inline-block rounded border-2 border-white bg-transparent px-8 py-4 text-sm font-medium tracking-tight text-white transition-opacity hover:opacity-90"
              >
                Request a Blueprint
              </Link>
            </motion.div>
          </section>
        </div>
      </main>

      <footer className="border-t border-white/20 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · The Universal Autonomy Layer for Every Watt of Compute
        </div>
      </footer>
    </div>
  );
}
