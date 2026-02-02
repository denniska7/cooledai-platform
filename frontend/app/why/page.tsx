"use client";

import { motion } from "framer-motion";
import { NavBar } from "../../components/NavBar";
import Link from "next/link";
import { CollisionGraph } from "../../components/why/CollisionGraph";
import { ReactionSchematic } from "../../components/why/ReactionSchematic";
import { ValueVectors } from "../../components/why/ValueVectors";

export default function WhyPage() {
  return (
    <div className="min-h-screen bg-transparent">
      <NavBar />

      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-6">
          {/* Section 1: The Collision Course */}
          <section className="py-32">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Chips Outpace Cooling.
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                AI chip performance doubles every 18–24 months. But data center
                cooling upgrades take years. Permits, budgets, and construction
                move at a pace that chips do not.
              </p>
              <p>
                The result is a &quot;Thermal Wall&quot;: facilities cannot
                deploy the newest hardware (H100s, B200s, and beyond) due to
                heat limits, not space. Racks sit partially empty. Chips slow
                down. The bottleneck is heat, not transistor count.
              </p>
            </motion.div>
            <div className="mt-16">
              <CollisionGraph />
            </div>
          </section>

          {/* Section 2: The Failure of Reaction */}
          <section className="py-32 border-t border-white/20">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Reactive Cooling is Already Too Late.
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                Traditional CRAC and CRAH units wait for a temperature sensor to
                spike before ramping up cooling. The control loop is simple:
                sense heat, then react. But AI workloads spike in milliseconds.
                A single training step can push a GPU from 40°C to 85°C before
                any cooling system has time to respond.
              </p>
              <p>
                This latency causes micro-throttling—the GPU backs off to protect
                itself—and cumulative hardware degradation. Thermal cycling
                (rapid heating and cooling) accelerates solder fatigue and
                reduces silicon lifespan. Reactive cooling doesn&apos;t just
                waste energy; it shortens the life of your most expensive
                assets.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
              className="mt-16"
            >
              <ReactionSchematic />
            </motion.div>
          </section>

          {/* Section 3: The Predictive Paradigm Shift */}
          <section className="py-32 border-t border-white/20">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Predict Before Heat.
            </motion.h2>
            <motion.div
              initial={{ opacity: 0, x: -60 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
              className="mt-12 space-y-8 text-white/80 leading-relaxed"
            >
              <p>
                CooledAI uses real-time data: workload, power use, rack temps,
                and past heat patterns. We build a model that predicts heat
                demand before it shows up.
              </p>
              <p>
                We don&apos;t react to heat. We anticipate it. Pre-cooling kicks
                in before the spike. Fans and chillers ramp in sync with
                compute, not in response to it. The result is a flat
                temperature—no spikes, no slowdown, no wasted cycles.
              </p>
            </motion.div>
          </section>

          {/* Target Markets */}
          <section className="py-32 border-t border-white/20">
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
                CooledAI scales from 100 kW pilot deployments to 100+ MW fleets.
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

          {/* Section 4: Three Vectors of Value */}
          <section className="py-32 border-t border-white/20">
            <motion.h2
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="mb-16 text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              Three Benefits
            </motion.h2>
            <ValueVectors />
          </section>

          {/* Section 5: The Final Call */}
          <section className="py-32 border-t border-white/20">
            <motion.p
              initial={{ opacity: 0, x: -80 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="max-w-2xl text-2xl font-medium tracking-tight text-white md:text-3xl"
            >
              The AI boom will not be limited by chips. It will be limited by
              heat.
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
                Request Your Efficiency Blueprint
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
