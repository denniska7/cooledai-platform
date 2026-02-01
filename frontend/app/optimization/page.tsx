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
            <h1 className="text-3xl font-medium tracking-tight text-white md:text-4xl">
              Legacy Cooling is Static. CooledAI is Fluid.
            </h1>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
              <p>
                Competitors use threshold-based logic: if temp &gt; 40°C, turn
                on fans. If temp &lt; 35°C, turn them off. The result is a
                jagged, reactive response—oscillation, overshoot, and wasted
                energy. The system is always chasing the last spike, never
                anticipating the next one.
              </p>
              <p>
                CooledAI uses inference-based logic. We don&apos;t wait for a
                sensor to spike. We predict thermal demand from workload
                scheduling, GPU voltage draw, and ambient conditions. The result
                is a smooth, predictive curve—no oscillation, no overshoot, no
                wasted cycles.
              </p>
            </div>
            <div className="mt-16">
              <StaticFluidDiagram />
            </div>
          </section>

          {/* Section 2: Proprietary Training Model */}
          <section className="py-32 border-t border-white/20">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Trained on 500,000+ Thermal Failure Hours.
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
              <p>
                Our model isn&apos;t a generic LLM. It&apos;s a specialized
                Reinforcement Learning (RL) model trained specifically on
                high-density server telemetry. We call it Thermal-Temporal
                Inference—it understands the thermal inertia of specific chip
                architectures: NVIDIA H100s, AMD EPYC, and the next generation
                of AI accelerators.
              </p>
              <p>
                The training data includes real failure scenarios—thermal
                runaways, cooling outages, workload spikes—from data centers
                running at the edge of capacity. The model learned to predict
                and prevent, not just react.
              </p>
            </div>
            <div className="mt-16">
              <NeuralMapDiagram />
            </div>
          </section>

          {/* Section 3: Hardware-Aware Intelligence */}
          <section className="py-32 border-t border-white/20">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Hardware-Aware Intelligence
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
              <p>
                CooledAI knows the hardware. It understands the specific
                heat-signature of different AI workloads—LLM training vs.
                inference, batch vs. real-time—and adjusts the environment
                before the workload even hits the silicon.
              </p>
              <p>
                An H100 under full training load has a different thermal profile
                than an H100 running inference. EPYC under mixed workload behaves
                differently than under sustained compute. Our model has learned
                these signatures. Pre-cooling kicks in when the scheduler
                assigns the job, not when the chip starts to heat.
              </p>
            </div>
          </section>

          {/* Section 4: Zero-Latency Edge Deployment */}
          <section className="py-32 border-t border-white/20">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Zero-Latency Edge Deployment
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
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
              </p>
            </div>
          </section>

          {/* Footer CTA */}
          <section className="py-32 border-t border-white/20">
            <p className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Experience the Shift in Thermodynamics.
            </p>
            <div className="mt-12">
              <Link
                href="/#request-audit"
                className="inline-block rounded border-2 border-white bg-transparent px-8 py-4 text-sm font-medium tracking-tight text-white transition-opacity hover:opacity-90"
              >
                Request a Blueprint
              </Link>
            </div>
          </section>
        </div>
      </main>

      <footer className="border-t border-white/20 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · Energy Autonomy for the AI Era
        </div>
      </footer>
    </div>
  );
}
