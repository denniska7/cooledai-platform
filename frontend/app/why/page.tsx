import { NavBar } from "../../components/NavBar";
import Link from "next/link";
import { CollisionGraph } from "../../components/why/CollisionGraph";
import { ReactionSchematic } from "../../components/why/ReactionSchematic";
import { ValueVectors } from "../../components/why/ValueVectors";

export default function WhyPage() {
  return (
    <div className="min-h-screen bg-black">
      <NavBar />

      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-6">
          {/* Section 1: The Collision Course */}
          <section className="py-32">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Chips Outpace Cooling.
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
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
            </div>
            <div className="mt-16">
              <CollisionGraph />
            </div>
          </section>

          {/* Section 2: The Failure of Reaction */}
          <section className="py-32 border-t border-white/20">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Reactive Cooling is Already Too Late.
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
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
            </div>
            <div className="mt-16">
              <ReactionSchematic />
            </div>
          </section>

          {/* Section 3: The Predictive Paradigm Shift */}
          <section className="py-32 border-t border-white/20">
            <h2 className="text-2xl font-medium tracking-tight text-white md:text-3xl">
              Predict Before Heat.
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
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
            </div>
          </section>

          {/* Section 4: Three Vectors of Value */}
          <section className="py-32 border-t border-white/20">
            <h2 className="mb-16 text-2xl font-medium tracking-tight text-white md:text-3xl">
              Three Benefits
            </h2>
            <ValueVectors />
          </section>

          {/* Section 5: The Final Call */}
          <section className="py-32 border-t border-white/20">
            <p className="max-w-2xl text-2xl font-medium tracking-tight text-white md:text-3xl">
              The AI boom will not be limited by chips. It will be limited by
              heat.
            </p>
            <div className="mt-12">
              <Link
                href="/#request-audit"
                className="inline-block rounded border-2 border-white bg-transparent px-8 py-4 text-sm font-medium tracking-tight text-white transition-opacity hover:opacity-90"
              >
                Request Your Efficiency Blueprint
              </Link>
            </div>
          </section>
        </div>
      </main>

      <footer className="border-t border-white/20 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · Energy Freedom for the AI Era
        </div>
      </footer>
    </div>
  );
}
