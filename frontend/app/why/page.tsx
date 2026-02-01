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
              The Decoupling of Compute and Power.
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
              <p>
                While AI chip performance doubles rapidly—following the curve of
                Moore&apos;s Law and beyond—data center power infrastructure
                upgrades take years. Permits, capital allocation, and physical
                construction move at a pace that silicon does not.
              </p>
              <p>
                The result is a &quot;Thermal Wall&quot;: facilities cannot
                deploy the newest hardware (H100s, B200s, and beyond) due to heat
                constraints, not space constraints. Racks sit partially empty.
                Chips throttle. The bottleneck is thermodynamics, not
                transistor count.
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
                Legacy CRAC and CRAH units wait for a temperature sensor to
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
              Intelligence Precedes Heat.
            </h2>
            <div className="mt-12 space-y-8 text-white/80 leading-relaxed">
              <p>
                CooledAI ingests real-time telemetry: workload scheduling, GPU
                voltage draw, ambient rack temperatures, and historical thermal
                signatures. We build a predictive thermal model that anticipates
                energy demand before it manifests as heat.
              </p>
              <p>
                We don&apos;t react to heat. We anticipate it. Pre-cooling kicks
                in before the spike. Fans and chillers ramp in sync with
                compute, not in response to it. The result is a thermal envelope
                that stays flat—no spikes, no throttling, no wasted cycles.
              </p>
            </div>
          </section>

          {/* Section 4: Three Vectors of Value */}
          <section className="py-32 border-t border-white/20">
            <h2 className="mb-16 text-2xl font-medium tracking-tight text-white md:text-3xl">
              Three Vectors of Value
            </h2>
            <ValueVectors />
          </section>

          {/* Section 5: The Final Call */}
          <section className="py-32 border-t border-white/20">
            <p className="max-w-2xl text-2xl font-medium tracking-tight text-white md:text-3xl">
              The AI boom will not be constrained by silicon. It will be
              constrained by thermodynamics.
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
          CooledAI · Energy Autonomy for the AI Era
        </div>
      </footer>
    </div>
  );
}
