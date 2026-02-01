import { NavBar } from "../../components/NavBar";
import Link from "next/link";

export default function WhyPage() {
  return (
    <div className="min-h-screen bg-black">
      <NavBar />

      <main className="pt-20">
        <section className="mx-auto max-w-3xl px-6 py-24">
          <h1 className="text-3xl font-medium tracking-tight text-white md:text-4xl">
            Why CooledAI?
          </h1>

          <div className="mt-16 space-y-16">
            <div>
              <h2 className="text-lg font-medium tracking-tight text-white border-b border-white/20 pb-2">
                The Rising Cost of Compute
              </h2>
              <p className="mt-4 text-white/80 leading-relaxed">
                Electricity prices and AI compute density are rising in tandem,
                creating a margin crisis for data centers. Every watt of cooling
                overhead erodes profitability. Legacy systems were built for
                predictable, steady-state workloads—not the spiky, high-density
                demands of modern AI training and inference.
              </p>
            </div>

            <div>
              <h2 className="text-lg font-medium tracking-tight text-white border-b border-white/20 pb-2">
                Performance vs. Efficiency
              </h2>
              <p className="mt-4 text-white/80 leading-relaxed">
                Most systems throttle performance to save energy. CooledAI uses
                predictive modeling to keep chips at peak performance while
                optimizing the environment around them. No trade-offs. No
                compromise.
              </p>
            </div>

            <div>
              <h2 className="text-lg font-medium tracking-tight text-white border-b border-white/20 pb-2">
                The AI Boom
              </h2>
              <p className="mt-4 text-white/80 leading-relaxed">
                Legacy cooling systems were built for the 2010s cloud—low
                density, high redundancy, reactive control. The 2020s AI boom
                demands the opposite: high density, predictive control, and
                energy autonomy. CooledAI was built for this era.
              </p>
            </div>
          </div>

          <div className="mt-20">
            <Link
              href="/#request-audit"
              className="inline-block rounded border border-white bg-white px-6 py-3 text-sm font-medium tracking-tight text-black transition-opacity hover:opacity-90"
            >
              Request a Custom Efficiency Blueprint
            </Link>
          </div>
        </section>
      </main>

      <footer className="border-t border-white/20 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · Energy Autonomy for the AI Era
        </div>
      </footer>
    </div>
  );
}
