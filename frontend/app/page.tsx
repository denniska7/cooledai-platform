import { NavBar } from "../components/NavBar";
import { SystemPulse } from "../components/SystemPulse";
import { LeadForm } from "../components/LeadForm";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-black">
      <NavBar />

      <main className="pt-20">
        {/* Hero */}
        <section className="mx-auto max-w-4xl px-6 py-24">
          <h1 className="text-4xl font-medium tracking-tight text-white md:text-5xl lg:text-6xl">
            Energy Autonomy for the AI Era.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-white/80">
            CooledAI provides predictive energy optimization for high-density
            data centers. Reduce cooling overhead by up to 12% without
            sacrificing a single cycle of compute performance.
          </p>
          <div className="mt-12 rounded border border-white/20 p-6 max-w-2xl">
            <p className="text-sm text-white/70 leading-relaxed">
              <span className="font-medium text-white">The Thermal Wall:</span>{" "}
              AI chips (H100s, H200s) are drawing more power than traditional
              infrastructure can handle. We solve the bottleneck.
            </p>
          </div>
          <div className="mt-16 max-w-sm">
            <SystemPulse />
          </div>
        </section>

        {/* Lead Form */}
        <LeadForm />
      </main>

      <footer className="border-t border-white/20 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40 tracking-tight">
          CooledAI · Energy Autonomy for the AI Era
        </div>
      </footer>
    </div>
  );
}
