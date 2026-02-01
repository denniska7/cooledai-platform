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
          <h1 className="text-4xl font-medium tracking-tight md:text-5xl lg:text-6xl">
            Predictive Thermal Intelligence for the EPYC Era.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-white/70">
            Stop reacting to heat. Start predicting it. CooledAI reduces cooling
            overhead by up to 12% with industrial-grade fail-safes.
          </p>
          <div className="mt-16 max-w-sm">
            <SystemPulse />
          </div>
        </section>

        {/* Lead Form */}
        <LeadForm />
      </main>

      <footer className="border-t border-white/10 py-8">
        <div className="mx-auto max-w-6xl px-6 text-center text-xs text-white/40">
          CooledAI · Predictive Thermal Intelligence
        </div>
      </footer>
    </div>
  );
}
