import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Terms of Service | CooledAI",
  description:
    "Terms of Service for CooledAI LLC. Use of our B2B SaaS platform, AI-driven audits, and the 7-Day Efficiency Test.",
};

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-transparent">
      <NavBar />

      <main className="mx-auto max-w-3xl px-6 pt-24 pb-16">
        <h1 className="text-3xl font-medium tracking-tight text-white md:text-4xl">
          Terms of Service
        </h1>
        <p className="mt-2 text-sm text-white/60">
          Last updated: January 2026 · CooledAI LLC
        </p>

        <div className="legal-prose mt-12 space-y-10 text-white/85 leading-relaxed">
          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              1. Acceptance of Terms
            </h2>
            <p>
              By accessing or using the website, portal, or other services (the
              “Services”) of CooledAI LLC (“CooledAI,” “we,” “us,” or “our”), you
              agree to these Terms of Service. If you are using the Services on
              behalf of an organization, you represent that you have authority to
              bind that organization to these terms.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2. Description of Services
            </h2>
            <p>
              CooledAI provides predictive thermal optimization and efficiency
              analytics for data center and high-density compute environments.
              Our Services may include audits, dashboards, reports, and optional
              integration with your infrastructure. Specific features and
              scope are defined in separate agreements or product documentation.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              3. AI-Driven Audits and Reports — Informational Purpose
            </h2>
            <p className="mb-4">
              <strong className="text-white">Important:</strong> Our AI-driven efficiency
              audits, recommendations, and reports are provided for{" "}
              <strong className="text-white">informational and decision-support
              purposes only</strong>. They are not a substitute for professional
              engineering, safety, or compliance review. You are responsible for:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Validating any recommendations against your own policies and standards.</li>
              <li>Obtaining any required internal or regulatory approvals before making changes.</li>
              <li>Ensuring that your operations remain compliant with applicable laws and safety requirements.</li>
            </ul>
            <p className="mt-4">
              CooledAI does not guarantee specific outcomes (e.g., percentage
              savings or uptime) from following our recommendations. Actual
              results depend on your environment, configuration, and implementation.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              4. The 7-Day Efficiency Test — Read-Only (“Shadow Mode”)
            </h2>
            <p className="mb-4">
              Our “7-Day Efficiency Test” (also referred to in documentation as
              shadow or read-only mode) is designed to run without making any
              changes to your systems. During this phase:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong className="text-white">Read-only:</strong> We do not send
                control commands or setpoints to your cooling or building systems.
                We only collect data and generate predictions and reports.
              </li>
              <li>
                <strong className="text-white">No automatic control:</strong> No
                equipment is operated or modified by CooledAI during the test.
                You retain full control of your infrastructure.
              </li>
              <li>
                <strong className="text-white">Purpose:</strong> To establish a
                baseline, demonstrate potential savings, and allow you to evaluate
                our platform before any decision to enable automated or
                advisory control.
              </li>
            </ul>
            <p className="mt-4">
              If you later opt into control features, those will be governed by
              separate terms or agreements.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              5. Your Obligations
            </h2>
            <p>You agree to:</p>
            <ul className="list-disc pl-6 space-y-2 mt-2">
              <li>Provide accurate information and use the Services in compliance with law.</li>
              <li>Not misuse the Services, attempt to gain unauthorized access, or interfere with our systems or other users.</li>
              <li>Keep login credentials and API keys secure and notify us of any unauthorized use.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              6. Intellectual Property and Data
            </h2>
            <p>
              CooledAI retains all rights in its software, algorithms, and
              materials. You retain rights in your data. We use your data as
              described in our{" "}
              <Link href="/privacy" className="text-white underline hover:no-underline">
                Privacy Policy
              </Link>{" "}
              and any applicable data processing terms.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              7. Disclaimers and Limitation of Liability
            </h2>
            <p>
              The Services are provided “as is” to the extent permitted by law.
              We disclaim warranties of merchantability, fitness for a
              particular purpose, and non-infringement. In no event shall
              CooledAI be liable for indirect, incidental, special, or
              consequential damages, or for loss of profits or data, arising
              from your use of the Services. Our total liability shall not
              exceed the amount you paid us in the twelve (12) months preceding
              the claim, if any.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              8. Changes and Contact
            </h2>
            <p>
              We may modify these Terms from time to time. Continued use after
              changes constitutes acceptance. For questions, contact CooledAI
              LLC at Roseville, CA, or via our{" "}
              <Link href="/#request-audit" className="text-white underline hover:no-underline">
                Contact
              </Link>{" "}
              form.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
