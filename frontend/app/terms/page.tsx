import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Terms of Service | CooledAI",
  description:
    "Industrial infrastructure Terms of Service for CooledAI LLC. Read-only safety, limitation of liability, protocol compliance, governing law California.",
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
              bind that organization to these terms. These Terms are intended for
              enterprise and industrial infrastructure partners; read-only safety
              and control integrity are central to our service design.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2. Definition of Services
            </h2>
            <p className="mb-4">
              CooledAI provides predictive thermal optimization and efficiency
              analytics for data center and high-density compute environments.
              Our Services may include audits, dashboards, reports, and optional
              integration with your infrastructure. Specific features and scope
              are defined in separate agreements or product documentation.
            </p>
            <h3 className="text-lg font-medium tracking-tight text-white mt-6 mb-3">
              2.1 7-Day Efficiency Audit (Shadow Mode)
            </h3>
            <p>
              The <strong className="text-white">7-Day Efficiency Audit</strong> (also
              referred to as “Shadow Mode”) is a <strong className="text-white">non-intrusive,
              read-only data collection service</strong>. It does not interfere with
              or modify the physical operation of the Partner’s infrastructure.
              CooledAI does not send control commands, setpoints, or write
              operations to cooling equipment, building systems, or any other
              facility assets during this phase. The purpose is to establish a
              baseline and demonstrate potential efficiency gains; the Partner
              retains full operational control at all times.
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
              Shadow Mode) is designed to run without making any changes to your
              systems. During this phase:
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
              4.1 No Guarantee of Savings
            </h2>
            <p>
              While CooledAI aims to optimize efficiency, <strong className="text-white">specific
              energy savings or ROI percentages are estimates and are not
              guaranteed</strong>. Actual results depend on many variables outside
              our control, including but not limited to: weather and ambient
              conditions, facility maintenance practices, hardware age and
              configuration, power and cooling infrastructure design, and
              operational changes made by the Partner. Any projections or
              recommendations are for planning purposes only.
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
              5.1 Protocol Compliance
            </h2>
            <p>
              You are responsible for ensuring that providing CooledAI with access
              to your systems via SNMP, BACnet, Modbus TCP, or other industrial
              protocols does not violate your own internal security policies,
              change-management requirements, or third-party service agreements
              (e.g., with equipment vendors or managed-service providers). CooledAI
              does not assume responsibility for your compliance with such
              policies or agreements. You represent that you have obtained any
              necessary internal approvals before granting such access.
            </p>
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
            <p className="mb-4">
              The Services are provided “as is” to the extent permitted by law.
              We disclaim all warranties of merchantability, fitness for a
              particular purpose, and non-infringement.
            </p>
            <h3 className="text-lg font-medium tracking-tight text-white mt-6 mb-3">
              Limitation of Liability
            </h3>
            <p>
              CooledAI LLC shall not be liable for any <strong className="text-white">consequential,
              incidental, indirect, special, or punitive damages</strong>, including
              but not limited to: server downtime, hardware overheating, loss of
              data, loss of revenue or profits, business interruption, or any
              other commercial loss—<strong className="text-white">even if the user
              claims the software or service was at fault</strong>. Our total
              aggregate liability for any claims arising out of or related to
              these Terms or the Services shall not exceed the amount paid by
              you to CooledAI LLC in the twelve (12) months immediately
              preceding the event giving rise to the claim. If no fees were paid
              in that period, our liability shall not exceed one hundred U.S.
              dollars (USD $100). These limitations apply to the fullest extent
              permitted by applicable law.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              8. Infrastructure Indemnification
            </h2>
            <p>
              You agree to <strong className="text-white">indemnify, defend, and hold
              CooledAI LLC and its officers, directors, employees, and agents
              harmless</strong> from and against any and all claims, damages,
              losses, liabilities, costs, and expenses (including reasonable
              attorneys’ fees) arising from or related to <strong className="text-white">mechanical
              hardware failure</strong> of your facility equipment—including but
              not limited to chillers, fans, CRAC/CRAH units, pumps, and other
              cooling or power infrastructure—that occurs during or in connection
              with your use of the Services. This indemnification applies whether
              or not such failure is alleged to be related to the Services; the
              read-only nature of the 7-Day Efficiency Audit does not alter your
              responsibility for the operation and maintenance of your own
              infrastructure.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              9. Governing Law and Legal Entity
            </h2>
            <p>
              These Terms are governed by the laws of the State of{" "}
              <strong className="text-white">California</strong>, United States,
              without regard to its conflict-of-laws principles. Any dispute
              arising out of or relating to these Terms or the Services shall be
              subject to the exclusive jurisdiction of the state and federal
              courts located in California.
            </p>
            <p className="mt-4">
              <strong className="text-white">Legal Entity:</strong> CooledAI LLC, a
              limited liability company, with its headquarters at Roseville, CA.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              10. Changes and Contact
            </h2>
            <p>
              We may modify these Terms from time to time. Continued use after
              changes constitutes acceptance. For questions regarding these
              Terms, contact:
            </p>
            <p className="mt-3 text-white font-medium">
              CooledAI LLC · Roseville, CA
            </p>
            <p className="mt-1">
              <a href="mailto:legal@cooledai.com" className="text-white underline hover:no-underline">legal@cooledai.com</a>
              {" "}or via our{" "}
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
