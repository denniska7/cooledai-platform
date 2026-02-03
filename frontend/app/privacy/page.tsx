import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Privacy Policy | CooledAI",
  description:
    "Enterprise-grade Privacy Policy for CooledAI LLC. CCPA/CPRA compliant. Industrial and facility data, shadow mode protection, no data monetization.",
};

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-transparent">
      <NavBar />

      <main className="mx-auto max-w-3xl px-6 pt-24 pb-16">
        <h1 className="text-3xl font-medium tracking-tight text-white md:text-4xl">
          Privacy Policy
        </h1>
        <p className="mt-2 text-sm text-white/60">
          Last updated: January 2026 · CooledAI LLC
        </p>

        <div className="legal-prose mt-12 space-y-10 text-white/85 leading-relaxed">
          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              1. Introduction
            </h2>
            <p>
              CooledAI LLC (“CooledAI,” “we,” “us,” or “our”) operates cooledai.com
              and enterprise thermal optimization services (the “Services”). This
              Privacy Policy describes how we collect, use, disclose, and protect
              information when your organization uses our platform, including
              industrial and facility data. We are committed to transparency,
              control integrity, and compliance with applicable privacy laws,
              including the California Consumer Privacy Act (CCPA) and the
              California Privacy Rights Act (CPRA). This policy is intended to
              satisfy the requirements of Directors of IT Infrastructure and
              Compliance Officers in enterprise and critical infrastructure
              environments.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2. Information We Collect
            </h2>
            <p className="mb-4">
              We collect the following categories of information in connection with
              our enterprise infrastructure services:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong className="text-white">Contact and account information:</strong>{" "}
                Name, email address, company name, job title, and other information
                you provide when requesting an audit, signing up for our beta, or
                contacting us.
              </li>
              <li>
                <strong className="text-white">Usage and analytics:</strong>{" "}
                Basic analytics such as pages visited, referral source, device type,
                and general usage patterns to improve our Services. We may use
                cookies and similar technologies as described in our{" "}
                <Link href="/cookies" className="text-white underline hover:no-underline">Cookie Policy</Link>.
              </li>
              <li>
                <strong className="text-white">Technical and operational data:</strong>{" "}
                When you use our portal or integration services, we may process
                non-personal telemetry (e.g., aggregate performance metrics) as
                necessary to deliver and operate the Services.
              </li>
            </ul>
            <h3 className="text-lg font-medium tracking-tight text-white mt-8 mb-3">
              2.1 Industrial & Facility Data
            </h3>
            <p className="mb-2">
              In the course of providing thermal optimization and efficiency
              services, we may ingest <strong className="text-white">industrial and
              facility data</strong>. This category includes telemetry data such as:
            </p>
            <ul className="list-disc pl-6 space-y-1 mb-4">
              <li>Temperature readings (inlet, outlet, ambient, and equipment-level)</li>
              <li>Fan speeds, airflow metrics, and cooling system status</li>
              <li>Power consumption and draw at rack or facility level</li>
              <li>Other operational metrics necessary to deliver predictive thermal modeling</li>
            </ul>
            <p>
              Such data is typically collected via industry-standard protocols
              (e.g., SNMP, BACnet, Modbus TCP) or through our Edge Agent or
              approved integrations. Industrial and facility data is used solely
              to provide and improve our Services, to generate efficiency reports,
              and to operate in accordance with your agreement. It is not used for
              advertising or unrelated commercial purposes.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2.2 Shadow Mode Protection (7-Day Efficiency Audit)
            </h2>
            <p>
              During the <strong className="text-white">7-Day Efficiency Test</strong> (also
              referred to as “Shadow Mode” or “read-only mode”), data collection
              is strictly <strong className="text-white">Read-Only</strong>. CooledAI
              does not transmit control commands, setpoint changes, or any
              write operations to the Partner’s infrastructure. We ingest
              telemetry for the purpose of building a thermal digital twin and
              proving ROI; no equipment is operated or modified by CooledAI during
              this phase. This read-only constraint is a core control-integrity
              safeguard for enterprise and critical infrastructure partners.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              3. How We Use Your Information
            </h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>To provide, operate, and improve our Services.</li>
              <li>To respond to inquiries and communicate with you about the Services.</li>
              <li>To send relevant product updates and marketing where you have consented or where permitted by law.</li>
              <li>To analyze usage and trends and to ensure security and compliance.</li>
              <li>To comply with legal obligations and enforce our agreements.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              4. Disclosure and Sharing
            </h2>
            <p className="mb-4">
              We do not sell personal information. We may share information with
              service providers (e.g., hosting, analytics, email delivery) who
              act on our behalf under contractual obligations to protect your data.
              We may disclose information where required by law or to protect our
              rights, safety, or property.
            </p>
            <h3 className="text-lg font-medium tracking-tight text-white mt-6 mb-3">
              No Data Monetization
            </h3>
            <p>
              CooledAI does not sell, trade, or share facility performance data,
              industrial telemetry, or operational metrics with third parties or
              competitors for marketing, advertising, or research purposes. Your
              facility data is used exclusively to deliver and support the
              Services under your agreement. We do not monetize partner or
              customer data.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              5. Your California Privacy Rights (CCPA/CPRA)
            </h2>
            <p className="mb-4">
              If you are a California resident, you have the right to:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong className="text-white">Know</strong> what personal information we collect, use, and disclose, including contact, account, and—where applicable—industrial and facility data categories described in this policy.</li>
              <li><strong className="text-white">Delete</strong> your personal information, subject to certain exceptions (e.g., where retention is required by law or for legitimate business purposes).</li>
              <li><strong className="text-white">Correct</strong> inaccurate personal information.</li>
              <li><strong className="text-white">Limit use of sensitive personal information</strong> to specified purposes.</li>
              <li><strong className="text-white">Non-discrimination</strong> for exercising these rights.</li>
            </ul>
            <p className="mt-4">
              These rights apply to personal information we hold, including
              information linked to industrial and facility data where it is
              associated with an identifiable individual or organization. To
              exercise these rights, contact us at{" "}
              <a href="mailto:legal@cooledai.com" className="text-white underline hover:no-underline">legal@cooledai.com</a> or
              via our{" "}
              <Link href="/#request-audit" className="text-white underline hover:no-underline">
                Contact
              </Link>{" "}
              form. We will verify your identity before processing your request.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              6. Data Retention and Security
            </h2>
            <p className="mb-4">
              We retain your information only as long as necessary to fulfill the
              purposes described in this policy or as required by law or your
              agreement.
            </p>
            <p className="mb-4">
              We protect data using industry-standard encryption and controls.
              <strong className="text-white"> Data in transit</strong> is protected
              using TLS (Transport Layer Security). <strong className="text-white">Data
              at rest</strong> is protected using AES-256 encryption where applicable.
              We implement technical and organizational measures to protect your
              information—including industrial and facility data—against
              unauthorized access, alteration, disclosure, or destruction, and we
              require the same standards from service providers that process data
              on our behalf.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              7. Changes and Contact
            </h2>
            <p>
              We may update this Privacy Policy from time to time. The “Last updated”
              date at the top will reflect the latest version. Continued use of the
              Services after changes constitutes acceptance of the updated policy.
            </p>
            <p className="mt-6">
              For privacy-related questions, data subject requests, or to exercise
              your rights under this policy or applicable law, contact:
            </p>
            <p className="mt-3 text-white font-medium">
              CooledAI LLC
            </p>
            <p className="mt-1">
              <strong className="text-white">Primary contact:</strong>{" "}
              <a href="mailto:legal@cooledai.com" className="text-white underline hover:no-underline">legal@cooledai.com</a>
            </p>
            <p className="mt-1 text-white/90">
              Headquarters: Roseville, CA
            </p>
            <p className="mt-3 text-white/70 text-sm">
              You may also use our website{" "}
              <Link href="/#request-audit" className="text-white underline hover:no-underline">Contact</Link> form
              for general inquiries; for legal and compliance matters, we recommend
              legal@cooledai.com.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
