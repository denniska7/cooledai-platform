import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Privacy Policy | CooledAI",
  description:
    "Privacy Policy for CooledAI LLC. California-compliant (CCPA) policy for B2B SaaS analytics and contact information.",
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
              and related services (the “Services”). This Privacy Policy describes
              how we collect, use, disclose, and protect information when you or
              your organization use our B2B SaaS platform and website. We are
              committed to transparency and compliance with applicable privacy
              laws, including the California Consumer Privacy Act (CCPA) and the
              California Privacy Rights Act (CPRA).
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2. Information We Collect
            </h2>
            <p className="mb-4">We collect the following categories of information:</p>
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
                cookies and similar technologies as described in our Cookie Policy.
              </li>
              <li>
                <strong className="text-white">Technical and operational data:</strong>{" "}
                When you use our portal or integration services, we may process
                non-personal telemetry (e.g., aggregate performance metrics) as
                necessary to deliver and operate the Services.
              </li>
            </ul>
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
            <p>
              We do not sell personal information. We may share information with
              service providers (e.g., hosting, analytics, email delivery) who
              act on our behalf under contractual obligations to protect your data.
              We may disclose information where required by law or to protect our
              rights, safety, or property.
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
              <li><strong className="text-white">Know</strong> what personal information we collect, use, and disclose.</li>
              <li><strong className="text-white">Delete</strong> your personal information, subject to certain exceptions.</li>
              <li><strong className="text-white">Correct</strong> inaccurate personal information.</li>
              <li><strong className="text-white">Limit use of sensitive personal information</strong> to specified purposes.</li>
              <li><strong className="text-white">Non-discrimination</strong> for exercising these rights.</li>
            </ul>
            <p className="mt-4">
              To exercise these rights, contact us at the address below or via our{" "}
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
            <p>
              We retain your information only as long as necessary to fulfill the
              purposes described in this policy or as required by law. We implement
              reasonable technical and organizational measures to protect your
              information against unauthorized access, alteration, disclosure, or
              destruction.
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
            <p className="mt-4">
              For privacy-related questions or to exercise your rights, contact:
            </p>
            <p className="mt-2 text-white font-medium">
              CooledAI LLC · Roseville, CA
            </p>
            <p>
              Via our website: <Link href="/#request-audit" className="text-white underline hover:no-underline">Contact</Link>.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
