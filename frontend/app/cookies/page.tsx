import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Cookie Policy | CooledAI",
  description:
    "Cookie Policy for CooledAI LLC. How we use cookies and similar technologies on our website and services.",
};

export default function CookiesPage() {
  return (
    <div className="min-h-screen bg-transparent">
      <NavBar />

      <main className="mx-auto max-w-3xl px-6 pt-24 pb-16">
        <h1 className="text-3xl font-medium tracking-tight text-white md:text-4xl">
          Cookie Policy
        </h1>
        <p className="mt-2 text-sm text-white/60">
          Last updated: January 2026 · CooledAI LLC
        </p>

        <div className="legal-prose mt-12 space-y-10 text-white/85 leading-relaxed">
          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              1. What Are Cookies?
            </h2>
            <p>
              Cookies are small text files stored on your device when you visit
              a website. They help the site remember your preferences, understand
              how you use the site, and improve performance. We may also use
              similar technologies such as local storage and pixels where
              relevant.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2. How We Use Cookies
            </h2>
            <p className="mb-4">We use cookies and similar technologies for:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong className="text-white">Essential operation:</strong>{" "}
                Enabling core features such as navigation, login, and session
                management. These are necessary for the Services to function.
              </li>
              <li>
                <strong className="text-white">Analytics:</strong>{" "}
                Understanding how visitors use our website (e.g., pages viewed,
                referral source) so we can improve content and experience. We may
                use first-party or third-party analytics providers in accordance
                with our{" "}
                <Link href="/privacy" className="text-white underline hover:no-underline">
                  Privacy Policy
                </Link>
                .
              </li>
              <li>
                <strong className="text-white">Preferences:</strong>{" "}
                Remembering settings or choices you make (e.g., language or
                region) so we can personalize your experience.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              3. Types of Cookies We Use
            </h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong className="text-white">Strictly necessary:</strong> Required
                for the website and portal to work. You cannot opt out of these
                without affecting functionality.
              </li>
              <li>
                <strong className="text-white">Analytics / performance:</strong>{" "}
                Help us measure traffic and usage. You may be able to opt out via
                your browser or our cookie preferences if we offer them.
              </li>
              <li>
                <strong className="text-white">Functional:</strong> Remember your
                preferences and improve experience. Optional where legally
                required.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              4. Your Choices
            </h2>
            <p>
              Most browsers allow you to block or delete cookies through their
              settings. Blocking all cookies may limit your ability to use certain
              parts of our website or portal. Where required by law (e.g., in certain
              jurisdictions), we will obtain your consent before placing
              non-essential cookies. You can learn more about how we handle
              personal data in our{" "}
              <Link href="/privacy" className="text-white underline hover:no-underline">
                Privacy Policy
              </Link>
              .
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              5. Updates and Contact
            </h2>
            <p>
              We may update this Cookie Policy from time to time. The “Last updated”
              date at the top reflects the latest version. For questions about our
              use of cookies, contact CooledAI LLC at Roseville, CA, or via our{" "}
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
