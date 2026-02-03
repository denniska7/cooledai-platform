import Link from "next/link";
import { NavBar } from "../../components/NavBar";

export const metadata = {
  title: "Cookie Policy | CooledAI",
  description:
    "Cookie Policy for CooledAI LLC. Cookies, local storage, session storage, third-party technologies, and choices for enterprise B2B users.",
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
              how you use the site, and improve performance.
            </p>
            <p className="mt-4">
              This policy also covers <strong className="text-white">similar
              technologies</strong> we may use, including:
            </p>
            <ul className="list-disc pl-6 space-y-1 mt-2">
              <li>
                <strong className="text-white">Local Storage</strong> — browser
                storage that persists after the session ends; we may use it to
                store UI preferences or other non-sensitive settings.
              </li>
              <li>
                <strong className="text-white">Session Storage</strong> —
                temporary storage cleared when the browser tab or window is
                closed; we may use it for temporary session data or in-session
                state (e.g., form drafts or UI state).
              </li>
            </ul>
            <p className="mt-4">
              Where we refer to “cookies” in this policy, we include these
              technologies unless otherwise stated.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              2. How We Use Cookies
            </h2>
            <p className="mb-4">
              We use cookies and similar technologies for the following purposes:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong className="text-white">Essential operation:</strong>{" "}
                Enabling core features such as navigation and session management.
                These are necessary for the Services to function.
              </li>
              <li>
                <strong className="text-white">Security &amp; authentication:</strong>{" "}
                We use essential cookies or tokens to identify authorized users in
                the CooledAI Customer Portal and to protect against unauthorized
                access. Session state may be maintained via cookies or browser
                storage so that you remain authenticated during your visit. These
                are required for secure access to the portal.
              </li>
              <li>
                <strong className="text-white">Analytics / performance:</strong>{" "}
                Understanding how visitors use our website (e.g., pages viewed,
                referral source) so we can improve content and experience, in
                accordance with our{" "}
                <Link href="/privacy" className="text-white underline hover:no-underline">
                  Privacy Policy
                </Link>
                .
              </li>
              <li>
                <strong className="text-white">Preferences:</strong>{" "}
                Remembering settings or choices you make (e.g., UI preferences)
                so we can personalize your experience where applicable.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              3. Specific Third-Party Technologies
            </h2>
            <p className="mb-4">
              We use the following third-party technologies in connection with our
              website and services:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong className="text-white">Vercel Analytics</strong> — For
                site performance and usage analytics (e.g., page views, referral
                source). Data is processed in accordance with Vercel’s policies
                and our Privacy Policy.
              </li>
              <li>
                <strong className="text-white">Session / authentication
                management</strong> — The CooledAI Customer Portal uses
                browser-based session state (e.g., session or local storage, or
                cookies) to maintain secure user sessions. We do not currently
                use a separate third-party identity provider (e.g., Clerk or
                Supabase); if we introduce one, we will update this policy and
                our Privacy Policy accordingly.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              4. Types of Cookies We Use
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
              5. Key Cookies Used
            </h2>
            <p className="mb-4">
              The following table summarizes key cookies and similar technologies
              we use. This list may be updated as we add or change features.
            </p>
            <div className="overflow-x-auto border border-white/20 rounded my-6">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-white/20">
                    <th className="p-4 font-medium text-white">Name</th>
                    <th className="p-4 font-medium text-white">Category</th>
                    <th className="p-4 font-medium text-white">Purpose</th>
                    <th className="p-4 font-medium text-white">Duration</th>
                  </tr>
                </thead>
                <tbody className="text-white/85">
                  <tr className="border-b border-white/10">
                    <td className="p-4">[Example: session_id]</td>
                    <td className="p-4">Essential</td>
                    <td className="p-4">Portal authentication and session</td>
                    <td className="p-4">Session</td>
                  </tr>
                  <tr className="border-b border-white/10">
                    <td className="p-4">[Example: _vercel_analytics]</td>
                    <td className="p-4">Performance</td>
                    <td className="p-4">Site analytics (Vercel)</td>
                    <td className="p-4">Persistent</td>
                  </tr>
                  <tr>
                    <td className="p-4">[Local / Session Storage]</td>
                    <td className="p-4">Essential / Preferences</td>
                    <td className="p-4">UI preferences or temporary session data</td>
                    <td className="p-4">Session or Persistent</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-sm text-white/60">
              Specific cookie names and durations may vary. Contact us for a
              current list if required for your compliance or security review.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              6. Do Not Track (DNT)
            </h2>
            <p>
              Some browsers offer a “Do Not Track” (DNT) signal. CooledAI LLC
              does not currently respond to browser DNT signals, as no
              industry-wide standard for recognizing or honoring DNT has been
              adopted. We continue to follow the practices described in this
              Cookie Policy and our{" "}
              <Link href="/privacy" className="text-white underline hover:no-underline">
                Privacy Policy
              </Link>
              . If a standard emerges and we change our approach, we will update
              this policy.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              7. Your Choices
            </h2>
            <p>
              Most browsers allow you to block or delete cookies (and in some
              cases local or session storage) through their settings. Blocking
              all cookies may limit your ability to use certain parts of our
              website or portal, including secure access to the Customer Portal.
              Where required by law (e.g., in certain jurisdictions), we will
              obtain your consent before placing non-essential cookies. You can
              learn more about how we handle personal data in our{" "}
              <Link href="/privacy" className="text-white underline hover:no-underline">
                Privacy Policy
              </Link>
              .
            </p>
          </section>

          <section>
            <h2 className="text-xl font-medium tracking-tight text-white mt-8 mb-4">
              8. Updates and Contact
            </h2>
            <p>
              We may update this Cookie Policy from time to time. The “Last updated”
              date at the top reflects the latest version.
            </p>
            <p className="mt-4">
              For questions about our use of cookies and similar technologies:
            </p>
            <p className="mt-3 text-white font-medium">
              CooledAI LLC · Roseville, CA
            </p>
            <p className="mt-1">
              <strong className="text-white">Email:</strong>{" "}
              <a href="mailto:legal@cooledai.com" className="text-white underline hover:no-underline">legal@cooledai.com</a>
            </p>
            <p className="mt-3 text-white/80 text-sm">
              You may also use our website{" "}
              <Link href="/#request-audit" className="text-white underline hover:no-underline">Contact</Link> form
              for general inquiries; for legal and compliance matters we recommend
              legal@cooledai.com.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
