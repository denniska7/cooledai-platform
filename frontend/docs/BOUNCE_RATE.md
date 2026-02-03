# Why Bounce Rate Is ~45% and How We're Addressing It

## What "bounce rate" means
A **bounce** is a session where the user leaves without a second page view or meaningful interaction (e.g. no click to Why, Optimization, or the form). **45% bounce** = 45% of visits are single-page.

## Why it can be high

1. **Traffic mix** – Cold or unqualified visitors (wrong audience, one-off clicks) leave quickly. B2B landing pages often see 40–60% bounce.
2. **Single-page layout** – Everything is on the home page. If users don’t click nav or scroll to the form, analytics counts one page and they’re gone → bounce.
3. **High-commitment first CTA** – “Request Custom Audit” is strong; some visitors want a lower-commitment step first (e.g. “See how it works”).
4. **Form is far down** – On mobile, many never reach the form. No second interaction → bounce.
5. **Value prop clarity** – If the hero doesn’t match intent in the first few seconds, visitors leave before scrolling.

## Changes we made on-site

- **Explore strip** – Right after the hero, three clear next steps: Why CooledAI, See the Science, Request Blueprint. Gives a second click that isn’t only “submit form.”
- **Sticky CTA on scroll** – After the user scrolls past the hero, a compact bar appears (e.g. “Request your efficiency blueprint”) so they can convert without scrolling to the bottom.
- **Section IDs** – Sections have IDs for deep links and future scroll-depth tracking.
- **Strong secondary CTAs** – “See the Science” and in-page links to Why / Optimization so low-commitment visitors have a path before the form.

## What to do next (outside the codebase)

- **Traffic quality** – Focus on channels and messaging that attract data center / infra leads (e.g. LinkedIn, industry terms). Better fit = lower bounce.
- **Landing variants** – Test different hero lines or audience-specific landing pages; measure bounce and conversions.
- **Vercel Analytics** – Use “Paths” or “Top Pages” to see where people go after the home page; that shows whether they’re using Why/Optimization/Implementation.
