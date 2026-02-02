# CooledAI Email Setup

Lead and Beta signup forms send emails to **dennis@cooledai.com** via [Resend](https://resend.com).

## Setup (5 minutes)

1. **Sign up at [resend.com](https://resend.com)** (free tier: 100 emails/day)

2. **Create an API key** at [resend.com/api-keys](https://resend.com/api-keys)

3. **Add environment variables** to your backend (Railway, `.env`, or local):

   ```
   RESEND_API_KEY=re_xxxxxxxxxxxx
   LEAD_EMAIL_TO=dennis@cooledai.com
   LEAD_EMAIL_FROM=CooledAI <onboarding@resend.dev>
   ```

4. **Verify your domain** (optional, for production):
   - In Resend dashboard: Domains → Add Domain → cooledai.com
   - Add the DNS records they provide
   - Then use: `LEAD_EMAIL_FROM=CooledAI <noreply@cooledai.com>`

## Without domain verification

Resend allows sending from `onboarding@resend.dev` for testing. Emails will arrive at `dennis@cooledai.com` as long as `RESEND_API_KEY` is set.

## Railway deployment

In Railway → your CooledAI API project → Variables:

- `RESEND_API_KEY` = your Resend API key
- `LEAD_EMAIL_TO` = dennis@cooledai.com (optional, this is the default)

## Local testing

```bash
export RESEND_API_KEY=re_xxxxxxxxxxxx
python3 -m uvicorn backend.api.main:app --reload --port 8000
```

Then submit a lead from the frontend. Check dennis@cooledai.com inbox.
