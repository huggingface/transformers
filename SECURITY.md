# Stripe Secret & Webhook Security

## Do
- Store all Stripe keys in a `.env` file, never in code.
- Add `.env` to your `.gitignore`.
- Rotate any keys that were ever committed.
- Set up Stripe webhooks in your dashboard for each environment.
- Use Stripe test keys (`sk_test_...`) for development.
- Review logs and audit for accidental leaks.

## Do Not
- Do not expose `STRIPE_SECRET_KEY` to the frontend or in public repos.
- Do not log full webhook payloads containing sensitive information.
- Do not commit `.env` or real secrets to source control.

## Management
- Periodically rotate and review Stripe API keys and webhooks.
- Remove all unused or test webhooks from your Stripe dashboard.
- Use Stripe’s dashboard to monitor for suspicious activity.

> For help, see the official [Stripe Node.js docs](https://stripe.com/docs/keys#nodejs) and [webhook docs](https://stripe.com/docs/webhooks).