# Deploying to Vercel

The site is mostly static — `npm run build` prerenders every page to plain
HTML — but a single pair of endpoints under `/api/views/*` runs server-side
as Vercel serverless functions. They back the self-hosted view counter
(footer total + per-section count). See "View counter — Upstash Redis
setup" below for the one-time storage configuration. The Vercel adapter
(`@astrojs/vercel`) is registered in `astro.config.mjs`; chapter pages,
landing, contents, about, contact, search, RSS, and the sitemap remain
fully static.

## One-time setup (dashboard)

1. Push the repository to GitHub.
2. In Vercel → **Add New… → Project**, import the repo.
3. **Root directory**: `site/`.
4. **Framework preset**: Astro (auto-detected).
5. **Build command**: `npm run build` (default).
6. **Output directory**: `dist` (default).
7. **Node.js version**: 22.x (matches `.nvmrc`).
8. **Environment variables**:
   - `SITE_URL` — optional. Set this to the production URL once a custom
     domain is attached (e.g. `https://action-models.com`). It's read by
     `astro.config.mjs` and used for canonical URLs, RSS `<link>`, and
     sitemap absolute URLs. Without it the build falls back to
     `https://action-models-book.vercel.app`.
   - `PUBLIC_GOOGLE_SITE_VERIFICATION` — optional. The token Google Search
     Console generates for the "HTML tag" verification method. When set,
     the layout emits the corresponding `<meta name="google-site-verification">`
     tag so Google can verify ownership of the site. See "Google Search
     Console" below.
9. Click **Deploy**.

## One-time setup (CLI alternative)

```bash
npm i -g vercel
cd site
vercel link        # link this directory to a new Vercel project
vercel env add SITE_URL production   # paste your domain
vercel --prod
```

## Custom domain

Add the domain in Vercel → Project → **Settings → Domains**, then update the
`SITE_URL` env var so canonical/RSS/sitemap URLs use the real domain.

## What deploys automatically

- Every push to the project's default branch → production deploy.
- Every other branch / PR → preview deploy with its own URL.

No further configuration is needed. The build reads `../book/` at build
time, so committing new section markdown is the only thing required to ship
a new section.

## Connect the MCP server

The build also emits a remote, read-only **MCP** endpoint as a Vercel
serverless function (`prerender = false`), so any AI agent can read and
search the whole book from one URL — no install, always current:

```
https://action-models-book.vercel.app/api/mcp/
```

It exposes four read-only tools (`get_table_of_contents`, `get_section`,
`get_chapter`, `search`). To connect Cursor, add the server to
`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "action-models-book": {
      "url": "https://action-models-book.vercel.app/api/mcp/"
    }
  }
}
```

The `/mcp/` page on the site has copy-paste snippets for Cursor, Claude
Desktop (via the `mcp-remote` bridge), and other clients. No environment
variables or storage are needed — the endpoint is stateless and public.

## Google Search Console

The site is built to be discoverable: each page has a canonical URL, the
build emits `/sitemap-index.xml` and `/robots.txt`, the home page carries
`Book` JSON-LD, and every section page carries `Article` JSON-LD that links
back to the book. To make Google actually index it:

1. Visit <https://search.google.com/search-console> and click
   **Add property → URL prefix**.
2. Paste the production URL (currently `https://action-models-book.vercel.app/`,
   or your custom domain).
3. Choose the **HTML tag** verification method. Google shows a snippet like
   `<meta name="google-site-verification" content="abc123..." />`. Copy the
   `content` value only.
4. In Vercel → Project → **Settings → Environment Variables**, add
   `PUBLIC_GOOGLE_SITE_VERIFICATION` = the copied value, scoped to
   Production. Redeploy.
5. Back in Search Console, click **Verify**.
6. Once verified, open **Sitemaps** in the left nav and submit
   `sitemap-index.xml`.

Initial indexing typically takes a few days to a few weeks for a new
property. Use **URL Inspection** to request indexing of specific pages.

## View counter — Upstash Redis setup

The footer line ("Read N times since launch") and the per-section line
("This section has been read N times.") read from two Astro API routes:
`/api/views/site` and `/api/views/section/{chapter}/{section}`. Both store
state in an Upstash Redis store attached to the project. Only integer
counts are stored — no IPs, cookies, user-agents, timestamps, or
geolocation. Counts start at zero on the first deploy.

> Note: Vercel KV no longer exists as a first-party product. It was
> always Upstash Redis under the hood, and Vercel now surfaces Upstash
> directly through the marketplace. The code uses the `@upstash/redis`
> client and reads `KV_REST_API_URL` / `KV_REST_API_TOKEN` (falling back
> to `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` if those are
> what your store injects). The Upstash integration sets the
> `KV_REST_API_*` pair automatically. Leave the integration's **Custom
> Prefix** field empty — a prefix renames the variables and the code
> won't find them.

1. Vercel dashboard → your project → **Storage → Create Database →
   Upstash** ("Serverless DB (Redis, Vector, Queue, Search)"). Create a
   **Redis** database.
2. Pick a region near most readers.
3. When prompted, **connect it to this project** (leave the Custom Prefix
   empty). This auto-injects `KV_REST_API_URL`, `KV_REST_API_TOKEN` (plus
   `KV_REST_API_READ_ONLY_TOKEN`, `KV_URL`, and `REDIS_URL`, which the
   counter does not use) into the project's environment variables for the
   selected scopes — no manual editing required.
4. Trigger a redeploy (any push to the default branch, or
   `vercel --prod`). The counter starts ticking from zero on the first
   deploy that sees the new env vars.
5. **To reset the counter** (e.g. after testing): open the Upstash data
   browser (from the store's page in the Vercel dashboard, or the Upstash
   console) and run `DEL views:site` for the total, or
   `DEL views:section:{chapter}:{section}` for one section (e.g.
   `DEL views:section:1:1`). To wipe every per-section key at once, use a
   prefix scan / flush on `views:`.

### Local development

`npm run dev` works without Redis credentials. When the env vars are
absent the API routes return `{ count: null, error: "kv_unavailable" }`
with HTTP 200, the UI keeps its em-dash placeholder, and nothing is
logged as an error. To exercise the real counter locally, run
`vercel env pull` from the `site/` directory after linking the project —
that populates `.env.local` with the live `KV_REST_API_URL` and
`KV_REST_API_TOKEN`. Wipe `.env.local` again to go back to the
unavailable-graceful-degrade path.

## Email newsletter — email transport + Cron setup

The site offers a double-opt-in email list (`/newsletter/`) that drips one
**complete** chapter a week to each subscriber. Subscriber state lives in the
**same Upstash Redis** as the view counter (keys prefixed `nl:`); a **weekly
Vercel Cron** (`vercel.json`, Mondays 14:00 UTC) advances everyone. A chapter
is only sent once it is both complete *and* has an authored excerpt file at
`src/content/newsletter/chapter-<N>.md`.

Email delivery goes through one of **two selectable transports**, chosen by the
`EMAIL_TRANSPORT` env var. Both code paths are kept intentionally so you can
start sending **today without owning a domain** and switch later with no code
change:

- **`smtp` (default when unset)** — send via **Brevo SMTP**. Brevo lets you
  verify a single *sender email address* (no domain purchase or DNS required),
  so this is the fastest way to start. Free tier: **300 emails/day**.
- **`resend`** — send via **Resend**, which requires a **verified sending
  domain**. Better deliverability once you own a domain.

Unlike the view counter, the newsletter does **not** degrade silently: if the
Redis or the selected transport's configuration is missing, the
`/api/newsletter/*` endpoints return HTTP 500 rather than pretending a signup
succeeded. There is **no auto-fallback** between transports — if the transport
you selected is misconfigured, delivery throws. This is intentional.

`NEWSLETTER_FROM`, `CRON_SECRET`, and `SITE_URL` mean the same thing for both
transports. Only the transport-specific secrets differ.

### Default path (no domain): Brevo SMTP

1. **Redis** — reuse the Upstash store from the view counter (see above). No
   extra database is needed; the newsletter uses the same `KV_REST_API_*`
   (or `UPSTASH_REDIS_REST_*`) credentials.
2. **Brevo account + verified sender** — create a free account at
   <https://www.brevo.com>. Under **Senders, Domains & Dedicated IPs →
   Senders**, add your sender email and **verify it by clicking the link Brevo
   emails you**. No domain purchase and no DNS records are required for a single
   verified sender.
3. **SMTP key** — under **SMTP & API → SMTP**, generate an SMTP key. Note your
   SMTP **login** (username) and the generated **key** (used as the password —
   this is not your account password).
4. **Environment variables** — in Vercel → Project → **Settings →
   Environment Variables**, add (Production, and Preview if you want to test
   there):
   - `EMAIL_TRANSPORT` — `smtp` (or leave unset; `smtp` is the default).
   - `SMTP_HOST` — `smtp-relay.brevo.com`.
   - `SMTP_PORT` — `587` (STARTTLS; the code uses implicit TLS only on `465`).
   - `SMTP_USER` — your Brevo SMTP login.
   - `SMTP_PASS` — the Brevo SMTP key from step 3.
   - `NEWSLETTER_FROM` — the verified sender, e.g.
     `Action Models <your-verified-email>`.
   - `CRON_SECRET` — any long random string. Vercel automatically sends this
     as `Authorization: Bearer <CRON_SECRET>` on scheduled cron invocations,
     and the cron endpoint rejects any request that doesn't match.
   - `SITE_URL` — set to the production origin (also used for canonical URLs)
     so the confirm/unsubscribe/read links in emails are absolute and correct.
5. **Redeploy.** The `crons` entry in `vercel.json` registers the weekly job
   automatically on deploy (weekly schedules are allowed on the Hobby plan).

> Deliverability note: a personal/verified-sender address (no domain-level
> SPF/DKIM/DMARC) is more likely to land in **Promotions or spam** than an
> authenticated sending domain. It works, but for the best inbox placement move
> to the Resend + domain path below once you have a domain.

### Later path (verified domain): Resend

Switching is **config-only — no code change**:

1. **Resend account + sending domain** — create an account at
   <https://resend.com>, then **verify a real sending domain** under
   **Domains** (add the DNS records Resend shows: SPF/DKIM, and DMARC if you
   want it). A `.vercel.app` subdomain **cannot** send — you must own a real
   domain for the `From` address.
2. **API key** — under **API Keys**, create a key with send permission.
3. **Environment variables** — in Vercel, set:
   - `RESEND_API_KEY` — the Resend API key from step 2.
   - `NEWSLETTER_FROM` — a sender on the verified domain, e.g.
     `Action Models <newsletter@yourdomain.com>`.
   - `EMAIL_TRANSPORT` — flip to `resend`.
4. **Redeploy.** `CRON_SECRET` and `SITE_URL` are unchanged; the SMTP vars can
   be left in place (they're ignored while `EMAIL_TRANSPORT=resend`).

### How the drip works

- A visitor submits their email → `POST /api/newsletter/subscribe` stores a
  `pending` subscriber and emails a confirmation link (valid 48h).
- Clicking the link → `GET /api/newsletter/confirm` marks them `confirmed`,
  adds them to the `nl:confirmed` set, and sends Chapter 1 immediately.
- Each Monday the cron (`GET /api/newsletter/cron`) iterates confirmed
  subscribers and sends each the next sendable chapter past their personal
  cursor, then advances the cursor. Caught-up subscribers are skipped until a
  newly completed chapter gets its excerpt file.
- Every email carries a one-click unsubscribe link
  (`GET /api/newsletter/unsubscribe`).

### Authoring a chapter excerpt

When a chapter becomes complete, author its teaser using the prompt at
`scripts/newsletter/excerpt-prompt.md` and save the result to
`src/content/newsletter/chapter-<N>.md`. The chapter is not emailed until that
file exists. Excerpt files are committed like the rest of the content.

### Local development

`npm run dev` works without any of the newsletter env vars — the
`/api/newsletter/*` endpoints will error (by design, since delivery is
unconfigured), but the prerendered `/newsletter/*` pages render normally. To
exercise real sending locally, run `vercel env pull` after linking the project
to populate `.env.local`, or set the newsletter vars there by hand
(`EMAIL_TRANSPORT` + the matching transport group — `SMTP_*` for `smtp` or
`RESEND_API_KEY` for `resend` — plus `NEWSLETTER_FROM` / `CRON_SECRET`).

Run the pure drip-logic unit tests with `npm test` (`node --test`, no
framework).

### Honesty / methodology

The numbers are **page loads**, not unique humans. A refresh inside the
same browser tab is deduped via `sessionStorage` and counts once. Common
crawlers and social-preview fetchers (Googlebot, Bingbot, Twitterbot,
LinkedInBot, facebookexternalhit, generic `bot`/`crawler`/`spider`
substrings, etc.) are skipped server-side — they still get the current
count back, but the counter does not advance. `HEAD` requests are 204'd
without touching Redis. There is no IP-based rate limiting because that
would require storing IPs, which we explicitly do not do.
