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

### Honesty / methodology

The numbers are **page loads**, not unique humans. A refresh inside the
same browser tab is deduped via `sessionStorage` and counts once. Common
crawlers and social-preview fetchers (Googlebot, Bingbot, Twitterbot,
LinkedInBot, facebookexternalhit, generic `bot`/`crawler`/`spider`
substrings, etc.) are skipped server-side — they still get the current
count back, but the counter does not advance. `HEAD` requests are 204'd
without touching Redis. There is no IP-based rate limiting because that
would require storing IPs, which we explicitly do not do.
