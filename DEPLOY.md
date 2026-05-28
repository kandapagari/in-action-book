# Deploying to Vercel

The site is static — `npm run build` produces `dist/` and Vercel serves it as
plain HTML. There is no SSR and no Vercel adapter is registered; the install
keeps `@astrojs/vercel` as a dependency in case we add image optimization or
ISR later.

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
