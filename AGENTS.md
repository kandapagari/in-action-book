# AGENTS.md

## Cursor Cloud specific instructions

This repo is the `site/` Astro app for the open-access textbook "Action Models
for Robot Learning". In the full monorepo the authoring source lives at
`../book/` and is mirrored into `src/content/book/` by
`../tools/sync-book-to-site.sh`. **In this standalone cloud checkout only the
`site/` subtree exists**, so that sync script is absent — but the
`src/content/book/` mirror is already committed, so no sync is needed.

### Node version (important)
- The project requires **Node 24** (`package.json` → `engines: 24.x`). The
  committed `.nvmrc` says `22`, which is stale — `npm test` imports `.ts`
  modules and relies on Node 24's native TypeScript type-stripping, so it
  **fails on Node 22** with `ERR_UNKNOWN_FILE_EXTENSION`.
- Interactive shells already default to Node 24 (installed via `nvm`, and
  `~/.bashrc` prepends it ahead of the platform's built-in `node`). If a
  command runs under Node 22, prefix it with a login shell: `bash -lc '...'`.

### Do NOT use `npm run dev` / `npm run build`
- Their `predev`/`prebuild` hooks run `npm run sync:book`
  (`bash ../tools/sync-book-to-site.sh`), which **exits 127 here** because that
  script is not part of this checkout.
- Run Astro directly instead (the book mirror is already present):
  - Dev server: `npx astro dev` — serves http://localhost:4321
  - Production build: `npx astro build` — outputs `dist/` + Vercel functions
  - Preview built site: `npx astro preview`
- Tests are unaffected: `npm test` (runs `node --test src/lib/*.test.mjs`).

### Environment variables / secrets
- All secrets are **optional for local development** (see `.env.example`).
  Without them the site fully builds and runs:
  - The `/api/views/*` endpoints degrade gracefully to
    `{ "count": null, "error": "kv_unavailable" }` (needs Upstash Redis
    `KV_REST_API_*` to store counts).
  - The `/api/newsletter/*` endpoints return HTTP 500 (needs SMTP/Resend), but
    the prerendered newsletter pages still build.
  - The read-only MCP endpoint at `/api/mcp/` and the search page work with no
    secrets.
- Note API/dynamic routes require a trailing slash (`trailingSlash: 'always'`),
  e.g. `/api/views/site/`, not `/api/views/site`.
