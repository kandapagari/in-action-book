# AGENTS.md

## Cursor Cloud specific instructions

This is the Astro site for the open-access textbook "Action Models for Robot
Learning". Standard commands are in `package.json` / `README.md`.

### Node version
- Requires **Node 24** (`package.json` → `engines: 24.x`, `.nvmrc`).
- `npm test` imports `.ts` modules and needs Node 24's native type-stripping;
  Node 22 fails with `ERR_UNKNOWN_FILE_EXTENSION`.
- The VM snapshot pins Node 24 ahead of the platform's `/exec-daemon/node`
  (v22) via shims in `/usr/local/cargo/bin/`. If `node --version` shows 22,
  re-run the shim or use `bash -lc '...'`.

### Book sync (`npm run sync:book`)
- In the full monorepo, this runs `../tools/sync-book-to-site.sh` to mirror
  `../book/` → `src/content/book/`.
- In this standalone checkout that script is absent. `scripts/sync-book.sh`
  no-ops and keeps the committed `src/content/book/` mirror, so
  `npm run dev` / `npm run build` work without the monorepo tools tree.
- **Never edit files under `src/content/book/` by hand** when the monorepo
  sync is available — it gets clobbered.

### Commands
- Dev: `npm run dev` → http://localhost:4321
- Build: `npm run build`
- Test: `npm test`
- There is no lint script.

### Environment variables / secrets
- All secrets are **optional for local development** (see `.env.example`).
  Without them the site fully builds and runs:
  - `/api/views/*` → `{ "count": null, "error": "kv_unavailable" }` (needs
    Upstash Redis `KV_REST_API_*`).
  - `/api/newsletter/*` → HTTP 500 (needs SMTP/Resend), but the prerendered
    newsletter pages still build.
  - Read-only MCP at `/api/mcp/` and `/search/` work with no secrets.
- Dynamic routes require a trailing slash (`trailingSlash: 'always'`), e.g.
  `/api/views/site/`.
