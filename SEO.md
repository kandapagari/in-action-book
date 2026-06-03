# SEO — *Action Models for Robot Learning*

This file documents the on-page SEO that ships with the site and the off-page
work only the author can do. Canonical site URL: <https://action-models-book.vercel.app>.

## The realistic picture

- **Achievable on this subdomain:** ranking for the author's name
  (*Pavan Kumar Kandapagari*) and building him as a recognized **entity** tied
  to vision-language-action (VLA) models, robot learning, and action models.
  The book title (*Action Models for Robot Learning*) and long-tail topic
  queries (e.g. "open-access VLA textbook", "π₀ vs OpenVLA explained") are also
  realistic with quality content and a few good backlinks.
- **Capped on this subdomain:** broad, competitive head terms like
  "robot learning" or "VLA models" on their own. `*.vercel.app` is a shared
  domain — Google treats it as lower-authority than a registered domain you own,
  and you cannot build domain-level authority on a subdomain you share with
  everyone else on Vercel.
- **Single highest-impact future lever: a custom domain.** Moving to something
  like `actionmodels.book` / `pavan.dev/book` and 301-redirecting from the
  `.vercel.app` URL is the one change that lifts the ceiling on every
  competitive term. Do the name + entity work now; it carries over to the
  custom domain later.

> Entity-building beats keyword repetition. The `Person` + `Book` structured
> data plus **consistent `sameAs` links** across the web are what teach Google's
> Knowledge Graph that "Pavan Kumar Kandapagari" is a real entity in the
> VLA / robotics-foundation-models field. This is a months-long compounding
> effect, not an overnight ranking.

## Off-page action checklist (author to-do)

### 1. Profile + link consistency (do these first)
Add the site URL `https://action-models-book.vercel.app` to every profile, and
keep the **name and one-line bio byte-identical everywhere** (entity consistency
is the whole game):

- [ ] **GitHub profile** (`github.com/kandapagari`) — add the site to the
      "Website" field and to the profile README.
- [ ] **GitHub repo** (`in-action-book`) — set the repo "Website" field (About
      gear, top-right) to the site URL; mention it in the README header.
- [ ] **LinkedIn** (`linkedin.com/in/kandapagari`) — add the URL to the
      *About* section, pin it in *Featured*, and put it in the contact/website
      field. Use the same one-line bio.
- [ ] **Personal site** (`kandapagari.vercel.app`) — link to the book
      prominently; this is a same-owner backlink that reinforces the entity.
- [ ] **Google Scholar** — if/when a profile exists, add the site to the
      profile and keep the display name identical.
- [ ] **ORCID** — add the site under "Websites & social links"; ORCID is a
      strong author-identity signal.
- [ ] **arXiv author page** — link the site from the author identifier page.
- [ ] **Hugging Face** — add the site to the HF profile bio.
- [ ] **X/Twitter** — put the site in the bio.

> One-line bio to reuse verbatim everywhere:
> *"Pavan Kumar Kandapagari — foundation models for robotics: vision-language-action (VLA) policies, robot learning, action models. Author of the open-access textbook* Action Models for Robot Learning.*"*

Once these exist, **add the new URLs to `sameAs`** in `site/src/lib/seo.ts`
(`PERSON_SAME_AS`) so the structured data and the live links agree.

### 2. Launch + backlinks
- [ ] Publish a launch post linking back to the site: a **LinkedIn article**,
      a **dev.to** post, and/or a thread on **r/MachineLearning** or
      **r/robotics**. Lead with what's novel (open-access, from-first-principles
      VLA textbook), not keywords.
- [ ] Where natural, get the book linked from reading lists / awesome-lists
      (e.g. "awesome-robot-learning", "awesome-VLA") via PRs.

### 3. Index registration (do once, immediately after deploy)
- [ ] **Google Search Console** — verify the property. The site already supports
      a verification meta tag via the `PUBLIC_GOOGLE_SITE_VERIFICATION` env var
      (set it in Vercel; it's emitted in `<head>` by `SiteLayout.astro`).
- [ ] Submit the sitemap: `https://action-models-book.vercel.app/sitemap-index.xml`.
- [ ] **URL Inspection → Request Indexing** for the **home** and **About**
      pages first (these carry the `Person`/`Book` entity), then key sections.
- [ ] (Optional) **Bing Webmaster Tools** — same sitemap submission.

## On-page SEO already implemented (the current SEO surface)

So future sessions know what exists and where:

- **`site/src/lib/seo.ts`** — single source of truth for structured data.
  Exports `personEntity()`, `bookEntity()`, `webSiteEntity()`,
  `techArticleEntity()`, `breadcrumbList()`, `graph()`, and `detectTopics()`.
  `Person` and `Book` are defined once and referenced by stable `@id` IRIs
  (`#person`, `#book`, `#website`) so Google stitches the nodes into one entity.
  `PERSON_SAME_AS` and `PERSON_KNOWS_ABOUT` live here — **update `sameAs` here**
  as new public profiles come online.
- **Home (`index.astro`)** — emits a JSON-LD `@graph` of `WebSite` + `Person` +
  `Book`. Title/description front-load *VLA*, *robot learning*, *action models*
  and the author name.
- **About (`about.astro`)** — the canonical author-entity page. Emits
  `Person` + `Book`; the `Person`'s `mainEntityOfPage` points here. Title:
  *"About Pavan Kumar Kandapagari — author of Action Models for Robot Learning"*.
- **Section pages (`chapters/[chapter]/[section]/`)** — `TechArticle` (author →
  Person by `@id`, `isPartOf` → Book, `datePublished` from `PROGRESS.md` when
  available) + `BreadcrumbList` (Home → Contents → Section). `about`/`keywords`
  are derived from terms that **actually appear** in the section (`detectTopics`)
  — no stuffing.
- **Appendix pages (`appendix/[letter]/`)** — same `TechArticle` +
  `BreadcrumbList` treatment.
- **Meta descriptions** for sections/appendices are real prose excerpts
  (`excerpt()` over the rendered first chunk), not templated boilerplate.
- **Footer** — the author name links to `/about/` site-wide (`rel="author"`),
  an internal-linking + entity signal on every page.
- **Sitemap** (`@astrojs/sitemap` in `astro.config.mjs`) — `serialize` sets
  priorities (home 1.0, About 0.9, Contents 0.8, sections/appendices 0.7,
  utility 0.4), `changefreq`, and build-time `lastmod`. Output:
  `/sitemap-index.xml`.
- **`robots.txt`** (generated by `src/pages/robots.txt.ts`) — allows all
  crawlers and references the sitemap (URL derived from `site`).
- **Canonical / OG / Twitter** — emitted by `SiteLayout.astro` (absolute https
  URLs on the vercel domain). `noindex` is opt-in per page and currently used by
  none of the indexable routes.

## Do-not list (white-hat only)
- No hidden text, doorway pages, cloaking, or keyword-stuffed blocks.
- No private data (phone/address/email beyond the public contact page) in any
  structured data — `sameAs` is public professional profiles only.
- Don't change the book's prose meaning for SEO; prefer metadata/structured-data
  changes over body copy.
