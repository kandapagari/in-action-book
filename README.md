# site/ — Action Models for Robot Learning

Astro static site that renders the open-access textbook from a mirror of the
authoring source at `../book/`.

## Content layout — read this first

- **Authoring source** lives at `../book/` (workspace root). The user and the
  daily scheduled task edit there; the `tools/*.py` scripts also operate on
  that tree. Treat it as the single source of truth.
- **Astro mirror** lives at `src/content/book/`. This is a generated, one-way
  copy of `../book/` produced by `tools/sync-book-to-site.sh`. **Never edit
  files under `src/content/book/` by hand** — every `npm run dev` and
  `npm run build` clobbers the mirror via a `predev`/`prebuild` hook.

## Run

```bash
nvm use            # Node 22 LTS (pinned in .nvmrc)
npm install
npm run dev        # syncs book → mirror, then serves http://localhost:4321
npm run build      # syncs book → mirror, then produces dist/
npm run preview    # serves dist/ locally
npm run sync:book  # manual one-way sync (book/ → src/content/book/)
```

The sync uses `rsync -a --delete` and excludes `.DS_Store`, `*~`, `*.bak`,
`*.orig`. It deletes anything in the mirror that no longer exists in the
source.

## How content flows

- **Sections** — `src/content.config.ts` registers a `chapters` content
  collection whose loader (`src/lib/book-loader.ts`) globs
  `src/content/book/chapter_*/section_*.md`, normalizes a couple of YAML
  quirks in the frontmatter (titles that contain colons; list items that look
  like single-key maps), and feeds each file through Astro's standard
  markdown pipeline. Section IDs are the value of the `section` field
  (`1.1`, `2.x`, `4.1`).
- **TOC + progress meter** — `src/lib/progress.ts` parses
  `src/content/book/PROGRESS.md` (imported via Vite's `?raw`) into a
  structured TOC, per-section statuses, and per-section drafted dates from
  the daily session log. This drives the `/contents/` page and the
  "Recently drafted" strip on the landing page.
- **Sidenotes** — the dynamic section page lifts `key_refs` from frontmatter
  and places them in the Tufte-style right margin (one after each H2-split
  body chunk; extras stack at the end). On viewports ≤900px the CSS collapses
  them inline.

## Math, code, search, RSS

- KaTeX (self-hosted via `katex/dist/katex.min.css` bundled by Vite) renders
  `$inline$` and `$$display$$` math.
- Shiki renders code blocks with `github-light` + `github-dark` themes
  swapped at runtime by the `data-theme` attribute.
- Pagefind builds a static search index at `dist/pagefind/`; `/search/`
  embeds the default Pagefind UI.
- RSS at `/rss.xml`, sitemap at `/sitemap-index.xml`.

## Audio narration

Every drafted section has a "Listen to this section" player at the top of
its page. Audio is generated locally (no cloud TTS) by
[Kokoro 82M](https://github.com/hexgrad/kokoro), encoded as 64 kbps mono
MP3, and committed under `public/audio/chapter-{N}/section-{N_N}.mp3`.

The Astro UI prefers the pre-generated MP3 and silently falls back to the
browser's Web Speech API (`SpeechSynthesisUtterance`) when the file is
missing, fails to load, or the section was never narrated.

Generation is **not** wired into `npm run build` — it's slow (multiple
minutes per section on CPU, ~30 s on Apple Silicon MPS). Run it manually:

```bash
npm run build:audio                  # incremental: only changed sections
npm run build:audio:force            # regenerate every section
npm run build:audio:section -- 1.1   # regenerate one section
```

These wrap `bash scripts/audio/run.sh`, which creates a dedicated venv at
`scripts/audio/.venv/` on first run, installs the pinned dependencies,
syncs the book mirror, then invokes `scripts/audio/generate.py`. Build-
time caching by content hash skips sections whose preprocessed text
hasn't changed.

### Changing the voice

Default voice: `af_heart` (American female, warm, well-paced for long-
form prose). Override per run with `--voice am_michael`, or persist by
editing `DEFAULT_VOICE` at the top of `scripts/audio/generate.py`. The
manifest records the voice each MP3 was generated with, so swapping
voices invalidates the cache and rebuilds the affected sections on the
next run.

Voice list and samples: <https://huggingface.co/hexgrad/Kokoro-82M>.

### Extending the pronunciation table

`scripts/audio/preprocess.py` opens with a `PRONUNCIATION_RULES` list
of `(regex, replacement)` pairs applied in order. Add a row whenever a
new term mispronounces. Re-run `npm run build:audio:section -- N.M` to
verify the fix on one section before pushing the rest.

### Disclosure

The audio is AI-narrated. The `AudioPlayer` component (`src/components/
AudioPlayer.astro`) shows a small italic "AI-narrated by Kokoro" line
next to the play controls. **Do not remove that disclosure** when
restyling the player — it's the only place we tell readers the audio
isn't human-recorded.

### Known limitations

- Code blocks are not narrated; Kokoro replaces them with the stage cue
  "Code block omitted from audio."
- Display math (`$$ … $$`) is replaced with "Equation omitted from
  audio."; inline math is best-effort rendered to spoken text via a
  small LaTeX→words table, otherwise dropped.
- Very long sections take ~10× wall time on CPU vs Apple Silicon MPS.
- The model runs at 24 kHz; re-encoding higher does not buy quality.

Deeper technical notes live at
[`scripts/audio/README.md`](./scripts/audio/README.md).

## Deploy

See [`DEPLOY.md`](./DEPLOY.md).
