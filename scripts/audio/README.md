# `site/scripts/audio/` — narration toolchain

Build-time text-to-speech pipeline that produces per-section MP3s the Astro
site serves alongside the markdown. Engine: **Kokoro 82M** (open-source,
local, no API key). Encoder: ffmpeg via pydub. Storage: committed MP3s
under `site/public/audio/`. Fallback at runtime: the browser's Web Speech
API.

Run it from `site/` with:

```bash
npm run build:audio                  # full incremental pass
npm run build:audio:force            # regenerate everything
npm run build:audio:section -- 1.1   # regenerate one section
```

These wrap `bash scripts/audio/run.sh`, which manages the venv and the
book-mirror sync before invoking `generate.py`.

## Why not in `npm run build`?

Generating audio is slow (multiple minutes per section on CPU, ~30 s on
Apple Silicon MPS). The site build runs in seconds. The author runs
`npm run build:audio` manually when they want the audio refreshed; the
MP3s sit in git and ride along with the rest of the deploy.

## Layout

```
site/scripts/audio/
  README.md            ← this file
  requirements.txt     ← pinned deps
  generate.py          ← entrypoint
  preprocess.py        ← markdown → spoken prose + pronunciation table
  manifest.py          ← cache manifest schema + helpers
  run.sh               ← venv bootstrap + invoker
  .venv/               ← created on first run (NOT committed)
  .audio-manifest.json ← per-section cache, content-hash keyed (NOT committed)

site/public/audio/
  manifest.json        ← what the Astro UI reads at build time (committed)
  chapter-1/section-1_1.mp3
  chapter-1/section-1_2.mp3
  …
```

## Venv

We isolate everything in `site/scripts/audio/.venv/`. Kokoro 0.9.4's
pypi wheel requires Python `>=3.10, <3.13` (the GitHub `pyproject.toml`
says `<3.14`, but the published wheel disagrees and pypi wins). On this
machine the toolchain pins Python **3.12.11** (`uv`-managed). `run.sh`
auto-discovers a compatible interpreter; override with the `PYTHON_BIN`
env var.

The venv is rebuilt automatically when `requirements.txt` changes (we
hash the file and stamp it under the venv as
`.requirements.sha`).

## Voice

Default: `af_heart` — American female, warm, well-paced for long-form
prose. Set explicitly because audio quality is the whole point and a
silent fallback would be worse than wrong.

Alternative voices worth trying on a one-section spot check before
committing to a swap:

* `am_michael` — American male, measured.
* `af_bella` — American female, brighter.
* `bf_emma` — British female, slightly more formal.

Pass with `--voice <id>` (and persist the change by editing
`DEFAULT_VOICE` in `generate.py`). The per-section manifest records the
voice the MP3 was generated with; a voice change invalidates the cache
for every section.

The Kokoro voice list and audio samples live at
<https://huggingface.co/hexgrad/Kokoro-82M>.

## Cache invalidation

A section regenerates iff any of these are true:

1. `--force` was passed.
2. `--section <id>` was passed and this section matches.
3. No prior entry for the section in `.audio-manifest.json`.
4. The SHA-256 of the preprocessed spoken text changed.
5. The voice ID in the manifest entry no longer matches the requested
   voice.
6. The MP3 file referenced in the manifest is missing on disk.

Otherwise we skip and move on.

The manifest is rewritten after every section finishes — interrupting
with Ctrl-C does not lose progress.

## Manifest schema

`site/public/audio/manifest.json` (consumed by the Astro UI) and
`site/scripts/audio/.audio-manifest.json` (build cache) share one
schema:

```json
{
  "schema_version": 1,
  "voice_default": "af_heart",
  "generated_at": "2026-05-28T16:42:00+00:00",
  "entries": {
    "1.1": {
      "section": "1.1",
      "chapter": 1,
      "title": "Why \"action\" is the hard part of robotics",
      "file": "/audio/chapter-1/section-1_1.mp3",
      "path": "/abs/path/on/build/machine.mp3",
      "content_hash": "abc123…",
      "duration_seconds": 742.5,
      "file_size_bytes": 5_940_000,
      "voice": "af_heart",
      "generated_at": "2026-05-28T16:42:00+00:00",
      "char_count": 11_200,
      "word_count": 2_060
    }
  }
}
```

The `path` field is build-machine-specific and is included only as a
debugging convenience — the Astro UI reads `file` (a relative URL).

`schema_version` bumps when the entry shape changes incompatibly, which
causes the loader to discard the on-disk file and treat every section as
uncached on the next run.

## Pronunciation table

`preprocess.py` ships an editable table near the top:

```python
PRONUNCIATION_RULES: list[tuple[str, str]] = [
    (r"π0|π₀|\bpi0\b", "pi zero"),
    (r"π", "pi"),
    (r"\barXiv\b", "archive"),
    ...
]
```

Each entry is `(regex, replacement)`. They are applied in order with
`re.UNICODE`, so more specific patterns must come first. Replacement may
be a string or a `re.sub` callable.

Add a rule whenever a term mispronounces in the generated audio. Don't
worry about exhaustive coverage — the audio is the test bench. Edit,
regenerate the offending section, listen, repeat.

## Math handling

Display math (`$$ … $$`) is always replaced with `(Equation omitted from
audio.)`. Inline math (`$ … $`) is best-effort rendered to spoken words
via a small token table at the bottom of `preprocess.py`; anything
unsupported is replaced with `(equation omitted)`. We err on the side of
silence over mispronunciation.

## Debugging

* **The script can't find Kokoro:** activate the venv manually
  (`source .venv/bin/activate`) and run `pip list`. If `kokoro` is
  missing the venv was created against an incompatible Python.
* **MPS device errors on Apple Silicon:** add `--device cpu` to fall
  back. CPU is ~5–10× slower but always works.
* **Audio is silent / very short:** Kokoro silently produces zero
  samples when the text is empty. Check the preprocessor output:
  ```python
  from preprocess import preprocess_markdown
  print(preprocess_markdown(open("site/src/content/book/chapter_01/section_1_1.md").read().split("---", 2)[2]).text[:500])
  ```
* **MP3 missing after a run:** the manifest is the source of truth. If
  the file was deleted on disk but the manifest still references it,
  the next pass will detect the missing file and regenerate.

## Limitations

* Code blocks are not narrated — replaced with "Code block omitted from
  audio." Reading Python aloud was not enjoyable to listen to in
  testing.
* LaTeX with macro-heavy display equations is dropped. We render the
  inline math we can confidently speak; everything else is dropped with
  a brief stage cue.
* Sectional drops in voice timbre on chunk boundaries: Kokoro is run
  per-1500-char chunk and joined; the seams are audible on close
  listening but inoffensive at 1× speed.
* The model itself runs at 24 kHz mono. Re-encoding to higher rates
  would not improve perceptual quality.
