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

## Enabling the player in production

The Astro `<AudioPlayer>` component is gated behind a build-time feature
flag so the toolchain can be staged without exposing a half-built player
to readers. The flag:

```
PUBLIC_AUDIO_ENABLED=true
```

Off by default — when unset (or any value other than `true`), every
section page renders **without** the audio block. No play button, no
browser-TTS fallback, nothing. The page looks like it did before the
audio feature was added.

The `PUBLIC_` prefix is mandatory: Astro/Vite only exposes
`PUBLIC_*` env vars to the build. We read the flag inside
`src/components/AudioPlayer.astro`, and Astro skips rendering the
component (markup and inline script alike) when the flag is off.

### Local development

Create `site/.env.local` (already ignored by git):

```bash
echo 'PUBLIC_AUDIO_ENABLED=true' >> site/.env.local
```

Restart `npm run dev` or rebuild for the change to take effect.
`.env.example` (committed) is the canonical reference.

### Vercel (production)

Settings → Environment Variables → add `PUBLIC_AUDIO_ENABLED=true` for
the Production and Preview scopes, then trigger a redeploy. The flag is
read at build time, not runtime — toggling it without a new build has no
effect.

### What "off" means

Verified with `curl … | grep -i 'audio-player\|listen'`:

| State                     | `chapters/1/1/` markup               | `chapters/4/1/` markup                                   | JS payload         |
|---------------------------|--------------------------------------|----------------------------------------------------------|--------------------|
| flag unset / `false`      | no `audio-player`, no `Listen`       | no `audio-player`, no `Listen`                           | nothing extra      |
| flag `true`, §1.1 has MP3 | real player, `AI-narrated by Kokoro` | fallback player, `Listen (browser TTS — quality varies)` | one bundled module |

## Running on a remote GPU machine

Kokoro inference on an Apple-Silicon MPS device takes ~0.5×–0.7× real
time. On an NVIDIA CUDA GPU it's substantially faster — a full 22-section
batch that takes ~2.5 hours on M-series MPS finishes in roughly 25–45
minutes on an A100 / RTX 4090 / RTX 3090. If you have CUDA hardware on
another box, this is the recommended way to run the batch.

### System requirements

| Component     | Version                                                                            |
|---------------|------------------------------------------------------------------------------------|
| OS            | Ubuntu 22.04 / 24.04 LTS (or any modern Linux with a recent libc)                  |
| Python        | 3.10–3.12 (Kokoro 0.9.4's pypi wheel pins `<3.13`)                                 |
| CUDA          | 12.1 or newer (matched against the PyTorch wheel)                                  |
| NVIDIA driver | Whatever the chosen CUDA version requires                                          |
| ffmpeg        | Any recent build on PATH (`apt install ffmpeg`)                                    |
| espeak-ng     | System install for misaki's G2P backend (`apt install espeak-ng libespeak-ng-dev`) |

Verify in advance:

```bash
nvidia-smi          # GPU + driver visible
python3 --version   # >= 3.10, < 3.13
ffmpeg -version     # any
espeak-ng --version # any
```

### 1. Get the workspace onto the GPU machine

Only `site/` is needed — the Astro mirror at `site/src/content/book/` is
the input to the generator, and it already contains every drafted
section. From your local machine:

```bash
# Replace HOST and the destination path to taste.
rsync -av --exclude=node_modules --exclude=dist --exclude=.astro \
      --exclude='scripts/audio/.venv' --exclude='scripts/audio/logs' \
      --exclude='scripts/audio/.audio-manifest.json' \
      "/Users/pavan.kumar.kandapagari/Documents/Claude/Projects/Book on Action Models/site/" \
      HOST:/path/to/site/
```

If you'd rather clone the git repo on the GPU box: `git clone …`, then
make sure the `site/src/content/book/` mirror is up to date by running
`bash ../tools/sync-book-to-site.sh` from inside `site/` (the parent
flow disabled the automatic `prebuild` sync, so this is now a manual
step).

### 2. Set up the venv on the GPU machine

```bash
cd /path/to/site/scripts/audio
sudo apt install -y python3.12 python3.12-venv ffmpeg espeak-ng libespeak-ng-dev
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1 wheels FIRST, then the rest.
# Adjust the cu121 tag if your CUDA version differs (cu118, cu124, …).
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Now install the rest from requirements.txt.
pip install -r requirements.txt

# Sanity check: PyTorch must see the GPU.
python -c "import torch; print('cuda:', torch.cuda.is_available(),
                                'devices:', torch.cuda.device_count(),
                                'name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

If `torch.cuda.is_available()` is `False`, the CUDA wheel didn't match
the system driver. Pick the correct `cu*` index URL from
<https://pytorch.org/get-started/locally/> and reinstall torch.

### 3. Run the batch

`generate.py` already supports `--device cuda`. The default
device-selection logic picks CUDA over CPU when available, so a bare
invocation is enough:

```bash
cd /path/to/site
bash scripts/audio/run.sh                # incremental: only changed sections
bash scripts/audio/run.sh --device cuda  # force CUDA explicitly
bash scripts/audio/run.sh --force        # clean rebuild of every section
```

You can also override which interpreter the script bootstraps the venv
with:

```bash
PYTHON_BIN=/usr/bin/python3.12 bash scripts/audio/run.sh
```

The first invocation downloads the Kokoro weights (~330 MB) into
`~/.cache/huggingface/`. Subsequent invocations are offline-capable.

### Estimated wall time per section

| GPU                     | Per section (~12 min audio) | Full 22-section batch |
|-------------------------|-----------------------------|-----------------------|
| RTX 3090                | 60–90 s                     | 25–35 min             |
| RTX 4090                | 40–70 s                     | 18–28 min             |
| A100 40GB / 80GB        | 35–60 s                     | 15–25 min             |
| Apple M4 MPS (baseline) | 5–10 min                    | 2–3 h                 |

(Numbers are rough; chunked synthesis dominates, and Kokoro's
per-chunk cost is roughly linear in characters.)

### 4. Bring the audio back

The MP3s + manifest are the only outputs. Two ways:

```bash
# Option A: rsync back to your local machine.
rsync -av HOST:/path/to/site/public/audio/ \
          "/Users/pavan.kumar.kandapagari/Documents/Claude/Projects/Book on Action Models/site/public/audio/"
```

```bash
# Option B: commit + push from the GPU machine.
cd /path/to/site
git add public/audio/
git commit -m "audio: generate batch on GPU machine"
git push
```

Either way, audio files belong in the repo — `site/.gitignore` excludes
the venv and the build-cache manifest, **not** `public/audio/`. The
public manifest at `site/public/audio/manifest.json` is the single
source of truth the Astro UI reads.

### 5. Flip the flag and redeploy

Once the MP3s are in the repo, set `PUBLIC_AUDIO_ENABLED=true` in your
Vercel project (or `site/.env.local` for local testing) and rebuild.
The player will appear on every section that has a manifest entry, and
fall back to the browser-TTS button on sections that don't.

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
