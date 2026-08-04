// Builds the ebook downloads (PDF + EPUB) from the book's markdown.
//
//   node scripts/build-ebook.mjs
//
// Inputs : src/content/book/PROGRESS.md (chapter/appendix titles) and the
//          section/appendix markdown under src/content/book/.
// Outputs: public/downloads/action-models.{pdf,epub}
//
// Section ordering mirrors sectionSortKey in src/lib/book-content.ts and the
// title parsing mirrors src/lib/progress.ts — both are reimplemented here
// because those modules use Vite-only imports (import.meta.glob, ?raw) that
// plain Node cannot load. The pure helpers are exported for the node --test
// self-check; the pandoc pipeline only runs when executed directly.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const BOOK_DIR = path.join(ROOT, 'src/content/book');
const OUT_DIR = path.join(ROOT, 'public/downloads');

const TITLE = 'Action Models for Robot Learning';
const AUTHOR = 'Pavan Kumar Kandapagari';

// ---- Pure helpers (unit-tested) --------------------------------------------

// Mirrors sectionSortKey in src/lib/book-content.ts: numeric ordering within
// a chapter, with the "N.x" exercise section sorted last.
export function sectionSortKey(id) {
  const rest = id.split('.')[1] ?? '';
  return rest === 'x' ? Number.POSITIVE_INFINITY : Number(rest);
}

export function sortSectionIds(ids) {
  return [...ids].sort((a, b) => sectionSortKey(a) - sectionSortKey(b));
}

// Chapter/appendix titles out of PROGRESS.md — the same two regexes
// src/lib/progress.ts uses (chapterHeading, appendixLine).
export function parseProgressTitles(raw) {
  const chapterHeading = /^####\s+Chapter\s+(\d+)\.\s+(.+)$/;
  const appendixLine = /^-\s+\[[ x~r]\]\s+([A-F])\.\s+(.+)$/;
  const chapters = new Map();
  const appendices = new Map();
  for (const line of raw.split(/\r?\n/)) {
    const ch = chapterHeading.exec(line);
    if (ch) {
      chapters.set(Number(ch[1]), ch[2].trim());
      continue;
    }
    const ap = appendixLine.exec(line);
    if (ap) appendices.set(ap[1], ap[2].trim());
  }
  return { chapters, appendices };
}

// The ebook discards frontmatter data entirely (titles come from
// PROGRESS.md), so a plain regex strip suffices — gray-matter's YAML parse
// would additionally need the src/lib/frontmatter.ts normalization for
// colon-in-title lines.
export function stripFrontmatter(raw) {
  return raw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '');
}

// Shift every markdown heading up one level so section H1s sit under the
// injected chapter H1s. Headings inside ``` code fences are left untouched.
// ponytail: fences are tracked at column 0 only, which is how the book's
// markdown writes them; indented fences would not be recognized.
export function shiftHeadings(md) {
  let fence = 0; // backtick-run length of the open fence; 0 = outside
  return md
    .split('\n')
    .map((line) => {
      const m = /^(`{3,})/.exec(line);
      if (m) {
        if (fence === 0) fence = m[1].length;
        else if (m[1].length >= fence) fence = 0;
        return line;
      }
      return fence === 0 && line.startsWith('#') ? '#' + line : line;
    })
    .join('\n');
}

// units: [{ heading, bodies: [markdown, ...] }] -> one markdown document.
export function assembleBook(units) {
  return units.map((u) => [u.heading, ...u.bodies].join('\n\n')).join('\n\n') + '\n';
}

// ---- Pipeline ---------------------------------------------------------------

function requireBinary(name, installHint) {
  try {
    return execSync(`command -v ${name}`, { encoding: 'utf8' }).trim();
  } catch {
    // The no-sudo pandoc install drops the binary in ~/.local/bin, which is
    // not on PATH in every shell — check there explicitly before failing.
    const local = path.join(os.homedir(), '.local', 'bin', name);
    if (fs.existsSync(local)) return local;
    throw new Error(`'${name}' not found on PATH. ${installHint}`);
  }
}

function sectionBodies(chapterDir, chapterNum) {
  const files = fs.readdirSync(chapterDir);
  const byId = new Map();
  for (const f of files) {
    const m = /^section_\d+_(\w+)\.md$/.exec(f);
    if (m) byId.set(`${chapterNum}.${m[1]}`, f);
  }
  return sortSectionIds([...byId.keys()]).map((id) => {
    const raw = fs.readFileSync(path.join(chapterDir, byId.get(id)), 'utf8');
    return shiftHeadings(stripFrontmatter(raw).trim());
  });
}

function buildUnits() {
  const progress = fs.readFileSync(path.join(BOOK_DIR, 'PROGRESS.md'), 'utf8');
  const { chapters, appendices } = parseProgressTitles(progress);

  const chapterDirs = fs
    .readdirSync(BOOK_DIR)
    .map((entry) => ({ entry, m: /^chapter_(\d+)$/.exec(entry) }))
    .filter(({ entry, m }) => m && fs.statSync(path.join(BOOK_DIR, entry)).isDirectory())
    .sort((a, b) => Number(a.m[1]) - Number(b.m[1]));

  const units = [];
  for (const { entry, m } of chapterDirs) {
    const n = Number(m[1]);
    const title = chapters.get(n);
    if (!title) throw new Error(`No title for Chapter ${n} in PROGRESS.md`);
    units.push({
      heading: `# Chapter ${n} — ${title}`,
      bodies: sectionBodies(path.join(BOOK_DIR, entry), n),
    });
  }

  const appendixFiles = fs
    .readdirSync(BOOK_DIR)
    .map((f) => /^appendix_([A-F])\.md$/.exec(f))
    .filter(Boolean)
    .sort((a, b) => a[1].localeCompare(b[1]));
  for (const m of appendixFiles) {
    const letter = m[1];
    const title = appendices.get(letter);
    if (!title) throw new Error(`No title for Appendix ${letter} in PROGRESS.md`);
    const raw = fs.readFileSync(path.join(BOOK_DIR, m[0]), 'utf8');
    units.push({
      heading: `# Appendix ${letter} — ${title}`,
      bodies: [shiftHeadings(stripFrontmatter(raw).trim())],
    });
  }

  return units;
}

function main() {
  const pandoc = requireBinary(
    'pandoc',
    'Install with: sudo apt install pandoc — or download a release tarball from https://github.com/jgm/pandoc/releases and put the binary in ~/.local/bin.',
  );
  requireBinary('xelatex', 'Install with: sudo apt install texlive-xetex.');

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const tmp = path.join(os.tmpdir(), `action-models-book-${process.pid}.md`);
  fs.writeFileSync(tmp, assembleBook(buildUnits()));

  const meta = ['--metadata', `title=${TITLE}`, '--metadata', `author=${AUTHOR}`];
  try {
    execFileSync(
      pandoc,
      [
        tmp,
        '-o',
        path.join(OUT_DIR, 'action-models.pdf'),
        '--pdf-engine=xelatex',
        '--toc',
        '--toc-depth=2',
        ...meta,
        '--variable',
        'documentclass=report',
        '--variable',
        'geometry:margin=1in',
        '--variable',
        'colorlinks',
        // Font calibration for this machine: xelatex's default Latin Modern
        // has no Greek / subscript glyphs (π₀, τ, ∈ …), so prose renders
        // blank. Linux Libertine O + DejaVu Sans Mono cover the full set.
        '--variable',
        'mainfont=Linux Libertine O',
        '--variable',
        'monofont=DejaVu Sans Mono',
      ],
      { stdio: 'inherit' },
    );
    execFileSync(
      pandoc,
      [
        tmp,
        '-o',
        path.join(OUT_DIR, 'action-models.epub'),
        '--mathml',
        '--toc',
        '--toc-depth=2',
        ...meta,
        `--epub-cover-image=${path.join(ROOT, 'public/og-image.png')}`,
      ],
      { stdio: 'inherit' },
    );
  } finally {
    fs.rmSync(tmp, { force: true });
  }

  console.log('Wrote:');
  for (const f of ['action-models.pdf', 'action-models.epub']) {
    const stat = fs.statSync(path.join(OUT_DIR, f));
    console.log(`  public/downloads/${f.padEnd(22)} ${stat.size.toString().padStart(9)} bytes`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) main();
