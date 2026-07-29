// Inlines the book's section + appendix markdown into the build (serverless
// functions can't read arbitrary disk), parses frontmatter with gray-matter,
// and exposes lookup + a pure full-text search ranker. Mirrors the
// import.meta.glob inlining pattern in newsletter-content.ts / progress.ts.
//
// The glob call is wrapped in try/catch so the *pure* helper `rankMatches`
// still loads under plain Node (`node --test`), where import.meta.glob is
// undefined and the doc list is simply empty.
//
// Relative type imports carry explicit `.ts` extensions for Node's native
// TypeScript loader; Astro/Vite and the project tsconfig accept them too.

import matter from 'gray-matter';
import { SITE } from './seo.ts';
import { normalizeFrontmatter } from './frontmatter.ts';

// Route-param split, matching sectionRouteParams() in progress.ts. Inlined
// (rather than imported) because progress.ts statically imports PROGRESS.md?raw
// at module load, which crashes under plain Node (`node --test`) — the same
// reason newsletter-content.ts only type-imports progress.ts. Keeping this
// module free of that import is what lets the test exercise rankMatches.
function routeParams(id: string): { chapter: string; rest: string } {
  const [chapter, ...rest] = id.split('.');
  return { chapter, rest: rest.join('.') };
}

export type BookDocKind = 'section' | 'appendix';

export type BookDoc = {
  /** Section id ("1.1", "1.x", "4.1") or appendix letter ("A"). */
  id: string;
  kind: BookDocKind;
  /** Chapter number for sections; undefined for appendices. */
  chapter?: number;
  title: string;
  /** Frontmatter status (e.g. "draft"). */
  status: string;
  prereqs: string;
  key_refs: string[];
  body: string;
  /** Absolute public URL of the section/appendix on the site. */
  url: string;
};

export type SearchResult = {
  id: string;
  title: string;
  url: string;
  excerpt: string;
};

type RawMap = Record<string, string>;

let rawMap: RawMap = {};
try {
  // Vite statically replaces this with an inlined object of raw file contents
  // at build time. Under plain Node the call throws / is undefined → {}.
  rawMap = import.meta.glob('../content/book/**/*.md', {
    query: '?raw',
    import: 'default',
    eager: true,
  }) as RawMap;
} catch {
  rawMap = {};
}

function sectionUrl(id: string): string {
  const { chapter, rest } = routeParams(id);
  return `${SITE}/chapters/${chapter}/${rest}/`;
}

function appendixUrl(letter: string): string {
  return `${SITE}/appendix/${letter.toLowerCase()}/`;
}

function parseDoc(path: string, raw: string): BookDoc | null {
  // PROGRESS.md lives under the same tree but is not a section/appendix.
  if (path.endsWith('/PROGRESS.md')) return null;

  // Same free-text-colon normalization the site's book-loader applies, so
  // gray-matter/js-yaml accepts titles like "Three loss families: …".
  const parsed = matter(normalizeFrontmatter(raw));
  const data = parsed.data as Record<string, unknown>;
  const body = parsed.content.trim();
  const title = typeof data.title === 'string' ? data.title : '';
  const status = typeof data.status === 'string' ? data.status : '';
  const prereqs = typeof data.prereqs === 'string' ? data.prereqs : '';
  const key_refs = Array.isArray(data.key_refs) ? data.key_refs.map(String) : [];

  if (data.appendix != null) {
    const id = String(data.appendix);
    return { id, kind: 'appendix', title, status, prereqs, key_refs, body, url: appendixUrl(id) };
  }
  if (data.section != null) {
    const id = String(data.section);
    const chapter = Number(data.chapter);
    return {
      id,
      kind: 'section',
      chapter: Number.isInteger(chapter) ? chapter : undefined,
      title,
      status,
      prereqs,
      key_refs,
      body,
      url: sectionUrl(id),
    };
  }
  return null;
}

function buildDocs(raws: RawMap): BookDoc[] {
  const docs: BookDoc[] = [];
  for (const [path, raw] of Object.entries(raws)) {
    const doc = parseDoc(path, raw);
    if (doc) docs.push(doc);
  }
  return docs;
}

const docs: BookDoc[] = buildDocs(rawMap);

const byId: Map<string, BookDoc> = new Map(docs.map((d) => [d.id, d]));

export function allDocs(): BookDoc[] {
  return docs;
}

// Section ids match exactly; appendix letters match case-insensitively so
// both "A" and "a" resolve.
export function getById(id: string): BookDoc | undefined {
  return byId.get(id) ?? byId.get(id.toUpperCase());
}

// Numeric section ordering within a chapter, with the "N.x" exercise section
// sorted last.
function sectionSortKey(id: string): number {
  const rest = id.split('.')[1] ?? '';
  return rest === 'x' ? Number.POSITIVE_INFINITY : Number(rest);
}

export function getByChapter(chapter: number): BookDoc[] {
  return docs
    .filter((d) => d.kind === 'section' && d.chapter === chapter)
    .sort((a, b) => sectionSortKey(a.id) - sectionSortKey(b.id));
}

// ---------------------------------------------------------------------------
// Search (pure — no glob dependency, so the test exercises it with fabricated
// docs)
// ---------------------------------------------------------------------------

const TITLE_WEIGHT = 5;

function countOccurrences(haystack: string, needle: string): number {
  if (!needle) return 0;
  let count = 0;
  let from = 0;
  for (;;) {
    const idx = haystack.indexOf(needle, from);
    if (idx < 0) break;
    count += 1;
    from = idx + needle.length;
  }
  return count;
}

function excerptAround(body: string, term: string, radius = 120): string {
  const idx = term ? body.toLowerCase().indexOf(term) : -1;
  const collapse = (s: string) => s.replace(/\s+/g, ' ').trim();
  if (idx < 0) return collapse(body.slice(0, radius * 2)) + (body.length > radius * 2 ? '…' : '');
  const start = Math.max(0, idx - radius);
  const end = Math.min(body.length, idx + term.length + radius);
  let ex = collapse(body.slice(start, end));
  if (start > 0) ex = '…' + ex;
  if (end < body.length) ex = ex + '…';
  return ex;
}

// Case-insensitive term-hit ranking over title + body. Title hits are weighted
// higher. Returns id/title/url plus an excerpt around the first body match.
// ponytail: naive O(n·terms) scan over ~107 small markdown files — fine at
// this size; upgrade path is a prebuilt index if the corpus grows large.
export function rankMatches(docs: BookDoc[], query: string, limit = 5): SearchResult[] {
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 0);
  if (terms.length === 0) return [];

  const scored: { doc: BookDoc; score: number }[] = [];
  for (const doc of docs) {
    const titleLower = doc.title.toLowerCase();
    const bodyLower = doc.body.toLowerCase();
    let score = 0;
    for (const term of terms) {
      score += countOccurrences(titleLower, term) * TITLE_WEIGHT;
      score += countOccurrences(bodyLower, term);
    }
    if (score > 0) scored.push({ doc, score });
  }

  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, Math.max(0, limit)).map(({ doc }) => {
    const bodyLower = doc.body.toLowerCase();
    const firstTerm = terms.find((t) => bodyLower.includes(t)) ?? terms[0];
    return {
      id: doc.id,
      title: doc.title,
      url: doc.url,
      excerpt: excerptAround(doc.body, firstTerm),
    };
  });
}
