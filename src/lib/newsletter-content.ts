// Pure newsletter drip logic + excerpt-artifact loading. No Redis, no email
// I/O. Chapter teasers are pre-authored artifacts stored at
// src/content/newsletter/chapter-<N>.md (frontmatter `chapter` + `subject`,
// then 150–250 words of prose). We inline them at build time with Vite's
// import.meta.glob — mirroring how progress.ts inlines PROGRESS.md — and parse
// with gray-matter.
//
// The drip logic itself (readyChapters / sendableChapterNumbers /
// nextChapterForCursor) is pure and array-driven so it runs directly under
// `node --test`. The import.meta.glob call is wrapped in try/catch so the same
// module also loads under plain Node (where import.meta.glob is undefined):
// there the excerpt map is simply empty, which is also the correct behavior
// when the excerpt directory doesn't exist yet.
//
// Relative type imports carry explicit `.ts` extensions for Node's native
// TypeScript loader; Astro/Vite and the project tsconfig accept them too.

import type { Progress, ChapterEntry } from './progress.ts';
import matter from 'gray-matter';

// ---------------------------------------------------------------------------
// Excerpt artifacts
// ---------------------------------------------------------------------------

export type NewsletterExcerpt = { chapter: number; subject: string; body: string };

type RawMap = Record<string, string>;

let excerptRaws: RawMap = {};
try {
  // Vite statically replaces this call with an inlined object of raw file
  // contents at build time. An empty/absent directory yields {}.
  excerptRaws = import.meta.glob('../content/newsletter/*.md', {
    query: '?raw',
    import: 'default',
    eager: true,
  }) as RawMap;
} catch {
  // Plain Node (e.g. `node --test`): import.meta.glob is undefined. No
  // excerpts are needed there — the drip logic is exercised with fabricated
  // arrays.
  excerptRaws = {};
}

const excerptMap: Map<number, NewsletterExcerpt> = buildExcerptMap(excerptRaws);

function buildExcerptMap(raws: RawMap): Map<number, NewsletterExcerpt> {
  const map = new Map<number, NewsletterExcerpt>();
  for (const raw of Object.values(raws)) {
    const parsed = matter(raw);
    const chapter = Number(parsed.data.chapter);
    if (!Number.isInteger(chapter)) continue;
    const subject = typeof parsed.data.subject === 'string' ? parsed.data.subject : '';
    map.set(chapter, { chapter, subject, body: parsed.content.trim() });
  }
  return map;
}

export function excerptForChapter(chapterNumber: number): NewsletterExcerpt | undefined {
  return excerptMap.get(chapterNumber);
}

export function excerptChapterNumbers(): number[] {
  return [...excerptMap.keys()].sort((a, b) => a - b);
}

// ---------------------------------------------------------------------------
// Drip logic (pure)
// ---------------------------------------------------------------------------

// The "complete chapter" predicate — identical to `chaptersComplete` in
// progress.ts. A chapter is complete only when it has sections and every one
// is drafted or revised.
export function isChapterComplete(chapter: ChapterEntry): boolean {
  return (
    chapter.sections.length > 0 &&
    chapter.sections.every((s) => s.status === 'drafted' || s.status === 'revised')
  );
}

// Complete chapter numbers, ascending. Appendices A–F are excluded by
// construction (they live in a separate `progress.appendices` array).
export function readyChapters(progress: Progress): number[] {
  const nums: number[] = [];
  for (const part of progress.parts) {
    for (const chapter of part.chapters) {
      if (isChapterComplete(chapter)) nums.push(chapter.chapter);
    }
  }
  return nums.sort((a, b) => a - b);
}

// A chapter is *sendable* only when it is complete AND has an excerpt
// artifact. ponytail: this is a deliberate gate, not a hidden fallback — a
// complete chapter without an excerpt file is simply skipped until its excerpt
// is authored, at which point the drip resumes for it automatically.
export function sendableChapterNumbers(readyNumbers: number[], excerptNumbers: number[]): number[] {
  const hasExcerpt = new Set(excerptNumbers);
  return readyNumbers.filter((n) => hasExcerpt.has(n)).sort((a, b) => a - b);
}

// Runtime convenience: sendable chapters for the current build's progress +
// loaded excerpt artifacts.
export function sendableChapters(progress: Progress): number[] {
  return sendableChapterNumbers(readyChapters(progress), excerptChapterNumbers());
}

// The next chapter to email a subscriber whose cursor (highest chapter number
// already sent) is `cursor`: the smallest sendable chapter strictly greater
// than the cursor, or null when they are caught up.
export function nextChapterForCursor(sendableNumbers: number[], cursor: number): number | null {
  let best: number | null = null;
  for (const n of sendableNumbers) {
    if (n > cursor && (best === null || n < best)) best = n;
  }
  return best;
}

// ---------------------------------------------------------------------------
// Email rendering
// ---------------------------------------------------------------------------

export type EmailSection = {
  /** Full section id, e.g. "1.1" or "1.x". */
  id: string;
  title: string;
  /** Absolute URL to read the section on the site. */
  url: string;
};

export type ChapterEmailInput = {
  number: number;
  title: string;
  excerpt?: NewsletterExcerpt;
};

export type BuiltEmail = { subject: string; html: string; text: string };

// Warm cream / off-black serif aesthetic, inlined for email clients. ET Book
// is not available in mail, so we fall back to a classic book serif stack.
const SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', Palatino, serif";
const PAPER = '#fffff8';
const INK = '#14110d';
const MUTED = '#5b5b58';
const RULE = '#d8d6cf';
const ACCENT = '#a02c2c';

// Builds the chapter email (subject + HTML + plaintext). Teaser comes from the
// stored excerpt artifact; the section list + read/unsubscribe links are
// generated. Every email carries a one-click unsubscribe link in the footer.
export function buildChapterEmail(
  chapter: ChapterEmailInput,
  sections: EmailSection[],
  siteUrl: string,
  unsubUrl: string,
): BuiltEmail {
  const subject = chapter.excerpt?.subject?.trim() || `New chapter — ${chapter.title}`;
  const heading = `Chapter ${chapter.number} — ${chapter.title}`;
  const readUrl = sections.length > 0 ? sections[0].url : `${siteUrl}/contents/`;
  const bodyText = chapter.excerpt?.body ?? '';

  const teaserHtml = paragraphs(bodyText)
    .map((p) => `<p style="margin:0 0 1.2em;color:${INK};">${escapeHtml(p)}</p>`)
    .join('\n    ');

  const sectionListHtml = sections
    .map(
      (s) =>
        `<li style="margin:0 0 .5em;">` +
        `<a href="${s.url}" style="color:${ACCENT};text-decoration:none;">` +
        `<span style="color:${MUTED};">§${s.id}</span>&nbsp; ${escapeHtml(s.title)}</a></li>`,
    )
    .join('\n');

  const html = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /></head>
<body style="margin:0;padding:0;background:${PAPER};">
  <div style="max-width:34rem;margin:0 auto;padding:2.5rem 1.5rem;background:${PAPER};color:${INK};font-family:${SERIF};font-size:17px;line-height:1.6;">
    <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:${MUTED};margin:0 0 1.2em;">Action Models for Robot Learning</p>
    <h1 style="font-weight:400;font-size:26px;line-height:1.25;margin:0 0 .8em;color:${INK};">${escapeHtml(heading)}</h1>
    ${teaserHtml}
    <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:${MUTED};margin:0 0 .6em;">Sections in this chapter</p>
    <ul style="list-style:none;padding:0;margin:0 0 1.6em;">
${sectionListHtml}
    </ul>
    <p style="margin:0 0 2em;"><a href="${readUrl}" style="display:inline-block;border:1px solid ${INK};padding:.55em 1.1em;color:${INK};text-decoration:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;letter-spacing:.03em;">Continue reading the full chapter on the site…</a></p>
    <hr style="border:0;border-top:1px solid ${RULE};margin:2em 0 1.2em;" />
    <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;line-height:1.5;color:${MUTED};margin:0;">
      You're receiving one complete chapter a week as the book is drafted openly.
      <br /><a href="${unsubUrl}" style="color:${MUTED};">Unsubscribe</a> · <a href="${siteUrl}/" style="color:${MUTED};">${escapeHtml(hostOf(siteUrl))}</a>
    </p>
  </div>
</body>
</html>`;

  const sectionListText = sections.map((s) => `  §${s.id}  ${s.title}\n  ${s.url}`).join('\n\n');
  const text = `${heading}

${bodyText}

Sections in this chapter:

${sectionListText}

Continue reading the full chapter on the site…
${readUrl}

—
You're receiving one complete chapter a week as the book is drafted openly.
Unsubscribe: ${unsubUrl}
${siteUrl}/
`;

  return { subject, html, text };
}

// Split short prose into paragraphs on blank lines. No markdown library — the
// excerpt body is plain prose (1–2 short paragraphs).
function paragraphs(body: string): string[] {
  return body
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\s+/g, ' ').trim())
    .filter((p) => p.length > 0);
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function hostOf(url: string): string {
  return url.replace(/^https?:\/\//, '').replace(/\/$/, '');
}
