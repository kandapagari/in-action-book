import type { APIRoute } from 'astro';
import {
  sectionKey,
  isBot,
  readCount,
  incrementCount,
  jsonResponse,
} from '../../../../../lib/views';

export const prerender = false;

// Per-URL-segment validation matching the routes Astro actually generates
// from `sectionRouteParams` (see `src/lib/progress.ts`):
//
//   /chapters/{chapter}/{section}/
//
// where `chapter` is the integer chapter number (1..18) and `section` is the
// part after the dot in the full section ID — either one-or-more digits
// ("1", "2", ..., "10") OR the literal "x" used for per-chapter hands-on
// sections (e.g. full ID "1.x" → chapter "1", section "x").
const SECTION_RE = /^(?:[0-9]+|x)$/;

function validate(chapter: string | undefined, section: string | undefined):
  | { ok: true; key: string }
  | { ok: false; reason: string } {
  if (!chapter || !section) {
    return { ok: false, reason: 'missing route params' };
  }
  const chapterNum = Number(chapter);
  if (!Number.isInteger(chapterNum) || chapterNum < 1 || chapterNum > 18) {
    return { ok: false, reason: 'invalid chapter' };
  }
  if (!SECTION_RE.test(section)) {
    return { ok: false, reason: 'invalid section' };
  }
  return { ok: true, key: sectionKey(chapter, section) };
}

export const GET: APIRoute = async ({ params }) => {
  const v = validate(params.chapter, params.section);
  if (!v.ok) return jsonResponse({ error: v.reason }, 400);
  return jsonResponse(await readCount(v.key));
};

export const POST: APIRoute = async ({ params, request }) => {
  const v = validate(params.chapter, params.section);
  if (!v.ok) return jsonResponse({ error: v.reason }, 400);

  if (isBot(request.headers.get('user-agent'))) {
    const current = await readCount(v.key);
    if (current.count === null) return jsonResponse(current);
    return jsonResponse({ count: current.count, skipped: true });
  }
  return jsonResponse(await incrementCount(v.key));
};

export const HEAD: APIRoute = async () => new Response(null, { status: 204 });
