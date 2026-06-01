// Shared helpers for the self-hosted view counter.
//
// All state lives in Upstash Redis as plain integer counters. We store ONLY
// integer counts — no IP, no UA, no timestamps, no per-visit data. The
// keyspace is small:
//
//   views:site                       — total page loads across the site
//   views:section:{chapter}:{section} — per-section page loads
//
// `chapter` and `section` are the URL parameters used by the chapter route
// `/chapters/{chapter}/{section}/` (e.g. chapter="1", section="1" → "1.1";
// chapter="2", section="x" → "2.x"). See `sectionRouteParams` in
// `progress.ts` for how full section IDs are split into URL params.
//
// Credentials are the Upstash REST URL + REST token, auto-injected by the
// Vercel → Upstash marketplace integration. The variable names depend on the
// integration: it injects the KV-style `KV_REST_API_URL` / `KV_REST_API_TOKEN`
// pair, while a bare Upstash setup may use `UPSTASH_REDIS_REST_URL` /
// `UPSTASH_REDIS_REST_TOKEN`. We accept either. (The `KV_URL` / `REDIS_URL`
// vars are TCP connection strings — not used by the REST client.)
//
// The client is created lazily so a missing-credentials environment (local
// dev) degrades gracefully to `{ count: null, error: "kv_unavailable" }`
// instead of throwing at import time.

import { Redis } from '@upstash/redis';

let _redis: Redis | null = null;

function getRedis(): Redis {
  if (_redis) return _redis;
  const url = process.env.KV_REST_API_URL ?? process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.KV_REST_API_TOKEN ?? process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) {
    // Caught by `withKv` → surfaces as kv_unavailable, never reaches the page.
    throw new Error('upstash_env_missing');
  }
  _redis = new Redis({ url, token });
  return _redis;
}

export const SITE_KEY = 'views:site';

export function sectionKey(chapter: string, section: string): string {
  return `views:section:${chapter}:${section}`;
}

export type ViewCountOk = { count: number; skipped?: boolean };
export type ViewCountUnavailable = { count: null; error: 'kv_unavailable' };
export type ViewCountResponse = ViewCountOk | ViewCountUnavailable;

const UNAVAILABLE: ViewCountUnavailable = { count: null, error: 'kv_unavailable' };

// Crawlers/social previewers we explicitly skip on POST. Case-insensitive
// substring match against the request's User-Agent header. The list is
// intentionally narrow — we accept some bot traffic in the count rather
// than maintaining an exhaustive regex.
const BOT_SUBSTRINGS = [
  'bot',
  'crawler',
  'spider',
  'googlebot',
  'bingbot',
  'slurp',
  'duckduckbot',
  'yandexbot',
  'facebookexternalhit',
  'linkedinbot',
  'twitterbot',
];

export function isBot(userAgent: string | null | undefined): boolean {
  if (!userAgent) return false;
  const lower = userAgent.toLowerCase();
  return BOT_SUBSTRINGS.some((needle) => lower.includes(needle));
}

const STANDARD_HEADERS: HeadersInit = {
  'Content-Type': 'application/json',
  'Cache-Control': 'no-store',
};

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: STANDARD_HEADERS });
}

// Wrap a Redis operation in a discriminated result. We distinguish:
//   { ok: true, value }   — call succeeded; `value` is the Redis return
//   { ok: false }         — client missing env vars or threw at runtime
// A successful `get` of a missing key returns `{ ok: true, value: null }`
// (Upstash returns null for unknown keys), which the callers translate to a
// count of 0. We never let a Redis failure propagate to the request handler —
// the page must keep rendering when the counter store is down.
type KvResult<T> = { ok: true; value: T } | { ok: false };

async function withKv<T>(op: (redis: Redis) => Promise<T>): Promise<KvResult<T>> {
  try {
    const redis = getRedis();
    return { ok: true, value: await op(redis) };
  } catch {
    return { ok: false };
  }
}

export async function readCount(key: string): Promise<ViewCountResponse> {
  const result = await withKv((redis) => redis.get<number>(key));
  if (!result.ok) return UNAVAILABLE;
  return { count: typeof result.value === 'number' ? result.value : 0 };
}

export async function incrementCount(key: string): Promise<ViewCountResponse> {
  const result = await withKv((redis) => redis.incr(key));
  if (!result.ok) return UNAVAILABLE;
  return { count: result.value };
}
