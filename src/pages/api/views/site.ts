import type { APIRoute } from 'astro';
import { SITE_KEY, isBot, readCount, incrementCount, jsonResponse } from '../../../lib/views';

// Server-rendered on Vercel as a serverless function. The rest of the site
// stays statically prerendered.
export const prerender = false;

export const GET: APIRoute = async () => {
  return jsonResponse(await readCount(SITE_KEY));
};

export const POST: APIRoute = async ({ request }) => {
  if (isBot(request.headers.get('user-agent'))) {
    const current = await readCount(SITE_KEY);
    if (current.count === null) return jsonResponse(current);
    return jsonResponse({ count: current.count, skipped: true });
  }
  return jsonResponse(await incrementCount(SITE_KEY));
};

// HEAD must never increment and must not return a body.
export const HEAD: APIRoute = async () => new Response(null, { status: 204 });
