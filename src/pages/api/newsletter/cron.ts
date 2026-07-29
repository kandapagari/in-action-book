import type { APIRoute } from 'astro';
import { jsonResponse } from '../../../lib/views';
import { runCron } from '../../../lib/newsletter';

export const prerender = false;

// Invoked weekly by Vercel Cron (see vercel.json). Vercel sends the
// `Authorization: Bearer ${CRON_SECRET}` header automatically when CRON_SECRET
// is set in the project environment.
export const GET: APIRoute = async ({ request }) => {
  const secret = process.env.CRON_SECRET;
  if (!secret) return jsonResponse({ error: 'cron_not_configured' }, 500);
  if (request.headers.get('authorization') !== `Bearer ${secret}`) {
    return jsonResponse({ error: 'unauthorized' }, 401);
  }
  try {
    return jsonResponse(await runCron());
  } catch {
    return jsonResponse({ error: 'cron_failed' }, 500);
  }
};
