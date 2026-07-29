import type { APIRoute } from 'astro';
import { jsonResponse } from '../../../lib/views';
import { requestConfirmation, InvalidEmailError } from '../../../lib/newsletter';

export const prerender = false;

export const POST: APIRoute = async ({ request }) => {
  let email: unknown;
  try {
    const body = (await request.json()) as { email?: unknown };
    email = body.email;
  } catch {
    return jsonResponse({ error: 'invalid_json' }, 400);
  }
  if (typeof email !== 'string' || email.trim() === '') {
    return jsonResponse({ error: 'email_required' }, 400);
  }

  try {
    return jsonResponse(await requestConfirmation(email));
  } catch (err) {
    if (err instanceof InvalidEmailError) {
      return jsonResponse({ error: 'invalid_email' }, 400);
    }
    // Missing Redis/Resend config or a delivery failure — a real 500, never a
    // faked success. Log the true cause (never secrets) for Vercel runtime logs.
    console.error('[newsletter] subscribe failed:', err);
    return jsonResponse({ error: 'subscribe_failed' }, 500);
  }
};
