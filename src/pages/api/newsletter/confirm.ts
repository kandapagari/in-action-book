import type { APIRoute } from 'astro';
import { confirmSubscription, InvalidTokenError } from '../../../lib/newsletter';
import { redirect, expiredLinkResponse, serverError } from './_shared';

export const prerender = false;

export const GET: APIRoute = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) return expiredLinkResponse();
  try {
    await confirmSubscription(token);
    return redirect('/newsletter/confirmed/');
  } catch (err) {
    if (err instanceof InvalidTokenError) return expiredLinkResponse();
    console.error('[newsletter] confirm failed:', err);
    return serverError();
  }
};
