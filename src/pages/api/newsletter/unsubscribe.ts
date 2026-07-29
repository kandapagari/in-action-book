import type { APIRoute } from 'astro';
import { unsubscribe, InvalidTokenError } from '../../../lib/newsletter';
import { redirect, expiredLinkResponse, serverError } from './_shared';

export const prerender = false;

export const GET: APIRoute = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) return expiredLinkResponse();
  try {
    await unsubscribe(token);
    return redirect('/newsletter/unsubscribed/');
  } catch (err) {
    if (err instanceof InvalidTokenError) return expiredLinkResponse();
    console.error('[newsletter] unsubscribe failed:', err);
    return serverError();
  }
};
