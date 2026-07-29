// Shared response helpers for the newsletter GET endpoints (confirm /
// unsubscribe). The leading underscore keeps Astro from turning this file
// into a route.

const NO_STORE = { 'Cache-Control': 'no-store' } as const;

export function redirect(path: string): Response {
  return new Response(null, { status: 302, headers: { Location: path, ...NO_STORE } });
}

// A missing/expired token is user input, not an outage — answer 410 Gone with
// a small on-brand page pointing back to the signup form.
export function expiredLinkResponse(): Response {
  const serif = "Georgia, 'Iowan Old Style', 'Palatino Linotype', Palatino, serif";
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>Link expired</title></head>
<body style="margin:0;background:#fffff8;color:#14110d;font-family:${serif};">
  <div style="max-width:34rem;margin:0 auto;padding:5rem 1.5rem;">
    <h1 style="font-weight:400;font-size:26px;">This link has expired</h1>
    <p style="line-height:1.6;">Confirmation links are valid for 48&nbsp;hours. Please <a href="/newsletter/" style="color:#a02c2c;">subscribe again</a> to receive a fresh one.</p>
  </div>
</body></html>`;
  return new Response(html, {
    status: 410,
    headers: { 'Content-Type': 'text/html; charset=utf-8', ...NO_STORE },
  });
}

// Missing Redis/Resend config or a delivery failure — surface as 500 rather
// than pretending the action worked.
export function serverError(): Response {
  return new Response('The newsletter service is temporarily unavailable. Please try again later.', {
    status: 500,
    headers: { 'Content-Type': 'text/plain; charset=utf-8', ...NO_STORE },
  });
}
