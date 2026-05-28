import type { APIRoute } from 'astro';

export const GET: APIRoute = ({ site }) => {
  if (!site) {
    throw new Error(
      'robots.txt: `site` is undefined. Set `site` in astro.config.mjs (or SITE_URL env) so sitemap URL can be absolute.',
    );
  }
  const sitemap = new URL('sitemap-index.xml', site).toString();
  const body = `User-agent: *
Allow: /

Sitemap: ${sitemap}
`;
  return new Response(body, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
};
