// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import pagefind from 'astro-pagefind';
import vercel from '@astrojs/vercel';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkStripH1 from './src/lib/remark-strip-h1.mjs';
import rehypeTableWrap from './src/lib/rehype-table-wrap.mjs';

const SITE_URL = process.env.SITE_URL ?? 'https://action-models-book.vercel.app';

// Astro 6 no longer has a `hybrid` output value (it was collapsed into the
// `static` + per-route opt-out model). All pages stay statically prerendered;
// only the `/api/views/*` routes opt into server execution via
// `export const prerender = false` in each endpoint file. The Vercel adapter
// is required so those API routes deploy as serverless functions.
export default defineConfig({
  site: SITE_URL,
  output: 'static',
  adapter: vercel(),
  trailingSlash: 'always',
  build: {
    format: 'directory',
  },
  integrations: [
    mdx(),
    sitemap({
      // Priority scheme: the author-entity pages rank highest, then the book
      // content, then utility pages. `lastmod` is the build time — honest for
      // a continuously-rebuilt living draft. The default changefreq is weekly.
      serialize(item) {
        const path = new URL(item.url).pathname;
        if (path === '/') {
          item.priority = 1.0;
          item.changefreq = 'daily';
        } else if (path === '/about/') {
          item.priority = 0.9;
          item.changefreq = 'monthly';
        } else if (path === '/contents/') {
          item.priority = 0.8;
          item.changefreq = 'daily';
        } else if (path.startsWith('/chapters/') || path.startsWith('/appendix/')) {
          item.priority = 0.7;
          item.changefreq = 'weekly';
        } else {
          // /search/, /cite/, /contact/
          item.priority = 0.4;
          item.changefreq = 'monthly';
        }
        item.lastmod = new Date().toISOString();
        return item;
      },
    }),
    pagefind(),
  ],
  markdown: {
    remarkPlugins: [remarkStripH1, remarkMath],
    rehypePlugins: [
      [rehypeKatex, { strict: false, throwOnError: false }],
      rehypeTableWrap,
    ],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      defaultColor: false,
      wrap: false,
    },
  },
});
