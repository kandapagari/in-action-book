// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import pagefind from 'astro-pagefind';
import vercel from '@astrojs/vercel';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkStripH1 from './src/lib/remark-strip-h1.mjs';

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
    sitemap(),
    pagefind(),
  ],
  markdown: {
    remarkPlugins: [remarkStripH1, remarkMath],
    rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false }]],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      defaultColor: false,
      wrap: true,
    },
  },
});
