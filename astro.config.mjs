// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import pagefind from 'astro-pagefind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkStripH1 from './src/lib/remark-strip-h1.mjs';

const SITE_URL = process.env.SITE_URL ?? 'https://action-models-book.vercel.app';

export default defineConfig({
  site: SITE_URL,
  output: 'static',
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
