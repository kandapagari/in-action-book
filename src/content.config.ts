import { defineCollection, z } from 'astro:content';
import { bookLoader } from './lib/book-loader';

const chapters = defineCollection({
  loader: bookLoader({
    pattern: 'chapter_*/section_*.md',
    base: './src/content/book',
  }),
  schema: z.object({
    chapter: z.number().int().positive(),
    section: z.union([z.string(), z.number()]).transform((v) => String(v)),
    title: z.string(),
    target_words: z.number().int().positive(),
    status: z.enum(['draft', 'in-progress', 'revised']),
    prereqs: z.string(),
    // YAML parses lines like `- Foo: Bar` inside a block sequence as a
    // single-key mapping. The book's key_refs are free-text strings that
    // happen to contain colons (e.g. journal subtitles). Coerce any object
    // entries back into "Key: Value" strings before schema validation runs.
    key_refs: z.preprocess((v) => {
      if (!Array.isArray(v)) return v;
      return v.map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          return Object.entries(item as Record<string, unknown>)
            .map(([k, val]) => `${k}: ${val}`)
            .join(' ');
        }
        return item;
      });
    }, z.array(z.string())),
  }),
});

export const collections = { chapters };
