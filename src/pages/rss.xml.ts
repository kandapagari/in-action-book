import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';
import { dateForSection, sectionRouteParams } from '../lib/progress';

export const GET: APIRoute = async (context) => {
  const entries = await getCollection('chapters');

  const items = entries
    .map((entry) => {
      const date = dateForSection(entry.data.section);
      return { entry, date };
    })
    .filter((x): x is { entry: typeof entries[number]; date: string } => x.date !== null)
    .sort((a, b) => (a.date < b.date ? 1 : -1))
    .map(({ entry, date }) => {
      const params = sectionRouteParams(entry.data.section);
      return {
        title: `§${entry.data.section} — ${entry.data.title}`,
        link: `/chapters/${params.chapter}/${params.section}/`,
        pubDate: new Date(`${date}T00:00:00Z`),
        description: `Section ${entry.data.section} of Action Models for Robot Learning. Prereqs: ${entry.data.prereqs}. ~${entry.data.target_words.toLocaleString()} target words.`,
      };
    });

  return rss({
    title: 'Action Models for Robot Learning',
    description:
      'New sections of the open-access textbook Action Models for Robot Learning, by Pavan Kumar Kandapagari, as they are drafted.',
    site: context.site ?? 'https://action-models-book.vercel.app',
    items,
    customData: '<language>en-us</language>',
  });
};
