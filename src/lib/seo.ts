// Small SEO helpers. Used to derive a real plain-text excerpt from
// already-rendered section HTML so per-page <meta description> reflects
// actual prose Google can match against instead of a templated string.

export function plainText(html: string): string {
  return html
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

export function excerpt(html: string, maxLen = 160): string {
  const text = plainText(html);
  if (text.length <= maxLen) return text;
  const cut = text.slice(0, maxLen);
  const lastSpace = cut.lastIndexOf(' ');
  return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut).trimEnd() + '…';
}

// ---------------------------------------------------------------------------
// Structured data (JSON-LD) — single source of truth for the author/book
// entity graph. The Person and Book are defined once here and referenced by
// @id elsewhere so search engines can stitch the nodes into one entity. The
// canonical production origin is fixed (the site lives on this subdomain);
// the @id IRIs must be stable, so it is hardcoded rather than read from env.
// ---------------------------------------------------------------------------

export const SITE = 'https://action-models-book.vercel.app';

export const PERSON_ID = `${SITE}/about/#person`;
export const BOOK_ID = `${SITE}/#book`;
export const WEBSITE_ID = `${SITE}/#website`;

type JsonLd = Record<string, unknown>;

// Public professional profiles only — no email/phone/address/private data.
const PERSON_SAME_AS = [
  'https://github.com/kandapagari',
  'https://www.linkedin.com/in/kandapagari/',
  'https://kandapagari.vercel.app',
  'https://orcid.org/0009-0003-7497-3833',
  'https://scholar.google.com/citations?user=EkSY9wUAAAAJ',
];

// Field terms the author is an authority on. Tasteful breadth, not a stuffed
// list: the canonical field names plus the systems the book actually covers.
const PERSON_KNOWS_ABOUT = [
  'Vision-language-action models',
  'Robot learning',
  'Action models',
  'Foundation models for robotics',
  'Imitation learning',
  'Behavior cloning',
  'Diffusion policy',
  'Flow matching',
  'OpenVLA',
  'π₀ (pi-zero)',
  'RT-1',
  'RT-2',
  'GR00T N1',
  'Helix',
  'Octo',
];

const BOOK_ABOUT = [
  'Action models',
  'Vision-language-action models',
  'Robot learning',
  'Imitation learning',
  'Reinforcement learning for robotics',
  'Robot manipulation',
  'Foundation models for robotics',
];

const BOOK_KEYWORDS = [
  'vision-language-action models',
  'VLA',
  'robot learning',
  'action models',
  'foundation models for robotics',
  'imitation learning',
  'OpenVLA',
  'π₀',
  'RT-1',
  'RT-2',
  'GR00T N1',
  'Helix',
  'Octo',
  'diffusion policy',
  'flow matching',
  'behavior cloning',
].join(', ');

export function personEntity(): JsonLd {
  return {
    '@type': 'Person',
    '@id': PERSON_ID,
    name: 'Pavan Kumar Kandapagari',
    url: `${SITE}/`,
    mainEntityOfPage: `${SITE}/about/`,
    jobTitle: 'Robotics foundation-models researcher',
    description:
      'Researcher and engineer working on foundation models for robotics — vision-language-action (VLA) policies, robot learning, and action models — and author of the open-access textbook Action Models for Robot Learning.',
    knowsAbout: PERSON_KNOWS_ABOUT,
    sameAs: PERSON_SAME_AS,
  };
}

export function bookEntity(): JsonLd {
  return {
    '@type': 'Book',
    '@id': BOOK_ID,
    name: 'Action Models for Robot Learning',
    alternateName:
      'Action Models for Robot Learning: an open-access VLA & robot-learning textbook',
    url: `${SITE}/`,
    inLanguage: 'en',
    author: { '@id': PERSON_ID },
    publisher: { '@id': PERSON_ID },
    genre: ['Textbook', 'Robotics', 'Machine learning'],
    keywords: BOOK_KEYWORDS,
    about: BOOK_ABOUT,
    description:
      'An open-access textbook on action models and vision-language-action policies for robot learning. Covers RT-1, RT-2, OpenVLA, π₀, Helix, and GR00T N1, organised around four families of action models: symbolic planners, geometric and inverse-dynamics controllers, value-based policies, and learned policies from demonstration.',
  };
}

export function webSiteEntity(): JsonLd {
  return {
    '@type': 'WebSite',
    '@id': WEBSITE_ID,
    name: 'Action Models for Robot Learning',
    url: `${SITE}/`,
    inLanguage: 'en',
    author: { '@id': PERSON_ID },
    description:
      'Open-access textbook on vision-language-action (VLA) models, action models, and robot learning by Pavan Kumar Kandapagari.',
  };
}

// Wraps one or more entity nodes into a single @graph document so a page can
// emit several linked entities under one <script type="application/ld+json">.
export function graph(...nodes: JsonLd[]): JsonLd {
  return { '@context': 'https://schema.org', '@graph': nodes };
}

type TechArticleInput = {
  headline: string;
  url: string;
  description: string;
  datePublished?: string | null;
  /** Topic strings detected in the section/appendix (real terms only). */
  topics?: string[];
};

export function techArticleEntity({
  headline,
  url,
  description,
  datePublished,
  topics,
}: TechArticleInput): JsonLd {
  return {
    '@type': 'TechArticle',
    headline,
    url,
    mainEntityOfPage: url,
    inLanguage: 'en',
    author: { '@id': PERSON_ID, name: 'Pavan Kumar Kandapagari' },
    publisher: { '@id': PERSON_ID },
    isPartOf: {
      '@id': BOOK_ID,
      '@type': 'Book',
      name: 'Action Models for Robot Learning',
      url: `${SITE}/`,
    },
    description,
    ...(topics && topics.length ? { about: topics, keywords: topics.join(', ') } : {}),
    ...(datePublished ? { datePublished, dateModified: datePublished } : {}),
  };
}

export function breadcrumbList(items: { name: string; url: string }[]): JsonLd {
  return {
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, i) => ({
      '@type': 'ListItem',
      position: i + 1,
      name: item.name,
      item: item.url,
    })),
  };
}

// Canonical-cased field/model terms to surface in per-page `about`/`keywords`.
// We only emit a term when it actually appears in the rendered prose, so the
// structured data stays truthful (no keyword stuffing).
const TOPIC_TERMS: { term: string; pattern: RegExp }[] = [
  { term: 'Vision-language-action models', pattern: /vision[- ]language[- ]action|\bVLA\b/i },
  { term: 'OpenVLA', pattern: /openvla/i },
  { term: 'RT-1', pattern: /\bRT-?1\b/i },
  { term: 'RT-2', pattern: /\bRT-?2\b/i },
  { term: 'RT-X', pattern: /\bRT-?X\b/i },
  { term: 'GR00T N1', pattern: /gr00t|groot/i },
  { term: 'Helix', pattern: /\bhelix\b/i },
  { term: 'Octo', pattern: /\bocto\b/i },
  { term: 'π₀ (pi-zero)', pattern: /π[₀0]|\bpi[- ]?zero\b/i },
  { term: 'Diffusion policy', pattern: /diffusion polic/i },
  { term: 'Flow matching', pattern: /flow matching/i },
  { term: 'Behavior cloning', pattern: /behavio(u)?r cloning/i },
  { term: 'Imitation learning', pattern: /imitation learning/i },
  { term: 'Reinforcement learning', pattern: /reinforcement learning/i },
  { term: 'Robot learning', pattern: /robot learning/i },
  { term: 'Action models', pattern: /action models?/i },
];

export function detectTopics(...sources: string[]): string[] {
  const haystack = sources.join(' \n ');
  const found: string[] = [];
  for (const { term, pattern } of TOPIC_TERMS) {
    if (pattern.test(haystack)) found.push(term);
  }
  return found;
}
