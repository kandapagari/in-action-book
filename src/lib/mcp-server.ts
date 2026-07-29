// Builds a fresh McpServer exposing the book to agents. Stateless: buildServer()
// is called once per HTTP request by createMcpHandler (see api/mcp.ts), so no
// shared mutable state leaks between requests.
//
// Content + TOC come from the same sources the site uses: loadProgress()
// (PROGRESS.md → parts/chapters/sections/appendices + status) and
// book-content.ts (the inlined section/appendix markdown).

import { McpServer } from '@modelcontextprotocol/server';
import { z } from 'zod';
import { loadProgress } from './progress.ts';
import type { SectionStatus } from './progress.ts';
import { allDocs, getById, getByChapter, rankMatches } from './book-content.ts';

function textResult(payload: unknown) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(payload, null, 2) }] };
}

function errorResult(message: string) {
  return { content: [{ type: 'text' as const, text: message }], isError: true as const };
}

// Canonical (drafted/revised/pending/in-progress) status by section id and by
// appendix letter, sourced from PROGRESS.md — richer than the uniform "draft"
// frontmatter status on the files.
function statusIndex(): {
  sections: Map<string, SectionStatus>;
  appendices: Map<string, SectionStatus>;
  chapterTitles: Map<number, string>;
} {
  const progress = loadProgress();
  const sections = new Map<string, SectionStatus>();
  const appendices = new Map<string, SectionStatus>();
  const chapterTitles = new Map<number, string>();
  for (const part of progress.parts) {
    for (const chapter of part.chapters) {
      chapterTitles.set(chapter.chapter, chapter.title);
      for (const s of chapter.sections) sections.set(s.section, s.status);
    }
  }
  for (const a of progress.appendices) appendices.set(a.letter, a.status);
  return { sections, appendices, chapterTitles };
}

export function buildServer(): McpServer {
  const server = new McpServer({ name: 'action-models-book', version: '1.0.0' });

  server.registerTool(
    'get_table_of_contents',
    {
      title: 'Table of contents',
      description:
        'Full table of contents of the book "Action Models for Robot Learning": parts → chapters → sections (with status) plus appendices A–F. `readable: true` marks entries whose full text can be fetched with get_section/get_chapter; pending entries have no body yet.',
    },
    async () => {
      const progress = loadProgress();
      const readable = new Set(allDocs().map((d) => d.id));
      const byId = new Map(allDocs().map((d) => [d.id, d]));
      const parts = progress.parts.map((part) => ({
        part: part.partNumber,
        title: part.title,
        chapters: part.chapters.map((chapter) => ({
          chapter: chapter.chapter,
          title: chapter.title,
          sections: chapter.sections.map((s) => ({
            id: s.section,
            title: s.title,
            status: s.status,
            readable: readable.has(s.section),
            url: byId.get(s.section)?.url,
          })),
        })),
      }));
      const appendices = progress.appendices.map((a) => ({
        id: a.letter,
        title: a.title,
        status: a.status,
        readable: readable.has(a.letter),
        url: byId.get(a.letter)?.url,
      }));
      return textResult({
        book: 'Action Models for Robot Learning',
        totals: {
          parts: progress.totalParts,
          chapters: progress.totalChapters,
          sections: progress.totalSections,
          drafted: progress.draftedSections,
        },
        parts,
        appendices,
      });
    },
  );

  server.registerTool(
    'get_section',
    {
      title: 'Get a section or appendix',
      description:
        'Fetch one section by id ("1.1", "4.1", "1.x") or one appendix by letter ("A"). Returns title, chapter, status, prerequisites, key references, the public URL, and the full markdown body. Only drafted/revised entries have a body.',
      inputSchema: z.object({
        id: z
          .string()
          .describe('Section id like "1.1", "4.1", or "1.x", or an appendix letter like "A".'),
      }),
    },
    async ({ id }) => {
      const doc = getById(id);
      const { sections, appendices } = statusIndex();
      if (!doc) {
        const known = sections.has(id) || appendices.has(id.toUpperCase());
        if (known) {
          return errorResult(
            `Section "${id}" is in the table of contents but has not been drafted yet, so it has no body. Use get_table_of_contents to see which entries are readable.`,
          );
        }
        return errorResult(
          `No section or appendix with id "${id}". Use get_table_of_contents to list valid ids.`,
        );
      }
      const status =
        doc.kind === 'appendix'
          ? (appendices.get(doc.id.toUpperCase()) ?? doc.status)
          : (sections.get(doc.id) ?? doc.status);
      return textResult({
        id: doc.id,
        kind: doc.kind,
        chapter: doc.chapter,
        title: doc.title,
        status,
        prereqs: doc.prereqs,
        key_refs: doc.key_refs,
        url: doc.url,
        body: doc.body,
      });
    },
  );

  server.registerTool(
    'get_chapter',
    {
      title: 'Get a chapter',
      description:
        "Fetch a whole chapter by number: the chapter title and every readable section's full markdown body, in order.",
      inputSchema: z.object({
        chapter: z.number().int().describe('Chapter number, e.g. 1.'),
      }),
    },
    async ({ chapter }) => {
      const { chapterTitles } = statusIndex();
      const title = chapterTitles.get(chapter);
      if (title === undefined) {
        return errorResult(
          `No chapter ${chapter} in the book. Use get_table_of_contents to list chapters.`,
        );
      }
      const sections = getByChapter(chapter).map((d) => ({
        id: d.id,
        title: d.title,
        url: d.url,
        body: d.body,
      }));
      return textResult({ chapter, title, sections });
    },
  );

  server.registerTool(
    'search',
    {
      title: 'Search the book',
      description:
        'Case-insensitive full-text search across all readable section and appendix bodies and titles. Returns ranked matches (title hits weighted higher) with id, title, URL, and an excerpt around the first match.',
      inputSchema: z.object({
        query: z.string().describe('Search terms.'),
        limit: z
          .number()
          .int()
          .positive()
          .max(50)
          .optional()
          .describe('Maximum number of results (default 5).'),
      }),
    },
    async ({ query, limit }) => {
      const results = rankMatches(allDocs(), query, limit ?? 5);
      return textResult({ query, count: results.length, results });
    },
  );

  return server;
}
