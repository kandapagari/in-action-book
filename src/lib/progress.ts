// Parses src/content/book/PROGRESS.md at build time into a structured TOC plus
// a per-section drafted-date index sourced from the "Daily session log".
//
// Note: src/content/book/ is a generated mirror of the authoring source at
// ../book/ (workspace root). The mirror is rebuilt by tools/sync-book-to-site.sh
// on `npm run dev` and `npm run build`. Never edit the mirror by hand.
//
// The single source of truth (modulo the sync) is the markdown file itself;
// nothing here fabricates titles or dates.

// PROGRESS.md is the single source of truth for the TOC + per-section draft
// dates. We import it as a raw string via Vite so the content is inlined into
// the build and no filesystem access is needed at render time.
import progressRaw from '../content/book/PROGRESS.md?raw';

export type SectionStatus = 'pending' | 'in-progress' | 'drafted' | 'revised';

export type SectionEntry = {
  section: string;
  title: string;
  status: SectionStatus;
};

export type ChapterEntry = {
  chapter: number;
  title: string;
  sections: SectionEntry[];
};

export type PartEntry = {
  partNumber: number;
  ordinal: 'ONE' | 'TWO' | 'THREE' | 'FOUR' | 'FIVE';
  title: string;
  chapters: ChapterEntry[];
};

export type AppendixEntry = {
  letter: string;
  title: string;
  status: SectionStatus;
};

export type Progress = {
  parts: PartEntry[];
  appendices: AppendixEntry[];
  sessionLog: SessionLogEntry[];
  totalSections: number;
  draftedSections: number;
  totalChapters: number;
  chaptersComplete: number;
  totalParts: number;
};

export type SessionLogEntry = {
  date: string; // ISO YYYY-MM-DD
  section: string; // e.g. "1.1"
  title: string;
  words: number;
};

const STATUS_MAP: Record<string, SectionStatus> = {
  ' ': 'pending',
  '~': 'in-progress',
  x: 'drafted',
  r: 'revised',
};

const ORDINALS: PartEntry['ordinal'][] = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'];

let cache: Progress | null = null;

export function loadProgress(): Progress {
  if (cache) return cache;
  const lines = progressRaw.split(/\r?\n/);

  const parts: PartEntry[] = [];
  const appendices: AppendixEntry[] = [];
  const sessionLog: SessionLogEntry[] = [];

  let currentPart: PartEntry | null = null;
  let currentChapter: ChapterEntry | null = null;
  let inAppendices = false;
  let inSessionLog = false;

  const partHeading = /^###\s+Part\s+(\d+)\s+—\s+(.+)$/;
  const chapterHeading = /^####\s+Chapter\s+(\d+)\.\s+(.+)$/;
  const appendicesHeading = /^###\s+Appendices\s*$/;
  const sessionLogHeading = /^##\s+Daily session log\s*$/;
  const sectionLine = /^-\s+\[([ x~r])\]\s+(\d+\.[\dxr]+)\s+(.+)$/;
  const appendixLine = /^-\s+\[([ x~r])\]\s+([A-F])\.\s+(.+)$/;
  const sessionLine =
    /^-\s+(\d{4}-\d{2}-\d{2})\s+—\s+drafted\s+§([\d]+\.[\dxr]+)\s+(.+?)\s+\(~([\d,]+)\s*words\)\s*$/;

  for (const line of lines) {
    if (sessionLogHeading.test(line)) {
      inSessionLog = true;
      inAppendices = false;
      currentPart = null;
      currentChapter = null;
      continue;
    }

    if (inSessionLog) {
      const m = sessionLine.exec(line);
      if (m) {
        sessionLog.push({
          date: m[1],
          section: m[2],
          title: m[3].trim(),
          words: parseInt(m[4].replace(/,/g, ''), 10),
        });
      }
      continue;
    }

    if (appendicesHeading.test(line)) {
      inAppendices = true;
      currentPart = null;
      currentChapter = null;
      continue;
    }

    if (inAppendices) {
      const m = appendixLine.exec(line);
      if (m) {
        appendices.push({
          letter: m[2],
          title: m[3].trim(),
          status: STATUS_MAP[m[1]],
        });
      }
      continue;
    }

    const partM = partHeading.exec(line);
    if (partM) {
      const partNumber = parseInt(partM[1], 10);
      const ordinal = ORDINALS[partNumber - 1];
      if (!ordinal) throw new Error(`Unknown part number ${partNumber} in PROGRESS.md`);
      currentPart = {
        partNumber,
        ordinal,
        title: partM[2].trim(),
        chapters: [],
      };
      parts.push(currentPart);
      currentChapter = null;
      continue;
    }

    const chM = chapterHeading.exec(line);
    if (chM && currentPart) {
      currentChapter = {
        chapter: parseInt(chM[1], 10),
        title: chM[2].trim(),
        sections: [],
      };
      currentPart.chapters.push(currentChapter);
      continue;
    }

    const secM = sectionLine.exec(line);
    if (secM && currentChapter) {
      currentChapter.sections.push({
        section: secM[2],
        title: secM[3].trim(),
        status: STATUS_MAP[secM[1]],
      });
      continue;
    }
  }

  const totalSections =
    parts.reduce((n, p) => n + p.chapters.reduce((m, c) => m + c.sections.length, 0), 0) +
    appendices.length;

  const draftedSectionsOnly = parts.reduce(
    (n, p) =>
      n +
      p.chapters.reduce(
        (m, c) =>
          m + c.sections.filter((s) => s.status === 'drafted' || s.status === 'revised').length,
        0,
      ),
    0,
  );
  const draftedAppendices = appendices.filter(
    (a) => a.status === 'drafted' || a.status === 'revised',
  ).length;

  const totalChapters = parts.reduce((n, p) => n + p.chapters.length, 0);
  const chaptersComplete = parts.reduce(
    (n, p) =>
      n +
      p.chapters.filter((c) =>
        c.sections.length > 0 &&
        c.sections.every((s) => s.status === 'drafted' || s.status === 'revised'),
      ).length,
    0,
  );

  cache = {
    parts,
    appendices,
    sessionLog,
    totalSections,
    draftedSections: draftedSectionsOnly + draftedAppendices,
    totalChapters,
    chaptersComplete,
    totalParts: parts.length,
  };
  return cache;
}

export function dateForSection(section: string): string | null {
  const log = loadProgress().sessionLog;
  // The log can have multiple entries for the same section (rare); take the
  // first (earliest) draft date.
  const entry = log.find((e) => e.section === section);
  return entry ? entry.date : null;
}

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

export function formatLongDate(iso: string): string {
  const [y, m, d] = iso.split('-').map((s) => parseInt(s, 10));
  return `${MONTHS[m - 1]} ${d}, ${y}`;
}

export function formatShortDate(iso: string): string {
  const [, m, d] = iso.split('-').map((s) => parseInt(s, 10));
  return `${MONTHS[m - 1].slice(0, 3)} ${d}`;
}

export function sectionRouteParams(section: string): { chapter: string; section: string } {
  const [chapter, ...rest] = section.split('.');
  return { chapter, section: rest.join('.') };
}
