// Self-check for the pure helpers in build-ebook.mjs. Run with `npm test`
// (`node --test`). Fabricated in-memory fixtures only — no pandoc, no
// filesystem; importing the module must not trigger the pandoc pipeline
// (guarded by an import.meta.url check in build-ebook.mjs).

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  sectionSortKey,
  sortSectionIds,
  parseProgressTitles,
  stripFrontmatter,
  shiftHeadings,
  assembleBook,
} from './build-ebook.mjs';

test('sortSectionIds orders numerically with N.x last', () => {
  assert.deepEqual(sortSectionIds(['1.x', '1.10', '1.2', '1.1']), ['1.1', '1.2', '1.10', '1.x']);
});

test('sectionSortKey matches the book-content.ts semantics', () => {
  assert.equal(sectionSortKey('4.2'), 2);
  assert.equal(sectionSortKey('4.x'), Number.POSITIVE_INFINITY);
});

test('parseProgressTitles picks up chapter and appendix titles only', () => {
  const raw = [
    '#### Chapter 3. Math and ML prerequisites in 30 minutes',
    '',
    '- [x] 3.1 Vectors, matrices, gradients, and why the chain rule rules robotics',
    '',
    '### Appendices',
    '',
    '- [x] A. Linear algebra refresher',
    '- [ ] F. VLA model zoo',
    '',
    '## Daily session log',
    '',
    '- 2026-06-03 — drafted Appendix B: Probability and information theory (~2226 words)',
  ].join('\n');
  const { chapters, appendices } = parseProgressTitles(raw);
  assert.equal(chapters.get(3), 'Math and ML prerequisites in 30 minutes');
  assert.equal(chapters.size, 1);
  assert.equal(appendices.get('A'), 'Linear algebra refresher');
  assert.equal(appendices.get('F'), 'VLA model zoo');
  // The session-log "Appendix B:" line has no checkbox and must not match.
  assert.equal(appendices.size, 2);
});

test('stripFrontmatter removes the leading block and keeps the body', () => {
  const raw = '---\nchapter: 3\nsection: 3.4\ntitle: Three loss families: a colon\n---\n\n# Body\n';
  assert.equal(stripFrontmatter(raw), '\n# Body\n');
  assert.equal(stripFrontmatter('# No frontmatter\n'), '# No frontmatter\n');
});

test('shiftHeadings shifts headings outside code fences only', () => {
  const md = [
    '# 1.1  Title',
    '## Sub',
    '```python',
    '# comment, not a heading',
    '```',
    '# After fence',
  ].join('\n');
  const out = shiftHeadings(md).split('\n');
  assert.equal(out[0], '## 1.1  Title');
  assert.equal(out[1], '### Sub');
  assert.equal(out[3], '# comment, not a heading');
  assert.equal(out[5], '## After fence');
});

test('shiftHeadings only closes a fence on an equal-or-longer backtick run', () => {
  const md = ['````', '```', '# still inside', '````', '# outside'].join('\n');
  const out = shiftHeadings(md).split('\n');
  assert.equal(out[2], '# still inside');
  assert.equal(out[4], '## outside');
});

test('assembleBook joins units with blank lines and a trailing newline', () => {
  const out = assembleBook([
    { heading: '# Chapter 1 — X', bodies: ['## 1.1 A', '## 1.2 B'] },
    { heading: '# Appendix A — Y', bodies: ['## Appendix A. Z'] },
  ]);
  assert.equal(out, '# Chapter 1 — X\n\n## 1.1 A\n\n## 1.2 B\n\n# Appendix A — Y\n\n## Appendix A. Z\n');
});
