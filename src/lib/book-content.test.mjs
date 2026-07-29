// Pure-logic checks for the book-content search ranker. Run with `npm test`
// (`node --test`). No glob, no filesystem — only rankMatches on fabricated
// docs (import.meta.glob is undefined under plain Node, so the doc list from
// book-content.ts itself is empty; rankMatches stays pure and testable).

import test from 'node:test';
import assert from 'node:assert/strict';

import { rankMatches } from './book-content.ts';

function doc(id, title, body) {
  return {
    id,
    kind: 'section',
    chapter: Number(id.split('.')[0]),
    title,
    status: 'draft',
    prereqs: '',
    key_refs: [],
    body,
    url: `https://example.test/${id}/`,
  };
}

const docs = [
  doc('1.1', 'Why action is hard', 'The action problem is turning intent into motion.'),
  doc('2.1', 'Diffusion policy', 'A diffusion policy models the action distribution with noise.'),
  doc('3.1', 'Perception', 'Cameras were noisy and compute was scarce in the 1980s.'),
];

test('rankMatches returns only docs that match the query', () => {
  const results = rankMatches(docs, 'diffusion', 5);
  assert.equal(results.length, 1);
  assert.equal(results[0].id, '2.1');
});

test('rankMatches weights title hits above body hits', () => {
  // "action" appears in the body of 1.1 (once) and 2.1 (once), but only 1.1
  // has it in the title → 1.1 must rank first.
  const results = rankMatches(docs, 'action', 5);
  assert.equal(results[0].id, '1.1');
});

test('rankMatches respects the limit', () => {
  const results = rankMatches(docs, 'the', 1);
  assert.ok(results.length <= 1);
});

test('rankMatches excerpt surrounds the first body match', () => {
  const results = rankMatches(docs, 'motion', 5);
  assert.equal(results[0].id, '1.1');
  assert.match(results[0].excerpt, /motion/);
});

test('rankMatches returns url and title for each result', () => {
  const [top] = rankMatches(docs, 'perception', 5);
  assert.equal(top.id, '3.1');
  assert.equal(top.title, 'Perception');
  assert.equal(top.url, 'https://example.test/3.1/');
});

test('rankMatches returns nothing for an empty query', () => {
  assert.deepEqual(rankMatches(docs, '   ', 5), []);
});
