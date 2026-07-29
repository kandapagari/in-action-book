// Pure-logic checks for the newsletter drip. Run with `npm test`
// (`node --test`). No framework, no Redis, no email — only the sendable-gating
// and next-chapter selection.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  readyChapters,
  sendableChapterNumbers,
  nextChapterForCursor,
} from './newsletter-content.ts';

// Minimal fabricated Progress: ch1 complete, ch2 incomplete (a pending
// section), ch3 complete. Only the fields the logic reads are populated.
function fakeProgress() {
  const sec = (status) => ({ section: 'x', title: 't', status });
  return {
    parts: [
      {
        partNumber: 1,
        ordinal: 'ONE',
        title: 'Part',
        chapters: [
          { chapter: 1, title: 'One', sections: [sec('drafted'), sec('revised')] },
          { chapter: 2, title: 'Two', sections: [sec('drafted'), sec('pending')] },
          { chapter: 3, title: 'Three', sections: [sec('revised')] },
        ],
      },
    ],
    appendices: [],
    sessionLog: [],
    totalSections: 0,
    draftedSections: 0,
    totalChapters: 3,
    chaptersComplete: 2,
    totalParts: 1,
  };
}

test('readyChapters returns only complete chapters, ascending', () => {
  assert.deepEqual(readyChapters(fakeProgress()), [1, 3]);
});

test('sendableChapterNumbers gates complete chapters by excerpt presence', () => {
  const ready = readyChapters(fakeProgress()); // [1, 3]
  // Chapter 3 is complete but has no excerpt yet → not sendable.
  assert.deepEqual(sendableChapterNumbers(ready, [1]), [1]);
  // A complete chapter with an excerpt but that isn't ready is never sendable.
  assert.deepEqual(sendableChapterNumbers(ready, [1, 2]), [1]);
  // Once chapter 3's excerpt is authored, it becomes sendable.
  assert.deepEqual(sendableChapterNumbers(ready, [1, 3]), [1, 3]);
});

test('nextChapterForCursor: new joiner receives the first sendable chapter', () => {
  assert.equal(nextChapterForCursor([1, 3], 0), 1);
});

test('nextChapterForCursor: caught-up subscriber receives null', () => {
  assert.equal(nextChapterForCursor([1], 1), null);
  assert.equal(nextChapterForCursor([1, 3], 3), null);
});

test('nextChapterForCursor: a newly-sendable chapter is returned next', () => {
  // Subscriber caught up at chapter 1 while only [1] was sendable → null.
  assert.equal(nextChapterForCursor([1], 1), null);
  // Chapter 3 becomes sendable (excerpt added) → it is delivered next.
  assert.equal(nextChapterForCursor([1, 3], 1), 3);
});
