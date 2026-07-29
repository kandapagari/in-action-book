// Pure-logic checks for frontmatter normalization. Run with `npm test`
// (`node --test`). frontmatter.ts is dependency-free; gray-matter is used here
// only to confirm the normalized block parses each key_refs item as a plain
// string (not the `[object Object]` mapping YAML would otherwise produce).

import test from 'node:test';
import assert from 'node:assert/strict';
import matter from 'gray-matter';

import { normalizeFrontmatter } from './frontmatter.ts';

const raw = `---
title: Why "action" is the hard part of robotics
prereqs: linear algebra, basic Python
key_refs:
  - Kober, Bagnell, Peters (2013). RL in Robotics: A Survey. IJRR 32(11).
  - Sapkota et al. (2025). VLA Models: Concepts, Progress. arXiv:2505.04769.
  - Stanford HAI (2025). AI Index Report — Robotics chapter.
---

body text
`;

test('key_refs items with colons normalize to plain strings', () => {
  const { data } = matter(normalizeFrontmatter(raw));
  assert.ok(Array.isArray(data.key_refs));
  for (const ref of data.key_refs) {
    assert.equal(typeof ref, 'string', `expected string, got ${typeof ref}: ${JSON.stringify(ref)}`);
  }
  // The colon+space that previously turned this into an object is preserved
  // verbatim inside the plain string.
  assert.match(data.key_refs[0], /RL in Robotics: A Survey\. IJRR 32\(11\)\./);
  assert.match(data.key_refs[1], /VLA Models: Concepts, Progress\./);
  // A colon-free item stays untouched.
  assert.equal(data.key_refs[2], 'Stanford HAI (2025). AI Index Report — Robotics chapter.');
});
