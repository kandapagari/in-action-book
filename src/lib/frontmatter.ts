// Shared, dependency-free frontmatter normalization for the book's markdown.
//
// The book's frontmatter is human-authored and includes a handful of cases
// that strict YAML rejects:
//
//   1. Unquoted `title:` lines that themselves contain a colon (e.g.
//      "Three loss families: supervised, RL, self-supervised"). js-yaml
//      treats the second colon as a nested mapping delimiter.
//   2. Unquoted `prereqs:` lines with colons inside (rare, but possible).
//   3. Unquoted `key_refs:` block-sequence items with colons inside (common —
//      e.g. "- Kober … RL in Robotics: A Survey. IJRR 32(11)."). YAML parses
//      "text: more" inside a `-` item as a single-key mapping, so the item
//      becomes an object instead of a plain string.
//
// We pre-process the frontmatter block to wrap any single-line value for the
// known free-text fields — and each item of the known block-sequence fields —
// in double quotes (escaping embedded quotes) before handing it to a YAML
// parser. Used by both book-loader.ts (Astro content pipeline) and
// book-content.ts (the MCP server's gray-matter parse) so the two stay in
// sync. Pure string/regex only — safe to import under plain Node.

const FREE_TEXT_FIELDS = ['title', 'prereqs'];
const SEQUENCE_FIELDS = ['key_refs'];

export function quoteIfNeeded(value: string): string {
  const trimmed = value.trim();
  if (trimmed === '') return trimmed;
  // Already quoted (single or double) — leave alone.
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed;
  }
  // No problematic colon — leave alone.
  if (!/:\s/.test(trimmed)) return trimmed;
  const escaped = trimmed.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `"${escaped}"`;
}

export function normalizeFrontmatter(raw: string): string {
  // Match a leading frontmatter block: --- ... ---
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n/.exec(raw);
  if (!m) return raw;
  const block = m[1];
  const rest = raw.slice(m[0].length);
  // Track whether we're inside a block-sequence field (e.g. `key_refs:`), whose
  // `- item` lines need per-item quoting rather than single-line value quoting.
  let inSequence = false;
  const fixed = block
    .split(/\r?\n/)
    .map((line) => {
      for (const field of FREE_TEXT_FIELDS) {
        const re = new RegExp(`^(${field}):\\s*(.*)$`);
        const fm = re.exec(line);
        if (fm) {
          inSequence = false;
          return `${fm[1]}: ${quoteIfNeeded(fm[2])}`;
        }
      }
      // A block-sequence header with no inline value (e.g. `key_refs:`) opens
      // the sequence; its items follow on subsequent `- …` lines.
      if (SEQUENCE_FIELDS.some((f) => new RegExp(`^${f}:\\s*$`).test(line))) {
        inSequence = true;
        return line;
      }
      if (inSequence) {
        const im = /^(\s*-\s+)(.*)$/.exec(line);
        if (im) return `${im[1]}${quoteIfNeeded(im[2])}`;
        // Any non-item, non-blank line ends the sequence.
        if (line.trim() !== '') inSequence = false;
      }
      return line;
    })
    .join('\n');
  return `---\n${fixed}\n---\n${rest}`;
}
