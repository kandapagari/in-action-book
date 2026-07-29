// Shared, dependency-free frontmatter normalization for the book's markdown.
//
// The book's frontmatter is human-authored and includes a handful of cases
// that strict YAML rejects:
//
//   1. Unquoted `title:` lines that themselves contain a colon (e.g.
//      "Three loss families: supervised, RL, self-supervised"). js-yaml
//      treats the second colon as a nested mapping delimiter.
//   2. Unquoted `prereqs:` lines with colons inside (rare, but possible).
//
// We pre-process the frontmatter block to wrap any single-line value for the
// known free-text fields in double quotes (escaping embedded quotes) before
// handing it to a YAML parser. Used by both book-loader.ts (Astro content
// pipeline) and book-content.ts (the MCP server's gray-matter parse) so the
// two stay in sync. Pure string/regex only — safe to import under plain Node.

const FREE_TEXT_FIELDS = ['title', 'prereqs'];

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
  const fixed = block
    .split(/\r?\n/)
    .map((line) => {
      for (const field of FREE_TEXT_FIELDS) {
        const re = new RegExp(`^(${field}):\\s*(.*)$`);
        const fm = re.exec(line);
        if (fm) {
          return `${fm[1]}: ${quoteIfNeeded(fm[2])}`;
        }
      }
      return line;
    })
    .join('\n');
  return `---\n${fixed}\n---\n${rest}`;
}
