// Helpers for turning the `key_refs` array (free-text reference strings) into
// link-decorated HTML. The book's frontmatter uses arXiv IDs of the form
// `arXiv:XXXX.XXXXX`; everything else is left as plain text.

const ARXIV_RE = /arXiv:(\d{4}\.\d{4,5})(v\d+)?/g;

export function linkifyRef(raw: string): string {
  const escaped = escapeHtml(raw);
  return escaped.replace(ARXIV_RE, (_match, id, v) => {
    const versioned = v ? `${id}${v}` : id;
    return `<a href="https://arxiv.org/abs/${id}">arXiv:${versioned}</a>`;
  });
}

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Appendix bodies open with a `# Appendix X. …` H1 that duplicates the
// frontmatter title we render in the page header. Drop the leading rendered
// `<h1>` so the title isn't shown twice.
export function stripLeadingH1(html: string): string {
  return html.replace(/^\s*<h1\b[^>]*>[\s\S]*?<\/h1>\s*/i, '');
}

// Split a rendered HTML body on H2 boundaries. Each chunk after the first
// begins with the `<h2>`; chunk 0 holds everything before the first H2.
export function splitOnH2(html: string): string[] {
  const parts = html.split(/(?=<h2[\s>])/i);
  return parts.length > 0 ? parts : [html];
}
