"""Markdown → clean spoken prose for Kokoro TTS.

The preprocessor's job is to take an authored book section (markdown with
LaTeX math, code blocks, sidenote references, citations, URLs, and a soup of
technical terminology) and produce text that Kokoro will pronounce naturally.

Be conservative: if a fragment cannot be cleanly spoken (e.g. a multi-line
display equation), skip it rather than mispronounce it. Better silence than
nonsense.

The pronunciation table at the top of this file is the user-facing knob — add
to it whenever a new term mispronounces in the audio. Each rule is a (pattern,
replacement) tuple applied as a regex `re.sub` with the `re.UNICODE` flag.
Order matters: more specific patterns must come before more general ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────────────
# Pronunciation substitutions
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry is (pattern, replacement). Patterns are interpreted as regex with
# re.UNICODE. Use \b for word boundaries when matching ASCII words; for
# patterns containing non-ASCII characters (π, q̇, etc.), use lookarounds
# explicitly because \b does not behave intuitively across the ASCII/Unicode
# boundary.
#
# Add new rules at the TOP of each block — they apply in order, so the most
# specific rules must fire first.

PRONUNCIATION_RULES: list[tuple[str, str]] = [
    # ── Specific tokens that contain π first, before the bare-π rule ──
    (r"π0|π₀|\bpi0\b", "pi zero"),
    (r"π", "pi"),

    # ── Acronyms and model names ────────────────────────────────────
    # arXiv mangles in Kokoro; "archive" is the cleanest reading.
    (r"\barXiv\b", "archive"),
    (r"\bOpenVLA\b", "Open V L A"),
    (r"\bRT-1\b", "R T one"),
    (r"\bRT-2\b", "R T two"),
    (r"\bRT-X\b", "R T X"),
    (r"\bGR00T\s*N1\b", "Groot N one"),
    (r"\bGR00T\s*N1.5\b", "Groot N one point five"),
    (r"\bGR00T\s*N1.6\b", "Groot N one point six"),
    (r"\bGR00T\s*N1.7\b", "Groot N one point seven"),
    (r"\bGR00T\b", "Groot"),
    (r"\bBC-Z\b", "B C Z"),
    (r"\bDAgger\b", "Dagger"),
    (r"\bSAC\b", "S A C"),
    (r"\bPPO\b", "P P O"),
    (r"\bDDPG\b", "D D P G"),
    (r"\bTD3\b", "T D three"),
    (r"\bMDP\b", "M D P"),
    (r"\bMDPs\b", "M D Ps"),
    (r"\bLIBERO\b", "Libero"),
    (r"\bCALVIN\b", "Calvin"),
    (r"\bIRL\b", "I R L"),
    (r"\bKL divergence\b", "K L divergence"),
    (r"\bKL\b", "K L"),
    (r"\bVLM\b", "V L M"),
    (r"\bVLMs\b", "V L Ms"),
    (r"\bVLA\b", "V L A"),
    (r"\bVLAs\b", "V L As"),
    (r"\bDQN\b", "D Q N"),
    (r"\bLLM\b", "L L M"),
    (r"\bLLMs\b", "L L Ms"),
    (r"\bLoRA\b", "Lora"),
    (r"\bACT\b(?=\s+(?:and|architecture|model))", "A C T"),
    (r"\bJEPA\b", "Jepa"),
    (r"\bV-JEPA\b", "V Jepa"),
    (r"\bRSSM\b", "R S S M"),
    (r"\bGGUF\b", "G G U F"),
    (r"\bSTRIPS\b", "Strips"),
    (r"\bPDDL\b", "P D D L"),
    (r"\bAPI\b", "A P I"),
    (r"\bAPIs\b", "A P Is"),
    (r"\bGPU\b", "G P U"),
    (r"\bGPUs\b", "G P Us"),
    (r"\bCPU\b", "C P U"),
    (r"\bCPUs\b", "C P Us"),

    # ── Math notation ──────────────────────────────────────────────
    (r"\bSO\(3\)", "S O three"),
    (r"\bSE\(3\)", "S E three"),
    # \mathbb{R}^{n} and friends. Run BEFORE the generic R^n rule.
    (r"\\mathbb\{R\}\^\{?(\d+)\}?", lambda m: f"R to the {NUMBER_WORDS.get(m.group(1), m.group(1))}"),
    (r"\\mathbb\{R\}\^\{?n\}?", "R to the n"),
    (r"\bR\^\{?(\d+)\}?", lambda m: f"R to the {NUMBER_WORDS.get(m.group(1), m.group(1))}"),
    (r"\bR\^\{?n\}?", "R to the n"),
    # J(q), J(q̇), f(x), etc. — only spell out the most book-specific.
    (r"\bJ\(q\)", "J of q"),
    (r"q̇|\\dot\s*q", "q dot"),
    (r"\b7-DoF\b", "seven D O F"),
    (r"\b6-DoF\b", "six D O F"),
    (r"\bDoF\b", "D O F"),

    # ── Citation noise: arXiv IDs and parenthetical IDs ─────────────
    # "(Kim et al., 2024, arXiv:2406.09246)" → "(Kim et al., 2024)"
    (r",\s*arXiv:\d{4}\.\d{4,5}\b", ""),
    (r",\s*archive:\d{4}\.\d{4,5}\b", ""),  # in case arXiv→archive ran first
    # Bare arXiv IDs sprinkled in prose
    (r"\barXiv:\d{4}\.\d{4,5}\b", ""),
    (r"\barchive:\d{4}\.\d{4,5}\b", ""),
    (r"\b\d{4}\.\d{4,5}\b", ""),

    # ── URLs ───────────────────────────────────────────────────────
    (r"https?://\S+", ""),
    (r"www\.\S+", ""),
]

NUMBER_WORDS: dict[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve",
}

# ─────────────────────────────────────────────────────────────────────────────
# Markdown structural cleanup
# ─────────────────────────────────────────────────────────────────────────────

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)
_INDENTED_CODE_RE = re.compile(r"(?:^|\n)((?: {4}|\t)[^\n]+\n?)+")
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$\n]+?)\$")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_SIDENOTE_REF_RE = re.compile(r"\[(?:\d+|ref-\d+)\]")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_([^_\n]+)_(?!_)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ─────────────────────────────────────────────────────────────────────────────
# Minimal inline-math → spoken text
# ─────────────────────────────────────────────────────────────────────────────
#
# Conservative: if we can't read the expression with high confidence, drop it.
# The table below covers the math that actually appears in chapters 1–4.

_MATH_TOKEN_RULES: list[tuple[str, str]] = [
    (r"\\pi_0\b", "pi zero"),
    (r"\\pi\b", "pi"),
    (r"\\theta\b", "theta"),
    (r"\\phi\b", "phi"),
    (r"\\alpha\b", "alpha"),
    (r"\\beta\b", "beta"),
    (r"\\gamma\b", "gamma"),
    (r"\\lambda\b", "lambda"),
    (r"\\mu\b", "mu"),
    (r"\\sigma\b", "sigma"),
    (r"\\epsilon\b", "epsilon"),
    (r"\\tau\b", "tau"),
    (r"\\nabla\b", "gradient"),
    (r"\\partial\b", "partial"),
    (r"\\sum\b", "sum"),
    (r"\\int\b", "integral"),
    (r"\\mathbb\{R\}", "R"),
    (r"\\mathbb\{E\}", "expectation"),
    (r"\\mathcal\{N\}", "normal"),
    (r"\\mathrm\{(\w+)\}", lambda m: m.group(1)),
    (r"\\text\{([^}]+)\}", lambda m: m.group(1)),
    (r"\\dot\s*\{?(\w+)\}?", lambda m: f"{m.group(1)} dot"),
    (r"\\hat\s*\{?(\w+)\}?", lambda m: f"{m.group(1)} hat"),
    (r"\\bar\s*\{?(\w+)\}?", lambda m: f"{m.group(1)} bar"),
    (r"\^\{?2\}?", " squared"),
    (r"\^\{?3\}?", " cubed"),
    (r"\^\{?T\}?", " transpose"),
    (r"\^\{?-1\}?", " inverse"),
    (r"\^\{?n\}?", " to the n"),
    (r"_\{?0\}?", " zero"),
    (r"_\{?t\}?", " t"),
    (r"\\cdot\b", " times "),
    (r"\\times\b", " times "),
    (r"\\to\b", " to "),
    (r"\\leq\b", " less than or equal to "),
    (r"\\geq\b", " greater than or equal to "),
    (r"\\neq\b", " not equal to "),
    (r"\\approx\b", " approximately "),
    (r"\\sim\b", " distributed as "),
    (r"\\in\b", " in "),
    (r"\\rightarrow\b", " to "),
    (r"\\leftarrow\b", " from "),
    (r"\\,|\\;|\\:|\\!", " "),
    (r"\\\\", " "),
    (r"\\\(|\\\)|\\\[|\\\]", ""),
    (r"\{|\}", ""),
]

# Characters that, after the rules above, indicate we did NOT successfully
# render the math. Any remaining backslash, caret, underscore (between word
# chars), or vertical bar means we punt and replace with "[equation omitted]".
_MATH_UNRENDERED_RE = re.compile(r"\\[a-zA-Z]+|[\^_|]\{|\\[\(\[\]\)]")


def _render_inline_math(expr: str) -> str:
    """Best-effort plain-text render of an inline LaTeX expression.

    Returns spoken text on success, or an empty string if the expression
    contains constructs we don't know how to read. The caller falls back to
    "(equation omitted)" for empty returns inside non-trivial contexts.
    """
    text = expr.strip()
    for pat, repl in _MATH_TOKEN_RULES:
        text = re.sub(pat, repl, text)
    if _MATH_UNRENDERED_RE.search(text):
        return ""
    # Collapse any stray whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    text: str
    char_count: int
    word_count: int


def preprocess_markdown(body: str) -> PreprocessResult:
    """Convert authored markdown to clean spoken prose.

    Steps (in order):
        1. Strip the leading H1 (the section title is announced by the page;
           we don't want to read it twice).
        2. Drop fenced code blocks entirely with a stage cue.
        3. Drop display math `$$...$$` with a stage cue.
        4. Render inline math `$...$` to plain words, or skip with a stage
           cue if we can't render it.
        5. Drop inline `code` (replace with the surrounding word so the
           sentence still scans).
        6. Drop sidenote references like `[1]` or `[ref-3]`.
        7. Unwrap markdown emphasis (**bold**, *italic*, _italic_).
        8. Strip links — keep the link text, drop the URL.
        9. Strip images entirely.
       10. Convert ATX headings to "Section. Title." with surrounding pauses.
       11. Apply the pronunciation substitution table.
       12. Replace em dashes with commas for natural pacing.
       13. Normalize whitespace.
    """
    text = body

    # Strip the leading H1 line(s) — there is one and only one per section.
    text = re.sub(r"^\s*#\s+[^\n]+\n+", "", text, count=1)

    # Strip fenced code blocks (```...```). They are nearly always Python or
    # YAML; reading them aloud is noise. Leave a brief stage cue so the
    # listener notices the gap.
    text = _FENCED_CODE_RE.sub("\n\n(Code block omitted from audio.)\n\n", text)
    # Strip 4-space / tab indented code blocks too.
    text = _INDENTED_CODE_RE.sub("\n\n(Code block omitted from audio.)\n\n", text)

    # Strip HTML tags (mostly <kbd>, <sub>, etc. — rare).
    text = _HTML_TAG_RE.sub("", text)

    # Display math → stage cue.
    text = _DISPLAY_MATH_RE.sub("\n\n(Equation omitted from audio.)\n\n", text)

    # Inline math → spoken or stage cue.
    def _inline_math_sub(m: re.Match[str]) -> str:
        spoken = _render_inline_math(m.group(1))
        if spoken:
            return spoken
        return "(equation omitted)"
    text = _INLINE_MATH_RE.sub(_inline_math_sub, text)

    # Inline `code` → keep the literal token but unwrap. Kokoro reads
    # `arr[0]` etc. about as well as it reads "arr 0". Drop the backticks.
    text = _INLINE_CODE_RE.sub(r"\1", text)

    # Sidenote references like `[1]`, `[ref-3]`.
    text = _SIDENOTE_REF_RE.sub("", text)

    # Images first (so the image alt text doesn't get caught by the link rule).
    text = _IMAGE_RE.sub("", text)
    # Links: keep link text, drop URL.
    text = _LINK_RE.sub(r"\1", text)

    # Emphasis: unwrap, keep text.
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _UNDERSCORE_ITALIC_RE.sub(r"\1", text)

    # Headings → spoken with a brief pause. We deliberately read "Section."
    # for every level so the listener gets the structural cue without us
    # having to mind-read whether ## is "Section" or "Subsection".
    text = _HEADING_RE.sub(lambda m: f"\n\nSection. {m.group(2)}.\n\n", text)

    # Horizontal rules: drop.
    text = _HORIZONTAL_RULE_RE.sub("", text)

    # Apply pronunciation rules.
    for pat, repl in PRONUNCIATION_RULES:
        text = re.sub(pat, repl, text, flags=re.UNICODE)

    # Em dashes → comma + space for natural pacing. Curly quotes stay.
    text = text.replace("—", ", ")
    # En dashes (used for number ranges) → "to".
    text = re.sub(r"(\d)–(\d)", r"\1 to \2", text)
    text = text.replace("–", ", ")

    # Normalize whitespace.
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    return PreprocessResult(
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


def split_into_chunks(text: str, max_chars: int = 4000) -> list[str]:
    """Split spoken text into chunks Kokoro can handle in a single pass.

    Kokoro's quality degrades on very long inputs (>~2000 chars) and the
    G2P pipeline can OOM on multi-thousand-char inputs. We split on
    paragraph boundaries when possible, falling back to sentence
    boundaries.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Single paragraph longer than max_chars → split on sentences.
        if len(para) > max_chars:
            for piece in _split_paragraph(para, max_chars):
                if buf and len(buf) + len(piece) + 2 <= max_chars:
                    buf = f"{buf}\n\n{piece}"
                else:
                    if buf:
                        chunks.append(buf)
                    buf = piece
            continue
        if buf and len(buf) + len(para) + 2 <= max_chars:
            buf = f"{buf}\n\n{para}"
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _split_paragraph(para: str, max_chars: int) -> Iterable[str]:
    """Split a single oversized paragraph on sentence boundaries."""
    # Sentence end = . ! ? optionally followed by closing quote, then a space.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if buf and len(buf) + len(sent) + 1 <= max_chars:
            buf = f"{buf} {sent}"
        else:
            if buf:
                yield buf
            # Sentence itself might exceed max_chars on rare long quotes.
            # We don't sub-split further; Kokoro will still read it.
            buf = sent
    if buf:
        yield buf
