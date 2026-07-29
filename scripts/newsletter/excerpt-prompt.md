# Newsletter excerpt authoring prompt

Run this after a chapter is **complete** (every section drafted or revised) to
produce its weekly-newsletter excerpt. The excerpt is a standalone teaser that
is emailed to subscribers, with links back to the full chapter on the site.

The drip only sends a chapter once its excerpt file exists, so authoring this
file is what releases the chapter to the mailing list.

## Output contract (must match exactly)

Save the result to:

```
src/content/newsletter/chapter-<N>.md
```

where `<N>` is the chapter number (e.g. `chapter-1.md`, `chapter-14.md`).

The file is:

```
---
chapter: <N>
subject: "<email subject line>"
---
<body>
```

- `chapter` — the integer chapter number. No quotes.
- `subject` — the email subject line, in double quotes. Concrete and specific
  to the chapter; no clickbait. A good default shape is
  `Chapter <N>: <chapter title>`, but a sharper hook is welcome.
- `<body>` — the teaser prose. **150–250 words.** One or two short paragraphs,
  separated by a blank line. **No headings, no lists, no code blocks, no
  images.** Paragraphs are split on blank lines and rendered as `<p>` in the
  email, so plain prose only.

## Voice and rules

- Academic-literary tone — the same register as the book itself. Explain, don't
  advertise.
- Real Unicode punctuation only: em dashes (—), curly quotes (" " ' '), the
  section sign (§), spelled-out subscripts where the book uses them (π₀, not
  pi0 or pi-zero). Never straight quotes or `--`.
- No fabricated claims, numbers, or results. Only describe what the chapter
  actually contains. If you are unsure a detail is in the chapter, leave it out.
- Do not name the publisher or the author's employer. Keep it about the ideas.
- End by inviting the reader to continue on the site (the email already carries
  the section links and a "read on the site" button, so a light nudge is
  enough — don't paste URLs into the body).

## Procedure

1. Read every section of the chapter under `book/chapter_<N>/section_*.md` (or
   the site mirror at `src/content/book/chapter_<N>/`). Note the chapter title
   and the through-line of its argument.
2. Draft a 150–250 word teaser that stands on its own: what question the
   chapter opens with, the idea it builds, and why it matters — without
   requiring the reader to already know the material.
3. Count the words; trim or expand to land inside 150–250.
4. Output the **full file contents** (frontmatter + body), ready to save at
   `src/content/newsletter/chapter-<N>.md`. Output nothing else.

## Filled-in example

```
---
chapter: 1
subject: "Chapter 1: What is an action model?"
---
Before a robot can pour a cup of coffee, something has to decide — thousands of
times a second — which way each joint should move next. That something is a
policy, and this chapter is about the surprisingly long lineage of ideas for
building one. We start from the plainest possible framing: an action model is a
function from what the robot senses to what the robot does. Everything that
follows — classical planners, hand-tuned controllers, value functions learned
from reward, and today's vision-language-action networks — is a different answer
to the question of how to write that function, or how to learn it.

The chapter lays out the four families the rest of the book is organized around
and shows why they are worth telling apart: they fail in different ways, they
demand different data, and they trust the world to different degrees. By the end
you'll have a vocabulary precise enough to say what π₀ and a 1980s inverse-
kinematics routine genuinely share, and where they part. It's the map we'll
keep returning to as the systems get more capable — and more opaque.

Read §1.1 onward on the site to follow the argument in full.
```
