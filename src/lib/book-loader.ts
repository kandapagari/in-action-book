// Custom Astro content loader for src/content/book/chapter_NN/section_N_N.md.
// (Generated mirror of the authoring source at ../book/ — see
// tools/sync-book-to-site.sh.)
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
// known free-text fields in double quotes (escaping embedded quotes), then
// hand the content to Astro's standard markdown entry pipeline. We do NOT
// edit the source files on disk.

import { existsSync, promises as fs } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { relative } from 'node:path';
import type { Loader } from 'astro/loaders';
import { glob as tinyglobby } from 'tinyglobby';
import picomatch from 'picomatch';

const FREE_TEXT_FIELDS = ['title', 'prereqs'];

function quoteIfNeeded(value: string): string {
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

function normalizeFrontmatter(raw: string): string {
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

export function bookLoader(options: {
  base: string;
  pattern: string;
  idField?: string;
}): Loader {
  const idField = options.idField ?? 'section';
  return {
    name: 'book-loader',
    load: async ({
      config,
      logger,
      watcher,
      parseData,
      store,
      generateDigest,
      entryTypes,
    }) => {
      const baseDir = new URL(options.base, config.root);
      if (!baseDir.pathname.endsWith('/')) {
        baseDir.pathname = `${baseDir.pathname}/`;
      }
      const basePath = fileURLToPath(baseDir);
      if (!existsSync(baseDir)) {
        logger.error(`book-loader base directory does not exist: ${basePath}`);
        return;
      }

      const files = await tinyglobby(options.pattern, {
        cwd: basePath,
        expandDirectories: false,
      });
      if (files.length === 0) {
        logger.warn(`book-loader: no files matched "${options.pattern}" in ${basePath}`);
        return;
      }

      const renderByEntryType = new WeakMap<object, any>();
      const untouched = new Set(store.keys());
      const fileToId = new Map<string, string>();

      async function syncOne(entry: string, oldId?: string): Promise<void> {
        const entryType = entryTypes.get('.md');
        if (!entryType) {
          logger.error(`No entry type registered for .md`);
          return;
        }
        const fileUrl = new URL(encodeURI(entry), baseDir);
        const raw = await fs.readFile(fileUrl, 'utf8');
        const normalized = normalizeFrontmatter(raw);

        const { body, data } = await entryType.getEntryInfo({
          contents: normalized,
          fileUrl,
        });

        const id = String(data[idField]);
        if (oldId && oldId !== id) store.delete(oldId);
        untouched.delete(id);

        const digest = generateDigest(normalized);
        const filePath = fileURLToPath(fileUrl);
        const relativePath = relative(fileURLToPath(config.root), filePath);

        const existing = store.get(id);
        if (existing && existing.digest === digest && existing.filePath) {
          if (existing.deferredRender) store.addModuleImport(existing.filePath);
          fileToId.set(filePath, id);
          return;
        }

        const parsedData = await parseData({ id, data, filePath });

        if (entryType.getRenderFunction) {
          let renderFn = renderByEntryType.get(entryType);
          if (!renderFn) {
            renderFn = await entryType.getRenderFunction(config);
            renderByEntryType.set(entryType, renderFn);
          }
          let rendered: any = undefined;
          try {
            rendered = await renderFn({
              id,
              data,
              body,
              filePath,
              digest,
            });
          } catch (err: any) {
            logger.error(`book-loader: render failed for ${entry}: ${err?.message ?? err}`);
          }
          store.set({
            id,
            data: parsedData,
            body,
            filePath: relativePath,
            digest,
            rendered,
            assetImports: rendered?.metadata?.imagePaths,
          });
        } else {
          store.set({
            id,
            data: parsedData,
            body,
            filePath: relativePath,
            digest,
            deferredRender: 'contentModuleTypes' in entryType ? true : undefined,
          });
        }
        fileToId.set(filePath, id);
      }

      await Promise.all(files.map((entry) => syncOne(entry)));
      untouched.forEach((id) => store.delete(id));

      if (!watcher) return;
      watcher.add(basePath);
      const matches = (rel: string) =>
        !rel.startsWith('../') && picomatch.isMatch(rel, options.pattern);
      async function onChange(changed: string): Promise<void> {
        const rel = relative(basePath, changed);
        if (!matches(rel)) return;
        const oldId = fileToId.get(changed);
        try {
          await syncOne(rel, oldId);
          logger.info(`book-loader: reloaded ${rel}`);
        } catch (err: any) {
          logger.error(`book-loader: reload failed ${rel}: ${err?.message ?? err}`);
        }
      }
      watcher.on('change', onChange);
      watcher.on('add', onChange);
      watcher.on('unlink', async (deleted: string) => {
        const rel = relative(basePath, deleted);
        if (!matches(rel)) return;
        const id = fileToId.get(deleted);
        if (id) {
          store.delete(id);
          fileToId.delete(deleted);
        }
      });
    },
  };
}
