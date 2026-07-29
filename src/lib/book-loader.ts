// Custom Astro content loader for src/content/book/chapter_NN/section_N_N.md.
// (Generated mirror of the authoring source at ../book/ — see
// tools/sync-book-to-site.sh.)
//
// Frontmatter normalization (wrapping free-text title/prereqs values that
// contain colons) lives in ./frontmatter.ts and is shared with book-content.ts
// so both parsers accept the same book files. We do NOT edit the source files
// on disk.

import { existsSync, promises as fs } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { relative } from 'node:path';
import type { Loader } from 'astro/loaders';
import { glob as tinyglobby } from 'tinyglobby';
import picomatch from 'picomatch';
import { normalizeFrontmatter } from './frontmatter.ts';

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
