// Client-side view-counter.
//
// Privacy model (mirrors `/api/views/*`):
//   - No cookies. No IPs. No localStorage of identifiers.
//   - `sessionStorage` is used purely to dedupe within the current browser
//     tab session, keyed by the counter (not by the visitor): a refresh in
//     the same tab counts ONCE. Closing the tab + reopening counts again.
//   - On every load we POST once per counter (incrementing in KV) and
//     thereafter GET only.
//   - On any error we leave the placeholder character untouched; no errors
//     are surfaced to the reader.

type SiteEl = HTMLElement & { dataset: { viewCount: 'site' } };
type SectionEl = HTMLElement & {
  dataset: { viewCount: 'section'; chapter: string; section: string };
};

const NUMBER_FORMAT = new Intl.NumberFormat('en-US');

function render(els: HTMLElement[], count: number | null | undefined): void {
  if (count == null) return; // leave the placeholder
  const text = NUMBER_FORMAT.format(count);
  for (const el of els) el.textContent = text;
}

type CountResponse = { count?: number | null; skipped?: boolean; error?: string };

async function bump(url: string, method: 'GET' | 'POST'): Promise<number | null> {
  try {
    const res = await fetch(url, { method, headers: { Accept: 'application/json' } });
    if (!res.ok) return null;
    const json = (await res.json()) as CountResponse;
    return typeof json.count === 'number' ? json.count : null;
  } catch {
    return null;
  }
}

async function processCounter(
  url: string,
  storageKey: string,
  targets: HTMLElement[],
): Promise<void> {
  if (targets.length === 0) return;

  let alreadyViewed = false;
  try {
    alreadyViewed = sessionStorage.getItem(storageKey) === '1';
  } catch {
    // sessionStorage can throw in privacy-locked contexts; treat as
    // "not viewed" but skip the marker write below to avoid noisy errors.
  }

  const method = alreadyViewed ? 'GET' : 'POST';
  const count = await bump(url, method);
  render(targets, count);

  if (!alreadyViewed) {
    try {
      sessionStorage.setItem(storageKey, '1');
    } catch {
      /* ignored — see above */
    }
  }
}

function init(): void {
  const siteEls = Array.from(
    document.querySelectorAll<SiteEl>('[data-view-count="site"]'),
  );
  void processCounter('/api/views/site', 'viewed:site', siteEls);

  const sectionEls = Array.from(
    document.querySelectorAll<SectionEl>('[data-view-count="section"]'),
  );
  // Group by (chapter, section) so a page with multiple span instances for
  // the same section still POSTs once.
  const groups = new Map<string, SectionEl[]>();
  for (const el of sectionEls) {
    const chapter = el.dataset.chapter;
    const section = el.dataset.section;
    if (!chapter || !section) continue;
    const key = `${chapter}/${section}`;
    const list = groups.get(key);
    if (list) list.push(el);
    else groups.set(key, [el]);
  }
  for (const [key, els] of groups) {
    void processCounter(
      `/api/views/section/${key}`,
      `viewed:section:${key}`,
      els,
    );
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init, { once: true });
} else {
  init();
}
