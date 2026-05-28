// Builds the full favicon / app-icon / Open Graph asset set for the site.
//
//   node scripts/build-icons.mjs
//
// Inputs : ET Book (roman + display italic) and Inter-Regular TTFs in assets/fonts/.
// Outputs: site/public/{favicon.svg, favicon.ico, apple-touch-icon.png,
//          icon-192.png, icon-512.png, icon-maskable-512.png, og-image.png,
//          manifest.webmanifest}
//
// Text is converted to SVG paths via opentype.js so rasterization is fully
// font-independent (sharp/librsvg never needs to find ET Book on the system).

import opentype from 'opentype.js';
import sharp from 'sharp';
import toIco from 'to-ico';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SITE_ROOT = path.resolve(__dirname, '..');
const FONT_DIR = path.join(SITE_ROOT, 'assets/fonts');
const PUBLIC_DIR = path.join(SITE_ROOT, 'public');

// ---- Palette (matches site/src/styles/global.css design tokens) ------------
const INK_LIGHT = '#111111';
const PAPER_LIGHT = '#fffff8';
const INK_DARK = '#e8e6df';
const PAPER_DARK = '#14110d';
const MUTED = '#5b5b58';
const BRICK = '#a02c2c';

// Flattened single-color letterform used for raster outputs (favicon.ico,
// apple-touch-icon.png, icon-192.png, icon-512.png). PNG/ICO can't react to
// prefers-color-scheme, so we pick a color that reads on both light and dark
// browser-tab chromes. Brick-red matches the site accent and has usable
// contrast on white tabs AND on dark tabs.
const LETTER_COLOR = '#a02c2c';

// ---- Font loading ----------------------------------------------------------
const ROMAN = await opentype.load(path.join(FONT_DIR, 'et-book-roman-line-figures.ttf'));
const ITALIC = await opentype.load(path.join(FONT_DIR, 'et-book-display-italic-old-style-figures.ttf'));
const SANS = await opentype.load(path.join(FONT_DIR, 'Inter-Regular.ttf'));

// ---- Icon SVG construction -------------------------------------------------

// Place a single glyph centered (optically) inside a square viewBox.
//   paddingRatio        — fraction of viewBox kept clear on every side
//   opticalUpShift      — fraction of viewBox to shift the glyph up so a
//                         triangular cap-A doesn't feel bottom-heavy
function placeGlyph(font, char, viewBox, paddingRatio, opticalUpShift = 0) {
  const fontSize = 1000;
  const p = font.getPath(char, 0, 0, fontSize);
  const bb = p.getBoundingBox();
  const glyphW = bb.x2 - bb.x1;
  const glyphH = bb.y2 - bb.y1;
  const inner = viewBox * (1 - 2 * paddingRatio);
  const scale = Math.min(inner / glyphW, inner / glyphH);
  const cx = viewBox / 2;
  const cy = viewBox / 2 - viewBox * opticalUpShift;
  const tx = cx - scale * (bb.x1 + glyphW / 2);
  const ty = cy - scale * (bb.y1 + glyphH / 2);
  return {
    d: p.toPathData(3),
    transform: `translate(${tx.toFixed(3)} ${ty.toFixed(3)}) scale(${scale.toFixed(6)})`,
  };
}

// Adaptive favicon: transparent background, letterform color swaps via the
// embedded @media (prefers-color-scheme: dark) inside the file. The browser
// composites the SVG against whatever the tab chrome is, so we don't paint a
// background — that lets the tab's own color show through.
//
// For a triangular cap-A the visual weight is concentrated at the wide
// base. Pushing the glyph down a hair (negative optical shift) gives the
// base a little more room and the apex a touch more sky above it, which
// reads as balanced rather than bottom-heavy.
function buildFaviconSVGAdaptive() {
  const size = 256;
  const { d, transform } = placeGlyph(ROMAN, 'A', size, 0.18, -0.015);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">
  <style>
    .ink { fill: ${INK_LIGHT}; }
    @media (prefers-color-scheme: dark) {
      .ink { fill: ${INK_DARK}; }
    }
  </style>
  <g transform="${transform}"><path class="ink" d="${d}"/></g>
</svg>
`;
}

// Static SVG used as source for raster PNG / ICO outputs. Transparent
// background, single-color letterform (LETTER_COLOR) — these are flattened
// bitmaps, so no theme adaptation here.
function buildFaviconSVGStatic({ size = 512, padding = 0.18, opticalUpShift = -0.015 } = {}) {
  const { d, transform } = placeGlyph(ROMAN, 'A', size, padding, opticalUpShift);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">
  <g transform="${transform}"><path fill="${LETTER_COLOR}" d="${d}"/></g>
</svg>
`;
}

// Maskable: glyph kept inside the safe-zone circle (~80% diameter), so we use
// larger padding so the letterform doesn't get cropped on any platform mask.
function buildMaskableSVG() {
  const size = 512;
  const { d, transform } = placeGlyph(ROMAN, 'A', size, 0.26, -0.015);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">
  <rect width="${size}" height="${size}" fill="${PAPER_LIGHT}"/>
  <g transform="${transform}"><path fill="${INK_LIGHT}" d="${d}"/></g>
</svg>
`;
}

// ---- OG image construction -------------------------------------------------

// Render a sequence of (font, text) segments as SVG paths on a single
// baseline, centered horizontally on `W`. Returns a string of <path> tags.
function renderLine(segments, fontSize, baselineY, color, W) {
  let totalAdvance = 0;
  for (const s of segments) {
    totalAdvance += s.font.getAdvanceWidth(s.text, fontSize);
  }
  let x = (W - totalAdvance) / 2;
  const parts = [];
  for (const s of segments) {
    const p = s.font.getPath(s.text, x, baselineY, fontSize);
    parts.push(`<path fill="${color}" d="${p.toPathData(3)}"/>`);
    x += s.font.getAdvanceWidth(s.text, fontSize);
  }
  return parts.join('\n  ');
}

function buildOGSvg() {
  const W = 1200;
  const H = 630;

  // Type sizes
  const headFS = 92;
  const headLeading = headFS * 1.15;
  const bylineFS = 34;
  const taglineFS = 24;

  // Baselines.  Headline block visually centered around y=H*0.45;
  // tagline anchored at the bottom edge with a comfortable margin.
  const line1Baseline = H * 0.38;
  const line2Baseline = line1Baseline + headLeading;
  const bylineBaseline = line2Baseline + headFS * 1.45;
  const taglineBaseline = H - 56;

  const line1 = renderLine(
    [{ font: ROMAN, text: 'Action Models' }],
    headFS, line1Baseline, INK_LIGHT, W,
  );
  const line2 = renderLine(
    [
      { font: ITALIC, text: 'for' },
      { font: ROMAN, text: ' Robot Learning' },
    ],
    headFS, line2Baseline, INK_LIGHT, W,
  );
  const byline = renderLine(
    [{ font: ITALIC, text: 'By Pavan Kumar Kandapagari' }],
    bylineFS, bylineBaseline, MUTED, W,
  );
  const tagline = renderLine(
    [{ font: SANS, text: 'An open-access textbook' }],
    taglineFS, taglineBaseline, BRICK, W,
  );

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
  <rect width="${W}" height="${H}" fill="${PAPER_LIGHT}"/>
  ${line1}
  ${line2}
  ${byline}
  ${tagline}
</svg>
`;
}

// ---- Rasterization helpers -------------------------------------------------

// Render an SVG buffer to a PNG of exact (w, h) pixels. We feed sharp at a
// high density so the internal librsvg rasterizes generously and then we
// downsample with lanczos for clean antialiased edges. The resize background
// is explicitly transparent so the alpha channel of the source SVG (e.g.
// favicon raster sources, which now have no background rect) is preserved.
async function rasterize(svgString, w, h) {
  const buf = Buffer.from(svgString);
  return sharp(buf, { density: 384 })
    .resize(w, h, {
      fit: 'fill',
      kernel: 'lanczos3',
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

// ---- Main ------------------------------------------------------------------
async function main() {
  await fs.mkdir(PUBLIC_DIR, { recursive: true });

  // 1. Adaptive favicon.svg (light/dark via embedded @media).
  const faviconSvg = buildFaviconSVGAdaptive();
  await fs.writeFile(path.join(PUBLIC_DIR, 'favicon.svg'), faviconSvg);

  // 2. Static SVG source for the transparent / brick-red raster outputs.
  const staticSvg = buildFaviconSVGStatic({ size: 1024 });

  // 3. PNG icons (transparent background, brick-red letterform).
  const [apple, i192, i512] = await Promise.all([
    rasterize(staticSvg, 180, 180),
    rasterize(staticSvg, 192, 192),
    rasterize(staticSvg, 512, 512),
  ]);
  await fs.writeFile(path.join(PUBLIC_DIR, 'apple-touch-icon.png'), apple);
  await fs.writeFile(path.join(PUBLIC_DIR, 'icon-192.png'), i192);
  await fs.writeFile(path.join(PUBLIC_DIR, 'icon-512.png'), i512);

  // 4. Multi-resolution favicon.ico (16/32/48).
  const [i16, i32, i48] = await Promise.all([
    rasterize(staticSvg, 16, 16),
    rasterize(staticSvg, 32, 32),
    rasterize(staticSvg, 48, 48),
  ]);
  const ico = await toIco([i16, i32, i48]);
  await fs.writeFile(path.join(PUBLIC_DIR, 'favicon.ico'), ico);

  // icon-maskable-512.png and og-image.png are intentionally NOT regenerated
  // here. The maskable icon needs an opaque cream background (PWA spec) and
  // the OG card needs an opaque cream background (social platforms render
  // it on varied surfaces). Both have already been produced and committed
  // by the previous run of this script via buildMaskableSVG() / buildOGSvg();
  // those helpers are kept defined above so a future run can re-enable them
  // when the design system changes, but for the transparent-background pass
  // we leave the two files on disk untouched.

  // 6. PWA manifest.
  const manifest = {
    name: 'Action Models for Robot Learning',
    short_name: 'Action Models',
    description:
      'An open-access textbook on action models and vision-language-action policies for robot learning.',
    start_url: '/',
    display: 'minimal-ui',
    background_color: PAPER_LIGHT,
    theme_color: PAPER_LIGHT,
    icons: [
      { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
      { src: '/icon-512.png', sizes: '512x512', type: 'image/png' },
      { src: '/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
    ],
  };
  await fs.writeFile(
    path.join(PUBLIC_DIR, 'manifest.webmanifest'),
    JSON.stringify(manifest, null, 2) + '\n',
  );

  // Summary
  const written = [
    'favicon.svg',
    'favicon.ico',
    'apple-touch-icon.png',
    'icon-192.png',
    'icon-512.png',
    'manifest.webmanifest',
  ];
  console.log('Wrote:');
  for (const f of written) {
    const stat = await fs.stat(path.join(PUBLIC_DIR, f));
    console.log(`  public/${f.padEnd(28)} ${stat.size.toString().padStart(7)} bytes`);
  }
  console.log('Skipped (kept as-is on disk):');
  for (const f of ['icon-maskable-512.png', 'og-image.png']) {
    console.log(`  public/${f}`);
  }
}

await main();
