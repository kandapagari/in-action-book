// Newsletter subscriber store + delivery. State lives in the same Upstash
// Redis used by the view counter (client reused via `getRedis`); email goes
// out through one of two selectable transports (see `sendEmail`): Brevo SMTP
// via nodemailer (the default, works without a verified domain) or Resend
// (once a sending domain is verified). Unlike the view counter, the newsletter
// must NOT degrade silently: if Redis or the selected transport's config is
// missing, the calling endpoints throw and return 500 rather than faking a
// successful signup.
//
// Redis keyspace (all keys prefixed `nl:`):
//   nl:sub:{email}   (hash)  status | cursor | token | created | confirmed
//   nl:token:{token} (string→email) confirm + unsubscribe lookup
//   nl:confirmed     (set)   confirmed emails, iterated by the weekly cron
//
// Emails are normalized to trimmed + lowercase; tokens are crypto UUIDs.

import type { Redis } from '@upstash/redis';
import { Resend } from 'resend';
import nodemailer, { type Transporter } from 'nodemailer';
import { getRedis } from './views';
import { loadProgress, sectionRouteParams } from './progress';
import { SITE } from './seo';
import {
  sendableChapters,
  nextChapterForCursor,
  buildChapterEmail,
  excerptForChapter,
} from './newsletter-content';
import type { EmailSection, BuiltEmail } from './newsletter-content';

const CONFIRMED_SET = 'nl:confirmed';
const subKey = (email: string) => `nl:sub:${email}`;
const tokenKey = (token: string) => `nl:token:${token}`;
const CONFIRM_TTL_SECONDS = 60 * 60 * 48; // 48h to click the confirm link

export type SubStatus = 'pending' | 'confirmed' | 'unsubscribed';

type SubRecord = {
  status: SubStatus;
  cursor: number;
  token: string;
  created: string;
  confirmed?: string;
};

// Thrown for bad user input so routes can answer 400/410 instead of 500.
export class InvalidEmailError extends Error {
  constructor() {
    super('invalid_email');
    this.name = 'InvalidEmailError';
  }
}
export class InvalidTokenError extends Error {
  constructor() {
    super('invalid_or_expired_token');
    this.name = 'InvalidTokenError';
  }
}

// ---------------------------------------------------------------------------
// Config + delivery
// ---------------------------------------------------------------------------

export type NewsletterConfig = { from: string; siteUrl: string };

// ponytail: `EMAIL_TRANSPORT` selects between two delivery paths that are BOTH
// kept intentionally (explicit user request): `smtp` (Brevo via nodemailer,
// the default — sends without owning a domain) and `resend` (once a sending
// domain is verified). No auto-fallback between them: missing config for the
// selected transport is a real error, not a reason to try the other.
export type EmailTransport = 'smtp' | 'resend';

function emailTransport(): EmailTransport {
  const t = process.env.EMAIL_TRANSPORT ?? 'smtp';
  if (t !== 'smtp' && t !== 'resend') {
    throw new Error(
      `newsletter_config_invalid: EMAIL_TRANSPORT must be 'smtp' or 'resend', got '${t}'`,
    );
  }
  return t;
}

// Read + validate the SMTP transport's env. A missing/invalid var is a real
// error (no silent fallback), so this throws rather than returning partial config.
function smtpEnv(): { host: string; port: number; user: string; pass: string } {
  const host = process.env.SMTP_HOST;
  const portRaw = process.env.SMTP_PORT;
  const user = process.env.SMTP_USER;
  const pass = process.env.SMTP_PASS;
  const missing = (
    [
      ['SMTP_HOST', host],
      ['SMTP_PORT', portRaw],
      ['SMTP_USER', user],
      ['SMTP_PASS', pass],
    ] as const
  )
    .filter(([, v]) => !v)
    .map(([k]) => k);
  if (missing.length) {
    throw new Error(`newsletter_config_missing: EMAIL_TRANSPORT=smtp requires ${missing.join(', ')}`);
  }
  const port = Number(portRaw);
  if (!Number.isFinite(port)) {
    throw new Error(`newsletter_config_invalid: SMTP_PORT must be a number, got '${portRaw}'`);
  }
  return { host: host!, port, user: user!, pass: pass! };
}

function resendEnv(): { apiKey: string } {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    throw new Error('newsletter_config_missing: EMAIL_TRANSPORT=resend requires RESEND_API_KEY');
  }
  return { apiKey };
}

export function newsletterConfig(): NewsletterConfig {
  const from = process.env.NEWSLETTER_FROM;
  if (!from) throw new Error('newsletter_config_missing: NEWSLETTER_FROM is not set');
  // Validate only the selected transport's secrets (fail fast for callers).
  if (emailTransport() === 'smtp') smtpEnv();
  else resendEnv();
  const siteUrl = (process.env.SITE_URL ?? SITE).replace(/\/+$/, '');
  return { from, siteUrl };
}

// Lazy module-level SMTP transporter singleton (mirrors the Redis client),
// built on first send from `smtpEnv()`. Brevo uses STARTTLS on 587 and
// implicit TLS on 465, hence `secure: port === 465`.
let smtpTransporter: Transporter | null = null;
function getSmtpTransporter(): Transporter {
  if (smtpTransporter) return smtpTransporter;
  const { host, port, user, pass } = smtpEnv();
  smtpTransporter = nodemailer.createTransport({
    host,
    port,
    secure: port === 465,
    auth: { user, pass },
  });
  return smtpTransporter;
}

// Single delivery seam both the confirmation and chapter emails go through.
// Picks the transport from `EMAIL_TRANSPORT`; throws (never falls back) when
// the selected transport is misconfigured or a send fails.
async function sendEmail(msg: {
  to: string;
  subject: string;
  html: string;
  text: string;
}): Promise<void> {
  const from = process.env.NEWSLETTER_FROM;
  if (!from) throw new Error('newsletter_config_missing: NEWSLETTER_FROM is not set');

  if (emailTransport() === 'smtp') {
    await getSmtpTransporter().sendMail({
      from,
      to: msg.to,
      subject: msg.subject,
      html: msg.html,
      text: msg.text,
    });
    return;
  }

  const { apiKey } = resendEnv();
  const { error } = await new Resend(apiKey).emails.send({
    from,
    to: msg.to,
    subject: msg.subject,
    html: msg.html,
    text: msg.text,
  });
  if (error) {
    throw new Error(`resend_send_failed: ${error.name ?? ''} ${error.message ?? ''}`.trim());
  }
}

// ---------------------------------------------------------------------------
// Email validation
// ---------------------------------------------------------------------------

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function isValidEmail(email: string): boolean {
  return email.length <= 254 && EMAIL_RE.test(email);
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

async function readSub(redis: Redis, email: string): Promise<SubRecord | null> {
  const h = await redis.hgetall<Record<string, string | number>>(subKey(email));
  if (!h || Object.keys(h).length === 0) return null;
  return {
    status: (h.status as SubStatus) ?? 'pending',
    cursor: Number(h.cursor ?? 0) || 0,
    token: String(h.token ?? ''),
    created: String(h.created ?? ''),
    confirmed: h.confirmed !== undefined ? String(h.confirmed) : undefined,
  };
}

// Upsert a pending subscriber and email them a confirmation link. Idempotent:
// an already-confirmed address is left alone; a still-valid pending token is
// reused (dedupe) instead of minting a second one.
export async function requestConfirmation(
  rawEmail: string,
): Promise<{ status: 'sent' | 'already_confirmed' }> {
  const email = normalizeEmail(rawEmail);
  if (!isValidEmail(email)) throw new InvalidEmailError();

  const { siteUrl } = newsletterConfig(); // fail fast if delivery unconfigured
  const redis = getRedis();

  const existing = await readSub(redis, email);
  if (existing?.status === 'confirmed') return { status: 'already_confirmed' };

  let token = existing?.token ?? '';
  const tokenResolves = token ? (await redis.get<string>(tokenKey(token))) === email : false;
  if (!tokenResolves) token = crypto.randomUUID();

  const now = new Date().toISOString();
  await redis.hset(subKey(email), {
    status: 'pending',
    cursor: existing?.cursor ?? 0,
    token,
    created: existing?.created ?? now,
  });
  await redis.set(tokenKey(token), email, { ex: CONFIRM_TTL_SECONDS });

  const confirmUrl = `${siteUrl}/api/newsletter/confirm?token=${token}`;
  await sendEmail({ to: email, ...buildConfirmEmail(siteUrl, confirmUrl) });
  return { status: 'sent' };
}

// Confirm a pending subscriber, add them to the cron set, persist the token
// (now the permanent unsubscribe handle), and send chapter 1 immediately.
export async function confirmSubscription(token: string): Promise<{ email: string }> {
  const redis = getRedis();
  const email = await redis.get<string>(tokenKey(token));
  if (!email) throw new InvalidTokenError();

  await redis.hset(subKey(email), {
    status: 'confirmed',
    confirmed: new Date().toISOString(),
  });
  await redis.sadd(CONFIRMED_SET, email);
  await redis.persist(tokenKey(token)); // token outlives the 48h TTL for unsubscribe

  await sendNextChapter(email); // fresh cursor=0 → chapter 1, then advances
  return { email };
}

export async function unsubscribe(token: string): Promise<{ email: string }> {
  const redis = getRedis();
  const email = await redis.get<string>(tokenKey(token));
  if (!email) throw new InvalidTokenError();

  await redis.hset(subKey(email), { status: 'unsubscribed' });
  await redis.srem(CONFIRMED_SET, email);
  return { email };
}

// Send the next complete chapter past the subscriber's cursor, then advance
// it. Shared by confirm (first send) and the weekly cron. A caught-up or
// non-confirmed subscriber is a no-op.
export async function sendNextChapter(
  email: string,
): Promise<{ sent: boolean; chapter: number | null }> {
  const { siteUrl } = newsletterConfig();
  const redis = getRedis();

  const sub = await readSub(redis, email);
  if (!sub || sub.status !== 'confirmed') return { sent: false, chapter: null };

  const next = nextChapterForCursor(sendableChapters(loadProgress()), sub.cursor);
  if (next === null) return { sent: false, chapter: null };

  const unsubUrl = `${siteUrl}/api/newsletter/unsubscribe?token=${sub.token}`;
  await sendEmail({ to: email, ...assembleChapterEmail(next, siteUrl, unsubUrl) });

  // ponytail: `cursor` is a high-water-mark by chapter number. If an earlier
  // chapter completes after a subscriber has passed its number, it will not
  // backfill. Chapters complete in order today, so this holds. Upgrade path:
  // track a set of sent chapter numbers instead of a single maximum.
  await redis.hset(subKey(email), { cursor: next });
  return { sent: true, chapter: next };
}

export async function runCron(): Promise<{ processed: number; sent: number }> {
  newsletterConfig(); // fail fast if delivery unconfigured
  const redis = getRedis();
  const emails = await redis.smembers(CONFIRMED_SET);

  let sent = 0;
  // ponytail: sends serially within one cron invocation. Fine for a small
  // list inside the Vercel function timeout. Upgrade path: Resend batch API /
  // pagination when the list outgrows a single invocation.
  for (const email of emails) {
    const res = await sendNextChapter(email);
    if (res.sent) sent++;
  }
  return { processed: emails.length, sent };
}

// ---------------------------------------------------------------------------
// Content assembly (bridges progress + astro:content into the pure builder)
// ---------------------------------------------------------------------------

function assembleChapterEmail(
  chapterNumber: number,
  siteUrl: string,
  unsubUrl: string,
): BuiltEmail {
  const chapter = loadProgress()
    .parts.flatMap((p) => p.chapters)
    .find((c) => c.chapter === chapterNumber);
  if (!chapter) throw new Error(`newsletter: chapter ${chapterNumber} not found in progress`);

  // The sendable gate guarantees an excerpt exists; a missing one here is a
  // real inconsistency, not something to paper over with an empty teaser.
  const excerpt = excerptForChapter(chapterNumber);
  if (!excerpt) {
    throw new Error(`newsletter: chapter ${chapterNumber} is sendable but has no excerpt artifact`);
  }

  const sections: EmailSection[] = chapter.sections.map((s) => {
    const params = sectionRouteParams(s.section);
    return {
      id: s.section,
      title: s.title,
      url: `${siteUrl}/chapters/${params.chapter}/${params.section}/`,
    };
  });

  return buildChapterEmail(
    { number: chapter.chapter, title: chapter.title, excerpt },
    sections,
    siteUrl,
    unsubUrl,
  );
}

// Small double-opt-in confirmation email. No user-supplied content is
// interpolated (only our own URLs), so no escaping is needed here.
function buildConfirmEmail(siteUrl: string, confirmUrl: string): BuiltEmail {
  const serif = "Georgia, 'Iowan Old Style', 'Palatino Linotype', Palatino, serif";
  const html = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /></head>
<body style="margin:0;padding:0;background:#fffff8;">
  <div style="max-width:34rem;margin:0 auto;padding:2.5rem 1.5rem;background:#fffff8;color:#14110d;font-family:${serif};font-size:17px;line-height:1.6;">
    <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#5b5b58;margin:0 0 1.2em;">Action Models for Robot Learning</p>
    <h1 style="font-weight:400;font-size:24px;line-height:1.25;margin:0 0 .8em;">Confirm your subscription</h1>
    <p style="margin:0 0 1.4em;">You asked to receive one complete chapter a week as the book is drafted openly. Confirm the address below and chapter&nbsp;1 arrives right away.</p>
    <p style="margin:0 0 1.6em;"><a href="${confirmUrl}" style="display:inline-block;border:1px solid #14110d;padding:.55em 1.1em;color:#14110d;text-decoration:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;letter-spacing:.03em;">Confirm subscription →</a></p>
    <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;line-height:1.5;color:#5b5b58;margin:0;">If you didn't request this, ignore this email and nothing happens — you won't be added. The link expires in 48&nbsp;hours.<br /><a href="${siteUrl}/" style="color:#5b5b58;">${siteUrl.replace(/^https?:\/\//, '')}</a></p>
  </div>
</body>
</html>`;
  const text = `Confirm your subscription — Action Models for Robot Learning

You asked to receive one complete chapter a week as the book is drafted openly.
Confirm the address below and chapter 1 arrives right away:

${confirmUrl}

If you didn't request this, ignore this email and nothing happens — you won't
be added. The link expires in 48 hours.

${siteUrl}/
`;
  return { subject: 'Confirm your subscription — Action Models for Robot Learning', html, text };
}
