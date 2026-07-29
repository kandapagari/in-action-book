// Diagnostic: verify Brevo (or any) SMTP credentials from process.env authenticate,
// and optionally send one plain-text self-test. Never prints SMTP_PASS.
//   Verify only:  node scripts/newsletter/check-smtp.mjs
//   Verify+send:  node scripts/newsletter/check-smtp.mjs --send you@example.com
// Config comes from env (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, NEWSLETTER_FROM).
import nodemailer from 'nodemailer';

const { SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, NEWSLETTER_FROM } = process.env;

const missing = ['SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASS']
  .filter((k) => !process.env[k]);
if (missing.length) {
  console.error(`FAIL: missing required env var(s): ${missing.join(', ')}`);
  console.error('(Set them in .env or .env.local — do not hardcode secrets.)');
  process.exit(1);
}

const port = Number(SMTP_PORT);
const transporter = nodemailer.createTransport({
  host: SMTP_HOST,
  port,
  secure: port === 465, // 465 = implicit TLS; 587 = STARTTLS
  auth: { user: SMTP_USER, pass: SMTP_PASS },
});

console.log(`Connecting to ${SMTP_HOST}:${port} as ${SMTP_USER} (secure=${port === 465})…`);

try {
  await transporter.verify();
  console.log('PASS: SMTP connection + auth succeeded.');
} catch (err) {
  console.error(`FAIL: SMTP verify failed${err.code ? ` [${err.code}]` : ''}: ${err.message}`);
  process.exit(1);
}

const sendIdx = process.argv.indexOf('--send');
if (sendIdx !== -1) {
  const recipient = process.argv[sendIdx + 1];
  if (!recipient) {
    console.error('FAIL: --send requires a recipient email address.');
    process.exit(1);
  }
  const from = NEWSLETTER_FROM || SMTP_USER;
  try {
    const info = await transporter.sendMail({
      from,
      to: recipient,
      subject: 'Brevo SMTP test',
      text: 'Brevo SMTP test from the Action Models newsletter setup.',
    });
    console.log(`SENT: test email accepted, messageId=${info.messageId}`);
  } catch (err) {
    console.error(`FAIL: send failed${err.code ? ` [${err.code}]` : ''}: ${err.message}`);
    process.exit(1);
  }
}
