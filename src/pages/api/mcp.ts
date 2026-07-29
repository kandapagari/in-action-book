import type { APIRoute } from 'astro';
import { createMcpHandler } from '@modelcontextprotocol/server';
import { buildServer } from '../../lib/mcp-server';

// Server-rendered on Vercel as a serverless function; the rest of the site
// stays statically prerendered.
export const prerender = false;

// Stateless Streamable HTTP: the factory runs once per request, so a fresh
// McpServer serves every call — no sessions, no shared globals. createMcpHandler
// returns a web-standard { fetch(request): Response } handler, which is exactly
// what an Astro API route consumes and returns.
const handler = createMcpHandler(() => buildServer());

// Public read-only endpoint — no auth. Permissive CORS so browser-based MCP
// clients (and preflight requests) can connect.
const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
  'Access-Control-Allow-Headers':
    'Content-Type, Accept, Authorization, Mcp-Session-Id, Mcp-Protocol-Version, Last-Event-ID',
  'Access-Control-Expose-Headers': 'Mcp-Session-Id, Mcp-Protocol-Version',
  'Access-Control-Max-Age': '86400',
};

export const ALL: APIRoute = async ({ request }) => {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }
  const response = await handler.fetch(request);
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(CORS_HEADERS)) headers.set(key, value);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
};
