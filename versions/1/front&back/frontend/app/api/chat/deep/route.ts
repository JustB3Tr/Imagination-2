import { NextRequest } from 'next/server';

import { buildChatRequestPayload } from '@/lib/build-chat-payload';

export const runtime = 'nodejs';

const PLACEHOLDER_API_ORIGIN = 'https://configure-next-public-api-url.invalid';

/** POST → FastAPI NDJSON (`application/x-ndjson`) deep research pipeline. */
function resolveBackendDeepUrl(): string {
  const raw = (process.env.NEXT_PUBLIC_API_URL || PLACEHOLDER_API_ORIGIN).trim();
  const base = raw.replace(/\/$/, '');
  if (base.endsWith('/api/chat/deep')) return base;
  if (base.endsWith('/api/chat/stream'))
    return base.replace(/\/api\/chat\/stream$/, '/api/chat/deep');
  if (base.endsWith('/api/chat'))
    return `${base.replace(/\/api\/chat$/, '')}/api/chat/deep`;
  return `${base}/api/chat/deep`;
}

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: 'invalid_json', type: 'error' }) + '\n', {
      status: 400,
      headers: { 'Content-Type': 'application/x-ndjson' },
    });
  }

  const target = resolveBackendDeepUrl();
  const upstreamJson = buildChatRequestPayload(body);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 1_800_000);

  try {
    const response = await fetch(target, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/x-ndjson, application/json',
        'ngrok-skip-browser-warning': '69420',
      },
      body: JSON.stringify(upstreamJson),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      const text = await response.text();
      console.error('Imagination upstream deep research error:', response.status, text.slice(0, 800));
      return new Response(
        JSON.stringify({
          error: `upstream_${response.status}`,
          detail: text.slice(0, 500),
          type: 'error',
        }) + '\n',
        {
          status: response.status >= 400 && response.status < 600 ? response.status : 502,
          headers: { 'Content-Type': 'application/x-ndjson' },
        },
      );
    }

    return new Response(response.body, {
      headers: {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (backendError) {
    console.error('Deep research backend unreachable:', backendError);
    return new Response(
      JSON.stringify({
        error: 'backend_unreachable',
        detail: String(backendError),
        type: 'error',
      }) + '\n',
      { status: 502, headers: { 'Content-Type': 'application/x-ndjson' } },
    );
  } finally {
    clearTimeout(timeoutId);
  }
}
