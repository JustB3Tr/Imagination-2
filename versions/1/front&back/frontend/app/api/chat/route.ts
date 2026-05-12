import { NextRequest, NextResponse } from 'next/server'

import { buildChatRequestPayload } from '@/lib/build-chat-payload'

export const runtime = 'nodejs'

/**
 * Used only when `NEXT_PUBLIC_API_URL` is unset. Replace via env (e.g. TrueNAS Custom App)
 * with your ngrok origin after each Colab restart: `https://xxxx.ngrok-free.app`
 */
const PLACEHOLDER_API_ORIGIN = 'https://configure-next-public-api-url.invalid'

const OFFLINE_USER_MESSAGE =
  'Imagination AI is currently offline. Set NEXT_PUBLIC_API_URL to your ngrok URL and ensure the Colab backend is running.'

function resolveBackendChatUrl(): string {
  const raw = (process.env.NEXT_PUBLIC_API_URL || PLACEHOLDER_API_ORIGIN).trim()
  const base = raw.replace(/\/$/, '')
  if (base.endsWith('/api/chat')) return base
  return `${base}/api/chat`
}

/** Forward full chat body (incl. `image` / `attachments` for CLIP+projector) to the Python API. */
function buildUpstreamPayload(body: Record<string, unknown>) {
  return buildChatRequestPayload(body)
}

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>
  try {
    body = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'invalid_json', response: OFFLINE_USER_MESSAGE },
      { status: 400 }
    )
  }

  const target = resolveBackendChatUrl()
  const upstreamJson = buildUpstreamPayload(body)

  const ac = request.headers.get('accept') || ''
  const wantsStream = ac.includes('text/event-stream')

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 600_000)

  try {
    const upstream = await fetch(target, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: wantsStream ? 'text/event-stream' : 'application/json',
        // Helps some ngrok free-tier / interstitial cases on server-side fetch
        'ngrok-skip-browser-warning': '69420',
      },
      body: JSON.stringify(upstreamJson),
      signal: controller.signal,
    })

    const ct = upstream.headers.get('content-type') || ''

    if (
      wantsStream &&
      upstream.ok &&
      (ct.includes('text/event-stream') || ct.includes('application/x-ndjson'))
    ) {
      return new NextResponse(upstream.body, {
        status: upstream.status,
        headers: {
          'Content-Type': ct.includes('text/event-stream')
            ? 'text/event-stream; charset=utf-8'
            : ct,
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
      })
    }

    const text = await upstream.text()
    if (!upstream.ok) {
      let parsed: { response?: string; detail?: string } = {}
      try {
        parsed = JSON.parse(text) as { response?: string; detail?: string }
      } catch {
        /* use text */
      }
      return NextResponse.json(
        {
          error: 'upstream_error',
          response:
            typeof parsed.response === 'string'
              ? parsed.response
              : OFFLINE_USER_MESSAGE,
          detail: parsed.detail ?? text.slice(0, 500),
        },
        { status: upstream.status >= 500 ? 503 : upstream.status }
      )
    }

    try {
      const json = JSON.parse(text) as Record<string, unknown>
      return NextResponse.json(json, { status: upstream.status })
    } catch {
      return NextResponse.json(
        { response: text, model: 'Imagination 1.3' },
        { status: 200 }
      )
    }
  } catch {
    return NextResponse.json(
      { error: 'unavailable', response: OFFLINE_USER_MESSAGE },
      { status: 503 }
    )
  } finally {
    clearTimeout(timeoutId)
  }
}
