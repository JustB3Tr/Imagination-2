import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'

const PLACEHOLDER_API_ORIGIN = 'https://configure-next-public-api-url.invalid'
const OFFLINE_USER_MESSAGE =
  'Imagination AI is currently offline. Set NEXT_PUBLIC_API_URL to your backend URL and ensure it is running.'

function resolveBackendUrl(path: string): string {
  const raw = (process.env.NEXT_PUBLIC_API_URL || PLACEHOLDER_API_ORIGIN).trim()
  const base = raw.replace(/\/$/, '')
  return `${base}${path}`
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

  const target = resolveBackendUrl('/api/chat/agent')
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 600_000)
  try {
    const upstream = await fetch(target, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/x-ndjson',
        'ngrok-skip-browser-warning': '69420',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => '')
      return NextResponse.json(
        {
          error: 'upstream_error',
          response: OFFLINE_USER_MESSAGE,
          detail: text.slice(0, 500),
        },
        { status: upstream.status >= 500 ? 503 : upstream.status }
      )
    }

    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: {
        'Content-Type': 'application/x-ndjson; charset=utf-8',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    })
  } catch {
    return NextResponse.json(
      { error: 'unavailable', response: OFFLINE_USER_MESSAGE },
      { status: 503 }
    )
  } finally {
    clearTimeout(timeoutId)
  }
}
