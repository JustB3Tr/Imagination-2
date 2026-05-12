import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'

const PLACEHOLDER_API_ORIGIN = 'https://configure-next-public-api-url.invalid'

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
    return NextResponse.json({ error: 'invalid_json' }, { status: 400 })
  }

  const upstream = await fetch(resolveBackendUrl('/api/agent/apply'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'ngrok-skip-browser-warning': '69420',
    },
    body: JSON.stringify(body),
  })

  const text = await upstream.text()
  return new NextResponse(text, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') || 'application/json',
    },
  })
}
