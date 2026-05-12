import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'

const PLACEHOLDER_API_ORIGIN = 'https://configure-next-public-api-url.invalid'

function resolveBackendUrl(path: string): string {
  const raw = (process.env.NEXT_PUBLIC_API_URL || PLACEHOLDER_API_ORIGIN).trim()
  const base = raw.replace(/\/$/, '')
  return `${base}${path}`
}

type Params = { params: { sessionId: string; artifactId: string } }

export async function GET(_request: NextRequest, { params }: Params) {
  const { sessionId, artifactId } = params
  const upstream = await fetch(
    resolveBackendUrl(`/api/agent/capture/${encodeURIComponent(sessionId)}/${encodeURIComponent(artifactId)}`),
    {
      headers: {
        'ngrok-skip-browser-warning': '69420',
      },
      cache: 'no-store',
    }
  )
  if (!upstream.ok || !upstream.body) {
    return NextResponse.json({ error: 'not_found' }, { status: upstream.status })
  }
  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') || 'application/octet-stream',
      'Cache-Control': 'no-store',
    },
  })
}
