import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'

const PLACEHOLDER_API_ORIGIN = 'https://configure-next-public-api-url.invalid'

function resolveBackendUrl(path: string): string {
  const raw = (process.env.NEXT_PUBLIC_API_URL || PLACEHOLDER_API_ORIGIN).trim()
  const base = raw.replace(/\/$/, '')
  return `${base}${path}`
}

export async function GET(request: NextRequest) {
  const sessionId = request.nextUrl.searchParams.get('session_id')
  const workspaceRoot = request.nextUrl.searchParams.get('workspace_root')
  const q = new URLSearchParams()
  if (sessionId) q.set('session_id', sessionId)
  if (workspaceRoot) q.set('workspace_root', workspaceRoot)
  const suffix = q.toString() ? `?${q.toString()}` : ''

  const upstream = await fetch(resolveBackendUrl(`/api/agent/workspace${suffix}`), {
    headers: {
      Accept: 'application/json',
      'ngrok-skip-browser-warning': '69420',
    },
    cache: 'no-store',
  })
  const text = await upstream.text()
  return new NextResponse(text, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') || 'application/json',
      'Cache-Control': 'no-store',
    },
  })
}
