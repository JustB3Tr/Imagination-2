'use client';

import type { AgentTraceEntry, WriteFileStreamState } from '@/lib/types';
import { cn } from '@/lib/utils';
import {
  AlertCircle,
  CheckCircle2,
  LayoutTemplate,
  MessageSquareText,
  Sparkles,
  Wrench,
  XCircle,
} from 'lucide-react';
import { LiveScriptEditorCard } from './live-script-editor-card';

function toolCallEyebrow(name: string): string {
  const n = (name || '').toLowerCase();
  if (n === 'run_shell') return 'Running shell';
  if (n === 'write_file') return 'Writing file';
  if (n === 'read_file') return 'Reading file';
  if (n === 'web_search') return 'Searching the web';
  if (n === 'capture_ui') return 'Capturing UI';
  return `Tool · ${name || '?'}`;
}

function toolResultEyebrow(name: string, ok?: boolean): string {
  const n = (name || '').toLowerCase();
  const tag = ok ? 'complete' : 'failed';
  if (n === 'run_shell') return `Shell ${tag}`;
  if (n === 'write_file') return `File ${tag}`;
  if (n === 'read_file') return `Read ${tag}`;
  if (n === 'web_search') return `Search ${tag}`;
  if (n === 'capture_ui') return `Capture ${tag}`;
  return `Result · ${name || '?'} (${tag})`;
}

function KindIcon({ kind, ok }: { kind: AgentTraceEntry['kind']; ok?: boolean }) {
  if (kind === 'thought')
    return <Sparkles className="h-3.5 w-3.5 shrink-0 text-muted-foreground/80" aria-hidden />;
  if (kind === 'error') return <AlertCircle className="h-3.5 w-3.5 shrink-0 text-destructive" aria-hidden />;
  if (kind === 'final' || kind === 'ui_artisan_final')
    return <MessageSquareText className="h-3.5 w-3.5 shrink-0 text-primary" aria-hidden />;
  if (kind === 'ui_artisan_revision')
    return <LayoutTemplate className="h-3.5 w-3.5 shrink-0 text-sky-500/90" aria-hidden />;
  if (kind === 'tool_call')
    return <Wrench className="h-3.5 w-3.5 shrink-0 text-muted-foreground/70" aria-hidden />;
  if (ok) return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-500/80" aria-hidden />;
  return <XCircle className="h-3.5 w-3.5 shrink-0 text-amber-500/80" aria-hidden />;
}

export function AgentTracePanel({
  entries,
  scriptPreview,
  isAgentRunning = false,
}: {
  entries: AgentTraceEntry[];
  scriptPreview: WriteFileStreamState | null;
  isAgentRunning?: boolean;
}) {
  if (!entries.length && !scriptPreview && !isAgentRunning) return null;

  return (
    <div className="rounded-xl border border-border/60 bg-muted/10 p-3 shadow-sm">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground/90">
        Agent trace
      </p>
      {entries.length === 0 && isAgentRunning ? (
        <div className="rounded-lg border border-dashed border-border/60 bg-muted/5 px-3 py-4 text-center text-[11px] text-muted-foreground">
          Agent running… events will appear here as they stream in.
        </div>
      ) : null}

      {entries.length > 0 ? (
        <ul className="max-h-[min(40vh,22rem)] space-y-1.5 overflow-y-auto overscroll-contain pr-1 [scrollbar-gutter:stable]">
          {entries.map(e => (
            <li
              key={e.id}
              className={cn(
                'agent-trace-row-in rounded-lg border px-2.5 py-2 text-xs leading-relaxed',
                e.kind === 'thought' &&
                  'border-transparent bg-muted/30 text-muted-foreground shadow-none',
                e.kind === 'tool_call' &&
                  'border-muted/25 bg-muted/25 text-muted-foreground shadow-none',
                e.kind === 'tool_result' &&
                  'border-muted/20 bg-muted/15 text-muted-foreground shadow-none',
                e.kind === 'error' && 'border-destructive/40 bg-destructive/5 text-foreground',
                (e.kind === 'final' || e.kind === 'ui_artisan_final') &&
                  'border-primary/30 bg-primary/5 text-foreground',
                e.kind === 'ui_artisan_revision' && 'border-sky-500/25 bg-sky-500/5 text-foreground'
              )}
            >
              <div className="flex items-start gap-2">
                <KindIcon kind={e.kind} ok={e.ok} />
                <div className="min-w-0 flex-1 space-y-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px]">
                    <span
                      className={cn(
                        'font-medium tracking-wide',
                        e.kind === 'thought' && 'text-muted-foreground',
                        e.kind === 'tool_call' && 'text-muted-foreground',
                        e.kind === 'tool_result' && 'text-muted-foreground',
                        e.kind === 'error' && 'uppercase text-destructive',
                        (e.kind === 'final' || e.kind === 'ui_artisan_final') && 'text-primary',
                        e.kind === 'ui_artisan_revision' && 'text-sky-600 dark:text-sky-400'
                      )}
                    >
                      {e.kind === 'thought'
                        ? 'Thinking'
                        : e.kind === 'error'
                          ? 'Error'
                          : e.kind === 'final'
                            ? 'Final answer'
                            : e.kind === 'ui_artisan_final'
                              ? 'Visual review'
                              : e.kind === 'ui_artisan_revision'
                                ? 'UI revision'
                                : e.kind === 'tool_call'
                                  ? toolCallEyebrow(e.name || '')
                                  : toolResultEyebrow(e.name || '', e.ok)}
                    </span>
                    {e.callId ? (
                      <span className="font-mono text-[10px] text-muted-foreground/50" title={e.callId}>
                        {e.callId.slice(0, 8)}…
                      </span>
                    ) : null}
                  </div>
                  {e.text ? (
                    <p className="whitespace-pre-wrap text-[11px] text-foreground/75">{e.text}</p>
                  ) : null}
                  {e.detail && !(e.kind === 'tool_call' && (e.name || '').toLowerCase() === 'write_file') ? (
                    <pre className="max-h-28 overflow-auto rounded-md bg-black/20 p-2 font-mono text-[10px] text-muted-foreground/90">
                      {e.detail}
                    </pre>
                  ) : null}
                </div>
              </div>
            </li>
          ))}
        </ul>
      ) : null}

      {scriptPreview ? <LiveScriptEditorCard state={scriptPreview} /> : null}
    </div>
  );
}
