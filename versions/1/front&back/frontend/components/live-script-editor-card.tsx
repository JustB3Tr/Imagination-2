'use client';

import type { WriteFileStreamState } from '@/lib/types';
import { FileCode2 } from 'lucide-react';

export function LiveScriptEditorCard({ state }: { state: WriteFileStreamState }) {
  const shown = state.fullText.slice(0, state.revealed);

  return (
    <div className="mt-2 overflow-hidden rounded-lg border border-border/80 bg-[#0d1117] shadow-inner ring-1 ring-white/[0.06]">
      <div className="flex items-center gap-2 border-b border-white/10 bg-black/30 px-3 py-1.5">
        <FileCode2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
        <span className="min-w-0 truncate font-mono text-[11px] text-muted-foreground">
          {state.path || 'script'}
        </span>
        {!state.complete ? (
          <span className="ml-auto shrink-0 animate-pulse text-[10px] text-muted-foreground">
            Typing…
          </span>
        ) : (
          <span className="ml-auto shrink-0 text-[10px] text-emerald-400/90">Done</span>
        )}
      </div>
      <pre className="max-h-[min(50vh,28rem)] overflow-auto whitespace-pre p-3 font-mono text-[12px] leading-relaxed text-[#7ee787]">
        {shown}
        {!state.complete && state.revealed < state.fullText.length ? (
          <span
            className="inline-block h-[1.15em] w-0.5 translate-y-0.5 animate-pulse bg-[#7ee787]"
            aria-hidden
          />
        ) : null}
      </pre>
    </div>
  );
}
