'use client';

import { CheckCircle2, LoaderCircle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { TerminalRun } from '@/lib/types';

interface TerminalCardProps {
  run: TerminalRun;
}

function statusUi(status: TerminalRun['status']) {
  if (status === 'running') {
    return {
      label: 'Running',
      icon: <LoaderCircle className="w-3.5 h-3.5 animate-spin" />,
      className: 'text-amber-300 border-amber-300/40 bg-amber-400/10',
    };
  }
  if (status === 'success') {
    return {
      label: 'Success',
      icon: <CheckCircle2 className="w-3.5 h-3.5" />,
      className: 'text-emerald-300 border-emerald-300/40 bg-emerald-400/10',
    };
  }
  return {
    label: 'Fail',
    icon: <XCircle className="w-3.5 h-3.5" />,
    className: 'text-red-300 border-red-300/40 bg-red-400/10',
  };
}

export function TerminalCard({ run }: TerminalCardProps) {
  const status = statusUi(run.status);
  const output = [run.stdout, run.stderr].filter(Boolean).join('\n');

  return (
    <div className="rounded-xl border border-border bg-card p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span
          className={cn(
            'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium',
            status.className
          )}
        >
          {status.icon}
          {status.label}
        </span>
        {typeof run.exitCode === 'number' && (
          <span className="text-[11px] text-muted-foreground">exit {run.exitCode}</span>
        )}
      </div>
      <pre className="whitespace-pre-wrap break-words rounded-md bg-black/70 p-2 text-xs text-zinc-100">
{`$ ${run.command}
${output || '(no output)'}`}
      </pre>
    </div>
  );
}
