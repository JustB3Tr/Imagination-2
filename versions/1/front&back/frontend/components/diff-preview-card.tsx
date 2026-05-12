'use client';

import { Button } from '@/components/ui/button';
import type { DiffProposal } from '@/lib/types';

interface DiffPreviewCardProps {
  proposal: DiffProposal;
  onApply: (proposalId: string) => void;
  disabled?: boolean;
}

export function DiffPreviewCard({ proposal, onApply, disabled = false }: DiffPreviewCardProps) {
  const lines = proposal.diff.split('\n').filter(Boolean);
  const bodyLines = lines.filter(
    line =>
      !line.startsWith('--- ') &&
      !line.startsWith('+++ ') &&
      !line.startsWith('@@ ')
  );

  return (
    <div className="rounded-xl border border-border bg-card p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-foreground">{proposal.path}</p>
          <p className="text-xs text-muted-foreground">Proposal {proposal.proposalId}</p>
        </div>
        <Button
          size="sm"
          onClick={() => onApply(proposal.proposalId)}
          disabled={disabled || proposal.applied}
        >
          {proposal.applied ? 'Applied' : 'Apply'}
        </Button>
      </div>
      <div className="space-y-2">
        <div className="rounded-md bg-secondary/30">
          <pre className="overflow-x-auto px-2 py-2 text-xs leading-5">
            {bodyLines.map((line, idx) => (
              <div
                key={`${proposal.proposalId}:${idx}:${line}`}
                className={
                  line.startsWith('+')
                    ? 'text-emerald-300'
                    : line.startsWith('-')
                      ? 'text-red-300'
                      : 'text-foreground'
                }
              >
                {line || ' '}
              </div>
            ))}
          </pre>
        </div>
      </div>
    </div>
  );
}
