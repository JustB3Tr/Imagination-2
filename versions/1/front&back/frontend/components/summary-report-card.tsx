'use client';

import type { SummaryReport } from '@/lib/types';

interface SummaryReportCardProps {
  report: SummaryReport;
}

export function SummaryReportCard({ report }: SummaryReportCardProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-3">
      <h3 className="text-sm font-semibold text-foreground mb-2">Summary Report</h3>
      <div className="space-y-2 text-xs text-muted-foreground">
        <p className="break-all">Session: {report.session_id}</p>
        <p className="break-all">Workspace: {report.workspace_root}</p>
      </div>

      <div className="mt-3">
        <p className="text-xs font-medium text-foreground mb-1">Files Modified</p>
        {report.files_modified.length === 0 ? (
          <p className="text-xs text-muted-foreground">No files modified yet.</p>
        ) : (
          <ul className="space-y-1 text-xs">
            {report.files_modified.map(item => (
              <li key={`${item.path}:${item.why}`} className="text-foreground">
                <span className="font-mono">{item.path}</span> - {item.why || 'Updated by agent'}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
