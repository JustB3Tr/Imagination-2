'use client';

import type { MediaArtifact } from '@/lib/types';

interface MediaArtifactsProps {
  items: MediaArtifact[];
}

export function MediaArtifacts({ items }: MediaArtifactsProps) {
  if (items.length === 0) return null;

  return (
    <div className="space-y-3">
      {items.map(item => (
        <div key={`${item.sessionId}:${item.artifactId}`} className="rounded-xl border border-border bg-card p-3">
          <p className="mb-2 text-xs text-muted-foreground">
            {item.kind} - {item.artifactId}
          </p>
          {item.kind === 'screenshot' ? (
            <img src={item.src} alt="Agent capture" className="rounded-md border border-border max-h-[420px] w-full object-contain bg-black/20" />
          ) : (
            <video src={item.src} controls className="w-full rounded-md border border-border bg-black/30" />
          )}
        </div>
      ))}
    </div>
  );
}
