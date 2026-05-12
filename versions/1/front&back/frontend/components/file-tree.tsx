'use client';

import { ChevronDown, ChevronRight, File, Folder } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { WorkspaceTreeNode } from '@/lib/types';
import { cn } from '@/lib/utils';

interface FileTreeProps {
  root: WorkspaceTreeNode | null;
}

function NodeRow({ node, depth = 0 }: { node: WorkspaceTreeNode; depth?: number }) {
  const isDir = node.type === 'dir';
  const [expanded, setExpanded] = useState(depth < 2);
  const children = useMemo(() => node.children || [], [node.children]);

  return (
    <div>
      <button
        type="button"
        onClick={() => isDir && setExpanded(v => !v)}
        className={cn(
          'w-full flex items-center gap-1.5 text-left text-xs rounded px-1.5 py-1 hover:bg-sidebar-accent/40',
          'text-sidebar-foreground'
        )}
        style={{ paddingLeft: `${6 + depth * 12}px` }}
      >
        {isDir ? (
          expanded ? (
            <ChevronDown className="w-3.5 h-3.5 shrink-0" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 shrink-0" />
          )
        ) : (
          <span className="w-3.5 h-3.5 shrink-0" />
        )}
        {isDir ? <Folder className="w-3.5 h-3.5 shrink-0" /> : <File className="w-3.5 h-3.5 shrink-0" />}
        <span className="truncate">{node.name || '.'}</span>
      </button>
      {isDir && expanded && children.length > 0 && (
        <div>
          {children.map(child => (
            <NodeRow key={`${child.path}:${child.name}`} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function FileTree({ root }: FileTreeProps) {
  if (!root) {
    return <div className="text-xs text-muted-foreground px-2 py-1">No workspace loaded.</div>;
  }
  return <NodeRow node={root} />;
}
