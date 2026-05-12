'use client';

import { useRef, useEffect, useLayoutEffect, useCallback, useState } from 'react';
import { Bot, User, Copy, Check, FileText, FileCode, Image as ImageIcon } from 'lucide-react';
import type { Attachment } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { useChatContext } from '@/lib/chat-context';
import { MathRenderer } from './math-renderer';
import { ResearchTracePanel } from './research-trace-panel';
import { TerminalCard } from './terminal-card';
import { DiffPreviewCard } from './diff-preview-card';
import { SummaryReportCard } from './summary-report-card';
import { MediaArtifacts } from './media-artifacts';
import { AgentTracePanel } from './agent-trace-panel';
import { cn } from '@/lib/utils';

/** Space for fixed composer + disclaimer so last messages stay scrollable above it */
const COMPOSER_SCROLL_PADDING =
  'pb-[calc(14rem+env(safe-area-inset-bottom,0px))]';

/** Pixels from bottom — if user is within this, we keep pinning to bottom on stream/resize */
const STICKY_BOTTOM_PX = 120;

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function AttachmentPreview({ attachment }: { attachment: Attachment }) {
  const getIcon = () => {
    switch (attachment.type) {
      case 'image': return <ImageIcon className="w-4 h-4" />;
      case 'code': return <FileCode className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  if (attachment.type === 'image') {
    return (
      <div className="relative rounded-lg overflow-hidden max-w-xs">
        <img
          src={attachment.url}
          alt={attachment.name}
          className="w-full h-auto max-h-64 object-cover"
        />
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-2">
          <p className="text-white text-xs truncate">{attachment.name}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 bg-muted/50 rounded-lg p-3 max-w-xs">
      <div className="flex items-center justify-center w-10 h-10 bg-secondary rounded-lg shrink-0">
        {getIcon()}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-foreground truncate">{attachment.name}</p>
        <p className="text-xs text-muted-foreground">{formatFileSize(attachment.size)}</p>
      </div>
    </div>
  );
}

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={handleCopy}
      className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0"
    >
      {copied ? (
        <Check className="w-4 h-4 text-accent" />
      ) : (
        <Copy className="w-4 h-4 text-muted-foreground" />
      )}
    </Button>
  );
}

function distanceFromBottom(el: HTMLElement) {
  return el.scrollHeight - el.scrollTop - el.clientHeight;
}

export function ChatMessages() {
  const {
    getCurrentChat,
    terminalRuns,
    diffProposals,
    mediaArtifacts,
    summaryReport,
    applyDiffProposals,
    isAgentRunning,
    agentTrace,
    workspaceSnapshot,
    writeFileStream,
    showAgentComposerUi,
  } = useChatContext();
  const chat = getCurrentChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  /** When true, stream + layout growth will follow the bottom; wheel up clears this via onScroll */
  const stickToBottomRef = useRef(true);
  const messageIdsRef = useRef<string>('');

  const messageIds = chat?.messages.map(m => m.id).join('\0') ?? '';
  const lastContentLen = chat?.messages.at(-1)?.content.length ?? 0;

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current = distanceFromBottom(el) <= STICKY_BOTTOM_PX;
  }, []);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || !chat?.messages.length) return;

    const newThreadShape = messageIds !== messageIdsRef.current;
    messageIdsRef.current = messageIds;

    if (newThreadShape) {
      stickToBottomRef.current = true;
      el.scrollTop = el.scrollHeight;
      return;
    }

    if (stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [
    messageIds,
    lastContentLen,
    chat?.messages.length,
    terminalRuns.length,
    diffProposals.length,
    mediaArtifacts.length,
    summaryReport,
    agentTrace.length,
    writeFileStream?.revealed,
    writeFileStream?.fullText,
  ]);

  useEffect(() => {
    const scrollEl = scrollRef.current;
    const inner = innerRef.current;
    if (!scrollEl || !inner) return;
    const ro = new ResizeObserver(() => {
      if (!stickToBottomRef.current) return;
      scrollEl.scrollTop = scrollEl.scrollHeight;
    });
    ro.observe(inner);
    return () => ro.disconnect();
  }, [
    chat?.messages?.length,
    messageIds,
    terminalRuns.length,
    diffProposals.length,
    mediaArtifacts.length,
    summaryReport,
    agentTrace.length,
    writeFileStream?.revealed,
    writeFileStream?.fullText,
  ]);

  if (!chat || chat.messages.length === 0) {
    return (
      <div
        className={cn(
          'flex min-h-0 flex-1 flex-col overflow-y-scroll overscroll-contain px-4 touch-pan-y',
          COMPOSER_SCROLL_PADDING
        )}
        onScroll={onScroll}
      >
        <div className="flex min-h-full flex-1 items-center justify-center py-8">
          <div className="max-w-md px-4 text-center">
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/10">
              <Bot className="h-8 w-8 text-accent" />
            </div>
            <h2 className="mb-2 text-2xl font-semibold text-foreground">
              How can I help you today?
            </h2>
            <p className="text-muted-foreground">
              Start a conversation with Imagination AI. Ask questions, get creative, or explore ideas
              together.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      className={cn(
        'relative z-0 min-h-0 flex-1 overflow-y-scroll overflow-x-hidden overscroll-contain px-4 touch-pan-y [scrollbar-gutter:stable]',
        COMPOSER_SCROLL_PADDING
      )}
    >
      <div ref={innerRef} className="mx-auto max-w-3xl space-y-6 py-8">
        {chat.messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              'group flex gap-4',
              message.role === 'user' ? 'flex-row-reverse' : ''
            )}
          >
            {/* Avatar */}
            <div
              className={cn(
                'flex items-center justify-center w-8 h-8 rounded-full shrink-0',
                message.role === 'user'
                  ? 'bg-accent text-accent-foreground'
                  : 'bg-secondary text-secondary-foreground'
              )}
            >
              {message.role === 'user' ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
            </div>

            {/* Message Bubble */}
            <div
              className={cn(
                'flex-1 max-w-[80%]',
                message.role === 'user' ? 'flex justify-end' : ''
              )}
            >
              <div
                className={cn(
                  'relative px-4 py-3 rounded-2xl',
                  message.role === 'user'
                    ? 'bg-accent text-accent-foreground rounded-tr-md'
                    : 'bg-card border border-border/80 rounded-tl-md shadow-sm ring-1 ring-white/10'
                )}
              >
                {/* Attachments */}
                {message.attachments && message.attachments.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-3">
                    {message.attachments.map((attachment) => (
                      <AttachmentPreview key={attachment.id} attachment={attachment} />
                    ))}
                  </div>
                )}
                
                {message.role === 'assistant' ? (
                  message.content.trim() ? (
                    <MathRenderer content={message.content} />
                  ) : (
                    <span
                      className="inline-block h-4 w-px animate-pulse bg-muted-foreground/60 align-middle"
                      aria-hidden
                    />
                  )
                ) : (
                  message.content && <p className="whitespace-pre-wrap">{message.content}</p>
                )}
                {message.role === 'assistant' &&
                  ((message.researchTrace?.length ?? 0) > 0 ||
                    message.answerPhase === 'preliminary' ||
                    message.answerPhase === 'final') ? (
                  <ResearchTracePanel
                    events={message.researchTrace ?? []}
                    phase={message.answerPhase ?? 'idle'}
                  />
                ) : null}
              </div>

              {/* Actions */}
              {message.role === 'assistant' && (
                <div className="flex items-center gap-1 mt-1 px-2">
                  <CopyButton content={message.content} />
                </div>
              )}
            </div>
          </div>
        ))}

        {showAgentComposerUi && workspaceSnapshot?.root ? (
          <div className="mb-2 rounded-lg border border-border/60 bg-muted/20 px-2.5 py-2 text-[11px] leading-snug text-muted-foreground">
            <span className="font-medium text-foreground/80">Workspace</span>
            <p className="mt-0.5 break-all font-mono text-[10px] text-foreground/70">
              {workspaceSnapshot.root}
            </p>
            {workspaceSnapshot.session_id ? (
              <p className="mt-1 text-[10px]">
                Session <span className="font-mono text-foreground/80">{workspaceSnapshot.session_id}</span>
              </p>
            ) : null}
          </div>
        ) : null}

        {showAgentComposerUi &&
        (agentTrace.length > 0 || isAgentRunning || writeFileStream) ? (
          <AgentTracePanel
            entries={agentTrace}
            scriptPreview={writeFileStream}
            isAgentRunning={isAgentRunning}
          />
        ) : null}

        {showAgentComposerUi && terminalRuns.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Terminal</p>
            {terminalRuns.map(run => (
              <TerminalCard key={run.id} run={run} />
            ))}
          </div>
        )}

        {showAgentComposerUi && diffProposals.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Diff Preview</p>
            {diffProposals.map(proposal => (
              <DiffPreviewCard
                key={proposal.proposalId}
                proposal={proposal}
                onApply={(proposalId) => {
                  void applyDiffProposals([proposalId]);
                }}
                disabled={isAgentRunning}
              />
            ))}
          </div>
        )}

        {showAgentComposerUi && mediaArtifacts.length > 0 && (
          <div>
            <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">Screen Capture</p>
            <MediaArtifacts items={mediaArtifacts} />
          </div>
        )}

        {showAgentComposerUi && summaryReport && (
          <SummaryReportCard report={summaryReport} />
        )}
      </div>
    </div>
  );
}
