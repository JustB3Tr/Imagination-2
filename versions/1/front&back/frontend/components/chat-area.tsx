'use client';

import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ModelSelector } from './model-selector';
import { ChatMessages } from './chat-messages';
import { ChatInput } from './chat-input';
import { BackendStatus } from './backend-status';
import { UI_BUILD_VERSION } from '@/lib/ui-version';

interface ChatAreaProps {
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

export function ChatArea({ onToggleSidebar, isSidebarOpen }: ChatAreaProps) {
  return (
    <div className="flex h-full min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden bg-background">
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between border-b border-border/90 px-4 py-3 shadow-sm">
        <div className="flex items-center gap-2">
          {!isSidebarOpen && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleSidebar}
              className="lg:hidden"
              aria-label="Open sidebar"
            >
              <Menu className="w-5 h-5" />
            </Button>
          )}
          <ModelSelector />
        </div>
        <div className="flex items-center gap-2">
          <span
            className="select-none rounded border border-border/50 bg-muted/20 px-1 py-px font-mono text-[10px] leading-none text-muted-foreground tabular-nums"
            title="UI build — bump in lib/ui-version.ts when you ship"
          >
            v{UI_BUILD_VERSION}
          </span>
          <BackendStatus />
        </div>
      </header>

      {/* flex-col so ChatMessages flex-1 gets a real height budget (wheel scroll target) */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <ChatMessages />
      </div>

      <ChatInput />
    </div>
  );
}
