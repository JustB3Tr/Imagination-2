'use client';

import { useState } from 'react';
import { X } from 'lucide-react';
import { ChatProvider } from '@/lib/chat-context';
import { ChatSidebar } from '@/components/chat-sidebar';
import { ChatArea } from '@/components/chat-area';
import { AuthModal } from '@/components/auth-modal';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

function ChatLayout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 z-50 transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0',
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="relative h-full">
          <ChatSidebar />
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarOpen(false)}
            className="absolute top-2 right-2 lg:hidden"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </Button>
        </div>
      </div>

      {/* Main Chat Area */}
      <ChatArea
        onToggleSidebar={() => setIsSidebarOpen(true)}
        isSidebarOpen={isSidebarOpen}
      />

      {/* Auth Modal */}
      <AuthModal />
    </div>
  );
}

export function ChatHome() {
  return (
    <ChatProvider>
      <ChatLayout />
    </ChatProvider>
  );
}
