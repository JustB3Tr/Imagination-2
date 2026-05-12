'use client';

import { Plus, MessageSquare, Trash2, Settings, LogOut, User, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChatContext } from '@/lib/chat-context';
import { cn } from '@/lib/utils';
import { FileTree } from '@/components/file-tree';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

export function ChatSidebar() {
  const {
    chats,
    currentChatId,
    createNewChat,
    selectChat,
    deleteChat,
    user,
    isAuthenticated,
    setShowAuthModal,
    signOut,
    workspaceSnapshot,
    refreshWorkspaceTree,
    showAgentComposerUi,
  } = useChatContext();

  return (
    <aside className="flex flex-col h-full w-64 bg-sidebar border-r border-sidebar-border">
      {/* Logo & New Chat */}
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center gap-2 mb-4">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent">
            <Sparkles className="w-5 h-5 text-accent-foreground" />
          </div>
          <span className="font-semibold text-sidebar-foreground">Imagination</span>
        </div>
        <Button
          onClick={createNewChat}
          className="w-full justify-start gap-2 bg-sidebar-accent hover:bg-sidebar-accent/80 text-sidebar-accent-foreground"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </Button>
      </div>

      {/* Chat History */}
      <ScrollArea className="flex-1 px-2 py-2">
        <div className="space-y-1">
          {chats?.map((chat) => (
            <div
              key={chat.id}
              className={cn(
                'group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors',
                currentChatId === chat.id
                  ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent/50'
              )}
              onClick={() => selectChat(chat.id)}
            >
              <MessageSquare className="w-4 h-4 shrink-0" />
              <span className="flex-1 truncate text-sm">{chat.title}</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(chat.id);
                }}
                className="shrink-0 p-1 opacity-100 hover:text-destructive transition-colors"
                aria-label="Delete chat"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>

        {showAgentComposerUi ? (
          <div className="mt-4 rounded-lg border border-sidebar-border/70 bg-sidebar-accent/20 p-2">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Workspace
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[11px]"
                onClick={() => {
                  void refreshWorkspaceTree();
                }}
              >
                Refresh
              </Button>
            </div>
            <FileTree root={workspaceSnapshot?.tree ?? null} />
          </div>
        ) : null}
      </ScrollArea>

      {/* User Profile Section */}
      <div className="p-4 border-t border-sidebar-border">
        {isAuthenticated && user ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="w-full justify-start gap-3 px-2 hover:bg-sidebar-accent"
              >
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-accent text-accent-foreground text-sm font-medium">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div className="flex flex-col items-start">
                  <span className="text-sm font-medium text-sidebar-foreground">{user.name}</span>
                  <span className="text-xs text-muted-foreground">{user.email}</span>
                </div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuItem>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={signOut} className="text-destructive">
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button
            onClick={() => setShowAuthModal(true)}
            variant="outline"
            className="w-full justify-start gap-2"
          >
            <User className="w-4 h-4" />
            Sign In
          </Button>
        )}
      </div>
    </aside>
  );
}
