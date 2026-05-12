'use client';

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  type ReactNode,
} from 'react';
import { flushSync } from 'react-dom';
import { useSession, signIn as nextAuthSignIn, signOut as nextAuthSignOut } from 'next-auth/react';
import type {
  AgentChatMemory,
  AgentTraceEntry,
  Attachment,
  Chat,
  DiffProposal,
  MediaArtifact,
  Message,
  ModelType,
  ResearchTraceEvent,
  SummaryReport,
  TerminalRun,
  User,
  VisualReportPayload,
  WorkspaceSnapshot,
  WriteFileStreamState,
} from './types';
import { deriveChatTitle } from './chat-title';
import {
  loadAppState,
  saveAppState,
  chatsToJsonlLines,
  type PersistedChat,
  type PersistedMessage,
} from './local-chat-store';

interface ChatContextType {
  chats: Chat[];
  currentChatId: string | null;
  currentModel: ModelType;
  user: User | null;
  isAuthenticated: boolean;
  showAuthModal: boolean;
  setShowAuthModal: (show: boolean) => void;
  storageHydrated: boolean;
  deepResearchActive: boolean;
  setDeepResearchActive: (v: boolean) => void;
  createNewChat: () => string;
  selectChat: (chatId: string) => void;
  deleteChat: (chatId: string) => void;
  addMessage: (
    content: string,
    role: 'user' | 'assistant',
    attachments?: Attachment[],
    forChatId?: string | null
  ) => string | undefined;
  updateMessageContent: (
    messageId: string,
    content: string,
    forChatId?: string | null
  ) => void;
  updateMessageFields: (
    messageId: string,
    patch: Partial<Pick<Message, 'content' | 'researchTrace' | 'answerPhase'>>,
    forChatId?: string | null
  ) => void;
  appendResearchTrace: (
    messageId: string,
    event: ResearchTraceEvent,
    forChatId?: string | null
  ) => void;
  setCurrentModel: (model: ModelType) => void;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  getCurrentChat: () => Chat | undefined;
  exportChatsJsonl: (scope?: 'all' | 'current') => void;
  syncChatsFromBackend: () => Promise<{ merged: number; error?: string }>;
  pushChatsToBackend: () => Promise<{ ok: boolean; error?: string }>;
  /** Deep research (/research …) → proxies to FastAPI `/api/chat/deep` (NDJSON). */
  sendResearchToBackend: (userTextRaw: string, attachments?: Attachment[]) => Promise<void>;
  sendAgentChatToBackend: (userText: string, attachments?: Attachment[]) => Promise<void>;
  applyDiffProposals: (proposalIds: string[]) => Promise<void>;
  refreshWorkspaceTree: () => Promise<void>;
  agentSessionId: string | null;
  workspaceSnapshot: WorkspaceSnapshot | null;
  terminalRuns: TerminalRun[];
  diffProposals: DiffProposal[];
  mediaArtifacts: MediaArtifact[];
  summaryReport: SummaryReport | null;
  isAgentRunning: boolean;
  agentTrace: AgentTraceEntry[];
  /** Live typewriter preview while a write_file tool call is in progress */
  writeFileStream: WriteFileStreamState | null;
  /** Agent workspace strip, trace, terminal, diffs — only while `/agent` is active */
  showAgentComposerUi: boolean;
  hideAgentComposerUi: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

const generateId = () => Math.random().toString(36).substring(2, 15);

  const OFFLINE_ASSISTANT_MESSAGE =
  'Imagination AI is currently offline. Please restart the backend.';

function stripResearchCommand(userRaw: string): { topic: string; displayLine: string } {
  const displayLine = userRaw.trim();
  const m = displayLine.match(/^\/research\b\s*(.*)$/is);
  if (!m) {
    return { topic: '', displayLine };
  }
  return { topic: (m[1] ?? '').trim(), displayLine };
}

type AgentEvent = Record<string, unknown>;

const MAX_TRACE_DETAIL_PERSIST = 8000;

function buildAgentMemorySnapshot(
  sessionId: string | null,
  workspaceSnapshot: WorkspaceSnapshot | null,
  agentTrace: AgentTraceEntry[],
  diffProposals: DiffProposal[],
  terminalRuns: TerminalRun[],
  mediaArtifacts: MediaArtifact[],
  summaryReport: SummaryReport | null
): AgentChatMemory | null {
  const hasSession = !!(sessionId && sessionId.trim());
  const hasWs = !!(workspaceSnapshot && workspaceSnapshot.root);
  const hasAny =
    hasSession ||
    hasWs ||
    agentTrace.length > 0 ||
    diffProposals.length > 0 ||
    terminalRuns.length > 0 ||
    mediaArtifacts.length > 0 ||
    !!summaryReport;
  if (!hasAny) return null;

  const trace = agentTrace.map(e => ({
    ...e,
    detail:
      e.detail && e.detail.length > MAX_TRACE_DETAIL_PERSIST
        ? e.detail.slice(0, MAX_TRACE_DETAIL_PERSIST) + '\u2026'
        : e.detail,
  }));
  const media = mediaArtifacts.map(m => ({
    ...m,
    src: m.src.startsWith('data:') ? '' : m.src,
  }));

  return {
    sessionId: sessionId ?? undefined,
    workspaceRoot: workspaceSnapshot?.root,
    workspaceSnapshot: workspaceSnapshot ?? undefined,
    agentTrace: trace,
    diffProposals: diffProposals.map(d => ({ ...d })),
    terminalRuns: terminalRuns.map(t => ({ ...t })),
    mediaArtifacts: media,
    summaryReport: summaryReport ?? undefined,
  };
}

function mergeLiveAgentIntoChats(
  chats: Chat[],
  currentChatId: string | null,
  sessionId: string | null,
  workspaceSnapshot: WorkspaceSnapshot | null,
  agentTrace: AgentTraceEntry[],
  diffProposals: DiffProposal[],
  terminalRuns: TerminalRun[],
  mediaArtifacts: MediaArtifact[],
  summaryReport: SummaryReport | null
): Chat[] {
  if (!currentChatId) return chats;
  const snap = buildAgentMemorySnapshot(
    sessionId,
    workspaceSnapshot,
    agentTrace,
    diffProposals,
    terminalRuns,
    mediaArtifacts,
    summaryReport
  );
  if (!snap) return chats;
  return chats.map(c => (c.id === currentChatId ? { ...c, agentMemory: snap } : c));
}

function persistedToChat(c: PersistedChat): Chat {
  return {
    id: c.id,
    title: c.title,
    model: c.model,
    createdAt: new Date(c.createdAt),
    messages: c.messages.map(m => ({
      id: m.id,
      role: m.role,
      content: m.content,
      attachments: m.attachments,
      createdAt: new Date(m.createdAt),
      researchTrace: m.researchTrace,
      answerPhase: m.answerPhase,
    })),
    agentMemory: c.agentMemory ?? undefined,
  };
}

function chatToPersisted(c: Chat): PersistedChat {
  const messages: PersistedMessage[] = c.messages.map(m => ({
    id: m.id,
    role: m.role,
    content: m.content,
    attachments: m.attachments,
    createdAt: m.createdAt.toISOString(),
    researchTrace: m.researchTrace,
    answerPhase: m.answerPhase,
  }));
  return {
    id: c.id,
    title: c.title,
    model: c.model,
    createdAt: c.createdAt.toISOString(),
    messages,
    agentMemory: c.agentMemory ?? undefined,
  };
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const { data: session, status: sessionStatus } = useSession();
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<ModelType>('imagination-1.3');
  const [user, setUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [storageHydrated, setStorageHydrated] = useState(false);
  const [deepResearchActive, setDeepResearchActive] = useState(false);
  const agentSessionIdRef = useRef<string | null>(null);
  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState<WorkspaceSnapshot | null>(null);
  const [terminalRuns, setTerminalRuns] = useState<TerminalRun[]>([]);
  const [diffProposals, setDiffProposals] = useState<DiffProposal[]>([]);
  const [mediaArtifacts, setMediaArtifacts] = useState<MediaArtifact[]>([]);
  const [summaryReport, setSummaryReport] = useState<SummaryReport | null>(null);
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [showAgentComposerUi, setShowAgentComposerUi] = useState(false);
  const [agentTrace, setAgentTrace] = useState<AgentTraceEntry[]>([]);
  const [writeFileStream, setWriteFileStream] = useState<WriteFileStreamState | null>(null);
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const currentChatIdRef = useRef<string | null>(null);
  const latestRef = useRef({
    chats,
    currentChatId,
    currentModel,
    agentSessionId,
    workspaceSnapshot,
    agentTrace,
    diffProposals,
    terminalRuns,
    mediaArtifacts,
    summaryReport,
    writeFileStream: null as WriteFileStreamState | null,
  });
  currentChatIdRef.current = currentChatId;
  latestRef.current = {
    chats,
    currentChatId,
    currentModel,
    agentSessionId,
    workspaceSnapshot,
    agentTrace,
    diffProposals,
    terminalRuns,
    mediaArtifacts,
    summaryReport,
    writeFileStream,
  };

  useEffect(() => {
    if (session?.user) {
      const u = session.user;
      setUser({
        id: (u as { id?: string }).id ?? u.email ?? generateId(),
        name: u.name ?? (u.email ? u.email.split('@')[0] : 'User'),
        email: u.email ?? '',
        avatar: u.image ?? undefined,
      });
    } else if (sessionStatus === 'unauthenticated') {
      setUser(null);
    }
  }, [session, sessionStatus]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const raw = await loadAppState();
      if (cancelled) return;
      if (raw && raw.chats?.length) {
        setChats(raw.chats.map(persistedToChat));
        setCurrentChatId(raw.currentChatId ?? raw.chats[0]?.id ?? null);
        setCurrentModel(raw.currentModel ?? 'imagination-1.3');
      }
      setStorageHydrated(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const schedulePersist = useCallback(() => {
    if (!storageHydrated) return;
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    persistTimerRef.current = setTimeout(() => {
      persistTimerRef.current = null;
      void (async () => {
        const r = latestRef.current;
        const merged = mergeLiveAgentIntoChats(
          r.chats,
          r.currentChatId,
          r.agentSessionId,
          r.workspaceSnapshot,
          r.agentTrace,
          r.diffProposals,
          r.terminalRuns,
          r.mediaArtifacts,
          r.summaryReport
        );
        const state = {
          chats: merged.map(chatToPersisted),
          currentChatId: r.currentChatId,
          currentModel: r.currentModel,
          updatedAt: new Date().toISOString(),
        };
        await saveAppState(state);
      })();
    }, 400);
  }, [storageHydrated]);

  useEffect(() => {
    schedulePersist();
    return () => {
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    };
  }, [
    chats,
    currentChatId,
    currentModel,
    agentSessionId,
    workspaceSnapshot,
    agentTrace,
    diffProposals,
    terminalRuns,
    mediaArtifacts,
    summaryReport,
    schedulePersist,
  ]);

  useEffect(() => {
    if (!storageHydrated) return;
    const flush = () => {
      const r = latestRef.current;
      const merged = mergeLiveAgentIntoChats(
        r.chats,
        r.currentChatId,
        r.agentSessionId,
        r.workspaceSnapshot,
        r.agentTrace,
        r.diffProposals,
        r.terminalRuns,
        r.mediaArtifacts,
        r.summaryReport
      );
      void saveAppState({
        chats: merged.map(chatToPersisted),
        currentChatId: r.currentChatId,
        currentModel: r.currentModel,
        updatedAt: new Date().toISOString(),
      });
    };
    window.addEventListener('beforeunload', flush);
    return () => window.removeEventListener('beforeunload', flush);
  }, [storageHydrated]);

  const applyAgentMemoryToUi = useCallback((mem: AgentChatMemory | null | undefined) => {
    const m = mem ?? null;
    const sid = m?.sessionId ?? null;
    setAgentSessionId(sid);
    agentSessionIdRef.current = sid;
    if (m?.workspaceSnapshot) {
      setWorkspaceSnapshot(m.workspaceSnapshot);
    } else if (m?.workspaceRoot && m?.sessionId) {
      setWorkspaceSnapshot({
        root: m.workspaceRoot,
        session_id: m.sessionId,
        tree: null,
        truncated: false,
      });
    } else {
      setWorkspaceSnapshot(null);
    }
    setAgentTrace(m?.agentTrace ?? []);
    setDiffProposals(m?.diffProposals ?? []);
    setTerminalRuns(m?.terminalRuns ?? []);
    setMediaArtifacts(m?.mediaArtifacts ?? []);
    setSummaryReport(m?.summaryReport ?? null);
    setWriteFileStream(null);
  }, []);

  useEffect(() => {
    if (!writeFileStream || writeFileStream.complete) return;
    const fullLen = writeFileStream.fullText.length;
    if (writeFileStream.revealed >= fullLen) {
      setWriteFileStream(s => (s ? { ...s, complete: true } : null));
      return;
    }
    const id = window.setTimeout(() => {
      setWriteFileStream(s => {
        if (!s || s.complete) return s;
        const step = Math.min(6, s.fullText.length - s.revealed);
        return { ...s, revealed: s.revealed + step };
      });
    }, 18);
    return () => clearTimeout(id);
  }, [writeFileStream]);

  useLayoutEffect(() => {
    if (!storageHydrated) return;
    if (!currentChatId) {
      applyAgentMemoryToUi(null);
      return;
    }
    const mem = latestRef.current.chats.find(c => c.id === currentChatId)?.agentMemory;
    applyAgentMemoryToUi(mem ?? null);
  }, [storageHydrated, currentChatId, applyAgentMemoryToUi]);

  useEffect(() => {
    if (!storageHydrated || !currentChatId) return;
    const r = latestRef.current;
    const snap = buildAgentMemorySnapshot(
      r.agentSessionId,
      r.workspaceSnapshot,
      r.agentTrace,
      r.diffProposals,
      r.terminalRuns,
      r.mediaArtifacts,
      r.summaryReport
    );
    if (!snap) return;
    const prevMem = r.chats.find(c => c.id === currentChatId)?.agentMemory ?? null;
    if (JSON.stringify(snap) === JSON.stringify(prevMem)) return;
    setChats(prev =>
      prev.map(c => (c.id === currentChatId ? { ...c, agentMemory: snap } : c))
    );
  }, [
    storageHydrated,
    currentChatId,
    agentSessionId,
    workspaceSnapshot,
    agentTrace,
    diffProposals,
    terminalRuns,
    mediaArtifacts,
    summaryReport,
  ]);

  const createNewChat = useCallback((): string => {
    const newChat: Chat = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      model: currentModel,
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    applyAgentMemoryToUi(null);
    setShowAgentComposerUi(false);
    return newChat.id;
  }, [applyAgentMemoryToUi, currentModel]);

  const selectChat = useCallback(
    (chatId: string) => {
      const leavingId = currentChatIdRef.current;
      if (leavingId) {
        const r = latestRef.current;
        const snap = buildAgentMemorySnapshot(
          r.agentSessionId,
          r.workspaceSnapshot,
          r.agentTrace,
          r.diffProposals,
          r.terminalRuns,
          r.mediaArtifacts,
          r.summaryReport
        );
        if (snap) {
          setChats(prev =>
            prev.map(c => (c.id === leavingId ? { ...c, agentMemory: snap } : c))
          );
        }
      }
      const target = latestRef.current.chats.find(c => c.id === chatId);
      if (target) setCurrentModel(target.model);
      setCurrentChatId(chatId);
      setShowAgentComposerUi(false);
    },
    []
  );

  const hideAgentComposerUi = useCallback(() => {
    setShowAgentComposerUi(false);
  }, []);

  const deleteChat = useCallback(
    (chatId: string) => {
      const r = latestRef.current;
      const deletingCurrent = chatId === currentChatId;
      const snap = deletingCurrent
        ? buildAgentMemorySnapshot(
            r.agentSessionId,
            r.workspaceSnapshot,
            r.agentTrace,
            r.diffProposals,
            r.terminalRuns,
            r.mediaArtifacts,
            r.summaryReport
          )
        : null;
      setChats(prev => {
        const patched =
          snap && deletingCurrent
            ? prev.map(c => (c.id === chatId ? { ...c, agentMemory: snap } : c))
            : prev;
        const filtered = patched.filter(c => c.id !== chatId);
        if (deletingCurrent) {
          setCurrentChatId(filtered[0]?.id ?? null);
        }
        return filtered;
      });
    },
    [currentChatId]
  );

  const addMessage = useCallback(
    (
      content: string,
      role: 'user' | 'assistant',
      attachments?: Attachment[],
      forChatId?: string | null
    ): string | undefined => {
      const chatId = forChatId ?? currentChatId;
      if (!chatId) return undefined;

      const newMessage: Message = {
        id: generateId(),
        role,
        content,
        attachments,
        createdAt: new Date(),
        answerPhase: role === 'assistant' ? 'idle' : undefined,
      };

      setChats(prev =>
        prev.map(chat => {
          if (chat.id !== chatId) return chat;
          const updatedMessages = [...chat.messages, newMessage];
          const title =
            chat.messages.length === 0 && role === 'user'
              ? deriveChatTitle(content)
              : chat.title;
          return { ...chat, messages: updatedMessages, title };
        })
      );
      return newMessage.id;
    },
    [currentChatId]
  );

  const updateMessageContent = useCallback(
    (messageId: string, content: string, forChatId?: string | null) => {
      const chatId = forChatId ?? currentChatId;
      if (!chatId) return;

      setChats(prev =>
        prev.map(chat => {
          if (chat.id !== chatId) return chat;
          return {
            ...chat,
            messages: chat.messages.map(m =>
              m.id === messageId ? { ...m, content } : m
            ),
          };
        })
      );
    },
    [currentChatId]
  );

  const updateMessageFields = useCallback(
    (
      messageId: string,
      patch: Partial<Pick<Message, 'content' | 'researchTrace' | 'answerPhase'>>,
      forChatId?: string | null
    ) => {
      const chatId = forChatId ?? currentChatId;
      if (!chatId) return;

      setChats(prev =>
        prev.map(chat => {
          if (chat.id !== chatId) return chat;
          return {
            ...chat,
            messages: chat.messages.map(m =>
              m.id === messageId ? { ...m, ...patch } : m
            ),
          };
        })
      );
    },
    [currentChatId]
  );

  const appendResearchTrace = useCallback(
    (messageId: string, event: ResearchTraceEvent, forChatId?: string | null) => {
      const chatId = forChatId ?? currentChatId;
      if (!chatId) return;

      setChats(prev =>
        prev.map(chat => {
          if (chat.id !== chatId) return chat;
          return {
            ...chat,
            messages: chat.messages.map(m => {
              if (m.id !== messageId) return m;
              const next = [...(m.researchTrace ?? []), event];
              return { ...m, researchTrace: next };
            }),
          };
        })
      );
    },
    [currentChatId]
  );

  const signIn = useCallback(async (email: string, password: string) => {
    const res = await nextAuthSignIn('credentials', {
      email,
      password,
      redirect: false,
    });
    if (res?.error) {
      throw new Error(res.error);
    }
    setShowAuthModal(false);
  }, []);

  const signOut = useCallback(async () => {
    await nextAuthSignOut({ redirect: false });
    setUser(null);
  }, []);

  const getCurrentChat = useCallback(() => {
    return chats.find(c => c.id === currentChatId);
  }, [chats, currentChatId]);

  const exportChatsJsonl = useCallback(
    (scope: 'all' | 'current' = 'all') => {
      const list =
        scope === 'current' && currentChatId
          ? chats.filter(c => c.id === currentChatId)
          : chats;
      const blob = new Blob([chatsToJsonlLines(list.map(chatToPersisted))], {
        type: 'application/x-ndjson',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download =
        scope === 'current' && currentChatId
          ? `imagination-chat-${currentChatId}.jsonl`
          : 'imagination-chats-export.jsonl';
      a.click();
      URL.revokeObjectURL(url);
    },
    [chats, currentChatId]
  );

  const syncChatsFromBackend = useCallback(async () => {
    try {
      const res = await fetch('/api/sync/chats', { method: 'GET' });
      if (!res.ok) {
        const t = await res.text();
        return { merged: 0, error: t || res.statusText };
      }
      const data = (await res.json()) as { chats?: PersistedChat[] };
      const remote = data.chats ?? [];
      if (!remote.length) return { merged: 0 };

      setChats(prev => {
        const byId = new Map(prev.map(c => [c.id, c]));
        let merged = 0;
        for (const pc of remote) {
          const incoming = persistedToChat(pc);
          const existing = byId.get(incoming.id);
          if (!existing) {
            byId.set(incoming.id, incoming);
            merged += 1;
            continue;
          }
          const exT = existing.createdAt.getTime();
          const inT = incoming.createdAt.getTime();
          if (inT >= exT) {
            const mergedChat: Chat = {
              ...incoming,
              agentMemory: incoming.agentMemory ?? existing.agentMemory ?? null,
            };
            byId.set(incoming.id, mergedChat);
            merged += 1;
          }
        }
        return Array.from(byId.values()).sort(
          (a, b) => b.createdAt.getTime() - a.createdAt.getTime()
        );
      });
      return { merged: remote.length };
    } catch (e) {
      return { merged: 0, error: String(e) };
    }
  }, []);

  const pushChatsToBackend = useCallback(async () => {
    try {
      const body = { chats: chats.map(chatToPersisted) };
      const res = await fetch('/api/sync/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const t = await res.text();
        return { ok: false, error: t || res.statusText };
      }
      return { ok: true };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  }, [chats]);

  const refreshWorkspaceTree = useCallback(async () => {
    if (!agentSessionId) return;
    try {
      const q = new URLSearchParams({ session_id: agentSessionId });
      const res = await fetch(`/api/agent/workspace?${q.toString()}`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      if (!res.ok) return;
      const data = (await res.json()) as WorkspaceSnapshot;
      setWorkspaceSnapshot(data);
    } catch {
      /* ignore */
    }
  }, [agentSessionId]);

  const handleAgentEvent = useCallback((event: AgentEvent) => {
    const type = String(event.type || '');
    if (type === 'session') {
      const sid = String(event.session_id || '');
      if (sid) {
        setAgentSessionId(sid);
        agentSessionIdRef.current = sid;
      }
      return;
    }

    if (type === 'workspace_tree') {
      setWorkspaceSnapshot({
        root: String(event.root || ''),
        session_id:
          String(event.session_id || '') || agentSessionIdRef.current || '',
        tree: (event.tree as WorkspaceSnapshot['tree']) ?? null,
        truncated: Boolean(event.truncated),
      });
      return;
    }

    if (type === 'thought') {
      const t = String(event.text || '').trim();
      if (t) {
        setAgentTrace(prev => [
          ...prev,
          { id: generateId(), ts: Date.now(), kind: 'thought', text: t },
        ]);
      }
      return;
    }

    if (type === 'error') {
      const msg = String(
        (event as { message?: string }).message ||
          (event as { error?: string }).error ||
          ''
      ).trim();
      if (msg) {
        setAgentTrace(prev => [
          ...prev,
          { id: generateId(), ts: Date.now(), kind: 'error', text: msg },
        ]);
      }
      return;
    }

    if (type === 'final') {
      const t = String((event as { text?: string }).text || '').trim();
      setAgentTrace(prev => [
        ...prev,
        {
          id: generateId(),
          ts: Date.now(),
          kind: 'final',
          text:
            t ||
            'Agent loop completed. Review terminal output, diffs, and summary cards below.',
        },
      ]);
      return;
    }

    if (type === 'ui_artisan_revision') {
      const crid = String((event as { code_revision_id?: string }).code_revision_id || '');
      const hint = String((event as { mandatory_capture_hint?: string }).mandatory_capture_hint || '');
      setAgentTrace(prev => [
        ...prev,
        {
          id: generateId(),
          ts: Date.now(),
          kind: 'ui_artisan_revision',
          codeRevisionId: crid || undefined,
          text: hint || `Code revision ${crid}`,
          detail: JSON.stringify(event).slice(0, 1500),
        },
      ]);
      return;
    }

    if (type === 'ui_artisan_final') {
      const fa = (event as { final_answer?: Record<string, unknown> }).final_answer;
      const vr = fa && typeof fa === 'object' ? (fa as { visual_report?: VisualReportPayload }).visual_report : undefined;
      const txt = fa && typeof fa === 'object' ? String((fa as { text?: string }).text || '') : '';
      setAgentTrace(prev => [
        ...prev,
        {
          id: generateId(),
          ts: Date.now(),
          kind: 'ui_artisan_final',
          text: txt || 'UI Artisan final answer',
          visualReport: vr,
          detail: JSON.stringify(fa || {}).slice(0, 2000),
        },
      ]);
      return;
    }

    if (type === 'tool_call') {
      const callId = String(event.id || generateId());
      const name = String(event.name || '').trim();
      const args = (event.args ?? {}) as Record<string, unknown>;
      const detail =
        name === 'run_shell'
          ? String(args.command || args.cmd || '')
          : name === 'write_file' || name === 'read_file'
            ? String(args.path || '')
            : JSON.stringify(args).slice(0, 800);
      setAgentTrace(prev => [
        ...prev,
        {
          id: generateId(),
          ts: Date.now(),
          kind: 'tool_call',
          callId,
          name,
          detail: detail || undefined,
        },
      ]);
      if (name === 'write_file') {
        const path = String(args.path || 'untitled');
        const fullText =
          typeof args.content === 'string' ? args.content : String(args.content ?? '');
        setWriteFileStream({
          callId,
          path,
          fullText,
          revealed: 0,
          complete: fullText.length === 0,
        });
      }
      if (name === 'run_shell') {
        const command = String(args.command || args.cmd || '');
        setTerminalRuns(prev => [...prev, { id: callId, command, status: 'running' }]);
      }
      return;
    }

    if (type === 'tool_result') {
      const callId = String(event.id || '');
      const name = String(event.name || '').trim();
      const ok = Boolean((event as { ok?: boolean }).ok);
      const data = ((event.data ?? {}) as Record<string, unknown>) || {};
      const preview = JSON.stringify(data).slice(0, 1200);
      setAgentTrace(prev => [
        ...prev,
        {
          id: generateId(),
          ts: Date.now(),
          kind: 'tool_result',
          callId,
          name,
          ok,
          detail: preview,
        },
      ]);
      if (name === 'write_file') {
        setWriteFileStream(s => {
          if (!s || s.callId !== callId) return s;
          return { ...s, revealed: s.fullText.length, complete: true };
        });
      }
      if (name === 'run_shell') {
        const status = String(data.status || 'fail') === 'success' ? 'success' : 'fail';
        setTerminalRuns(prev =>
          prev.map(run =>
            run.id === callId
              ? {
                  ...run,
                  status,
                  cwd: typeof data.cwd === 'string' ? data.cwd : run.cwd,
                  stdout: typeof data.stdout === 'string' ? data.stdout : '',
                  stderr: typeof data.stderr === 'string' ? data.stderr : '',
                  exitCode: typeof data.exit_code === 'number' ? data.exit_code : undefined,
                }
              : run
          )
        );
      }
      return;
    }

    if (type === 'diff_preview') {
      const proposalId = String(event.proposal_id || '');
      const path = String(event.path || '');
      const diff = String(event.diff || '');
      const applied = Boolean(event.applied);
      if (!proposalId) return;
      setDiffProposals(prev => {
        const has = prev.some(p => p.proposalId === proposalId);
        if (has) return prev;
        return [...prev, { proposalId, path, diff, applied }];
      });
      return;
    }

    if (type === 'media') {
      const kind = String(event.kind || 'screenshot');
      const sid = String(event.session_id || '') || agentSessionIdRef.current || '';
      const artifactId = String(event.artifact_id || '');
      if (!artifactId || !sid) return;

      if (kind === 'screenshot') {
        const base64 = String(event.base64 || '');
        const mimeType = String(event.mime_type || 'image/png');
        const src = base64
          ? `data:${mimeType};base64,${base64}`
          : `/api/agent/capture/${encodeURIComponent(sid)}/${encodeURIComponent(artifactId)}`;
        setMediaArtifacts(prev => [
          ...prev,
          { artifactId, sessionId: sid, kind: 'screenshot', mimeType, src },
        ]);

        const video = (event.video ?? null) as Record<string, unknown> | null;
        if (video && typeof video.artifact_id === 'string') {
          const videoArtifactId = video.artifact_id;
          const videoMimeType =
            typeof video.mime_type === 'string' ? video.mime_type : 'video/webm';
          const videoSrc = `/api/agent/capture/${encodeURIComponent(sid)}/${encodeURIComponent(videoArtifactId)}`;
          setMediaArtifacts(prev => [
            ...prev,
            {
              artifactId: videoArtifactId,
              sessionId: sid,
              kind: 'video',
              mimeType: videoMimeType,
              src: videoSrc,
            },
          ]);
        }
      }
      return;
    }

    if (type === 'summary') {
      const report = (event.report ?? null) as SummaryReport | null;
      if (report) setSummaryReport(report);
    }
  }, []);

  const sendChatToBackend = useCallback(
    async (userText: string, currentAttachments?: Attachment[]) => {
      let chatId = currentChatId;
      if (!chatId) {
        chatId = createNewChat();
      }
      const priorChat = chats.find(c => c.id === chatId);

      const priorMessages = (priorChat?.messages ?? []).map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }));

      const atts = currentAttachments ?? [];
      const userMessage = (userText || '').trim() || (atts.length > 0 ? 'Attached files' : '');
      if (!userMessage && atts.length === 0) return;

      let augmented = userMessage;
      if (atts.length > 0) {
        const names = atts.map(a => `${a.name} (${a.type})`).join(', ');
        augmented = `[Attached: ${names}]\n\n${userMessage || 'See attachments.'}`;
      }

      const messages = [...priorMessages, { role: 'user' as const, content: augmented }];

      const firstImageDataUrl = atts.find(
        a => a.type === 'image' && typeof a.url === 'string'
      )?.url;

      addMessage(userText || (atts.length > 0 ? 'Attached files' : ''), 'user', atts, chatId);
      const assistantMessageId = addMessage('', 'assistant', undefined, chatId);
      if (!assistantMessageId) return;

      if (firstImageDataUrl) {
        updateMessageContent(
          assistantMessageId,
          'Encoding the image and running vision (CLIP + LM) — the first reply can take 1–3+ minutes on Colab.',
          chatId
        );
      }

      const jsonBody = () =>
        JSON.stringify({
          prompt: userText,
          currentModel,
          messages,
          message: userText,
          model: currentModel,
          attachments: atts,
          image: firstImageDataUrl,
        });

      const applyFallback = () => {
        updateMessageContent(assistantMessageId, OFFLINE_ASSISTANT_MESSAGE, chatId);
      };

      const streamAbort = new AbortController();
      const idleMs = 120_000;
      let lastStreamActivity = Date.now();
      const bump = () => {
        lastStreamActivity = Date.now();
      };

      let streamRes: Response;
      try {
        streamRes = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: jsonBody(),
          signal: streamAbort.signal,
        });
      } catch {
        applyFallback();
        return;
      }

      if (streamRes.ok && streamRes.body) {
        const reader = streamRes.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let gotText = false;
        const idleTicker = setInterval(() => {
          if (Date.now() - lastStreamActivity > idleMs) streamAbort.abort();
        }, 3000);
        try {
          while (true) {
            const { done, value } = await reader.read();
            bump();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const line of lines) {
              const t = line.trim();
              if (!t) continue;
              let obj: { text?: string; error?: string };
              try {
                obj = JSON.parse(t) as { text?: string; error?: string };
              } catch {
                continue;
              }
              if (typeof obj.error === 'string' && obj.error.length > 0) {
                updateMessageContent(assistantMessageId, `**Error:** ${obj.error}`, chatId);
                return;
              }
              if (typeof obj.text === 'string') {
                gotText = true;
                updateMessageContent(assistantMessageId, obj.text, chatId);
              }
            }
          }
          const tail = buffer.trim();
          if (tail) {
            try {
              const obj = JSON.parse(tail) as { text?: string; error?: string };
              if (typeof obj.error === 'string' && obj.error.length > 0) {
                updateMessageContent(assistantMessageId, `**Error:** ${obj.error}`, chatId);
                return;
              }
              if (typeof obj.text === 'string') {
                gotText = true;
                updateMessageContent(assistantMessageId, obj.text, chatId);
              }
            } catch {
              /* ignore */
            }
          }
        } finally {
          clearInterval(idleTicker);
        }
        if (gotText) return;
      }

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: jsonBody(),
        });
        if (response.ok) {
          const data = (await response.json()) as { response?: string };
          updateMessageContent(assistantMessageId, data.response ?? '', chatId);
        } else {
          applyFallback();
        }
      } catch {
        applyFallback();
      }
    },
    [
      addMessage,
      chats,
      createNewChat,
      currentChatId,
      currentModel,
      updateMessageContent,
    ]
  );

  const sendResearchToBackend = useCallback(
    async (userTextRaw: string, currentAttachments?: Attachment[]) => {
      const attsEarly = currentAttachments ?? [];
      const { topic, displayLine } = stripResearchCommand(userTextRaw);
      if (!topic.trim() && attsEarly.length === 0) {
        return;
      }

      let chatId = currentChatId;
      if (!chatId) {
        chatId = createNewChat();
      }
      const priorChat = chats.find(c => c.id === chatId);
      const priorMessages = (priorChat?.messages ?? []).map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }));

      const atts = currentAttachments ?? [];
      let augmented = displayLine;
      if (atts.length > 0) {
        const names = atts.map(a => `${a.name} (${a.type})`).join(', ');
        augmented = `[Attached: ${names}]\n\n${displayLine}`;
      }

      const messages = [...priorMessages, { role: 'user' as const, content: augmented }];

      addMessage(displayLine || 'Research', 'user', atts.length > 0 ? atts : undefined, chatId);
      const assistantMessageId = addMessage('', 'assistant', undefined, chatId);
      if (!assistantMessageId) return;

      setDeepResearchActive(true);

      updateMessageFields(
        assistantMessageId,
        {
          answerPhase: 'preliminary',
          content: '_Deep research starting (web → retrieve → verify)…_',
        },
        chatId
      );

      const promptPayload = topic.trim() ? topic.trim() : augmented.trim();

      const bodyJson = JSON.stringify({
        prompt: promptPayload,
        currentModel,
        messages,
        message: promptPayload,
        model: currentModel,
        attachments: atts,
        image: atts.find(a => a.type === 'image' && typeof a.url === 'string')?.url,
      });

      const controller = new AbortController();
      const idleMs = 300_000;
      let last = Date.now();
      const bump = () => {
        last = Date.now();
      };
      let preliminary = '';
      let finalAnswer = '';

      try {
        const res = await fetch('/api/chat/deep', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/x-ndjson, application/json',
          },
          body: bodyJson,
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          const txt = await res.text();
          updateMessageFields(
            assistantMessageId,
            {
              answerPhase: 'idle',
              content:
                `**Deep research unavailable** (${res.status})\n\n\`\`\`\n${txt.slice(0, 900)}\n\`\`\``,
            },
            chatId
          );
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const idleTicker = setInterval(() => {
          if (Date.now() - last > idleMs) controller.abort();
        }, 4000);

        const composeContent = (): string => {
          const chunks: string[] = [];
          if (preliminary.trim()) {
            chunks.push(
              `## Preliminary findings\n\n${preliminary}\n\n---\n\n`
            );
          }
          if (finalAnswer.trim()) {
            chunks.push(`## Answer\n\n${finalAnswer}`);
          }
          const c = chunks.join('');
          return c.trim() ? c : '_Deep research running…_';
        };

        try {
          while (true) {
            const { done, value } = await reader.read();
            bump();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const rawLine of lines) {
              const line = rawLine.trim();
              if (!line) continue;
              try {
                const obj = JSON.parse(line) as {
                  type?: string;
                  text?: string;
                  phase?: string;
                  error?: string;
                  step?: string;
                  summary?: string;
                  id?: string;
                  detail?: Record<string, unknown>;
                };
                if (typeof obj.error === 'string' && obj.error) {
                  updateMessageFields(
                    assistantMessageId,
                    {
                      answerPhase: 'idle',
                      content: `**Error:** ${obj.error}`,
                    },
                    chatId
                  );
                  return;
                }
                if (obj.type === 'trace') {
                  appendResearchTrace(
                    assistantMessageId,
                    {
                      id: typeof obj.id === 'string' && obj.id ? obj.id : generateId(),
                      ts: Date.now(),
                      step: typeof obj.step === 'string' ? obj.step : 'trace',
                      summary: typeof obj.summary === 'string' ? obj.summary : '',
                      detail: obj.detail && typeof obj.detail === 'object' ? obj.detail : undefined,
                    },
                    chatId
                  );
                  continue;
                }
                if (obj.type === 'token' && typeof obj.text === 'string') {
                  const phase = (obj.phase || 'final').toLowerCase();
                  if (phase === 'preliminary') {
                    preliminary = obj.text;
                    updateMessageFields(
                      assistantMessageId,
                      {
                        answerPhase: 'preliminary',
                        content: composeContent(),
                      },
                      chatId
                    );
                  } else {
                    finalAnswer = obj.text;
                    updateMessageFields(
                      assistantMessageId,
                      {
                        answerPhase: 'final',
                        content: composeContent(),
                      },
                      chatId
                    );
                  }
                }
              } catch {
                /* ignore malformed line */
              }
            }
          }
          const tail = buffer.trim();
          if (tail) {
            try {
              const obj = JSON.parse(tail) as {
                type?: string;
                text?: string;
                phase?: string;
                error?: string;
              };
              if (obj.type === 'token' && typeof obj.text === 'string') {
                if ((obj.phase || '').toLowerCase() === 'preliminary') preliminary = obj.text;
                else finalAnswer = obj.text;
                updateMessageFields(
                  assistantMessageId,
                  { answerPhase: 'final', content: composeContent() },
                  chatId
                );
              }
            } catch {
              /* ignore */
            }
          }
        } finally {
          clearInterval(idleTicker);
        }

        updateMessageFields(
          assistantMessageId,
          { answerPhase: finalAnswer.trim() ? 'final' : 'idle', content: composeContent() },
          chatId
        );
      } catch {
        updateMessageFields(
          assistantMessageId,
          {
            answerPhase: 'idle',
            content: OFFLINE_ASSISTANT_MESSAGE,
          },
          chatId
        );
      } finally {
        setDeepResearchActive(false);
      }
    },
    [
      addMessage,
      appendResearchTrace,
      chats,
      createNewChat,
      currentChatId,
      currentModel,
      updateMessageFields,
    ]
  );

  const sendAgentChatToBackend = useCallback(
    async (userText: string, attachments?: Attachment[]) => {
      let cid = currentChatId;
      if (!cid) {
        cid = createNewChat();
      }

      const priorMessages = (chats.find(c => c.id === cid)?.messages ?? []).map(m => ({
        role: m.role,
        content: m.content,
      }));

      let augmented = userText;
      if (attachments?.length) {
        const names = attachments.map(a => `${a.name} (${a.type})`).join(', ');
        augmented = `[Attached: ${names}]\n\n${userText}`;
      }

      const history = [...priorMessages, { role: 'user' as const, content: augmented }];

      addMessage(userText, 'user', attachments, cid);
      setShowAgentComposerUi(true);
      setIsAgentRunning(true);
      setTerminalRuns([]);
      setDiffProposals([]);
      setMediaArtifacts([]);
      setSummaryReport(null);
      setAgentTrace([]);
      setWriteFileStream(null);

      const body = {
        prompt: userText,
        currentModel,
        messages: history,
        session_id: agentSessionId ?? undefined,
        max_tool_iters: 28,
        confirm_apply: false,
        allow_network_tools: process.env.NEXT_PUBLIC_UI_ARTISAN_MODE === '1' ? false : true,
        ui_artisan_mode: process.env.NEXT_PUBLIC_UI_ARTISAN_MODE === '1',
      };

      try {
        const res = await fetch('/api/chat/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
          body: JSON.stringify(body),
        });
        if (!res.ok || !res.body) {
          addMessage(OFFLINE_ASSISTANT_MESSAGE, 'assistant', undefined, cid);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        /** Let React paint between NDJSON rows (otherwise one batched update at end of loop). */
        const yieldToUi = () => new Promise<void>(r => setTimeout(r, 0));

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
              const ev = JSON.parse(trimmed) as AgentEvent;
              flushSync(() => {
                handleAgentEvent(ev);
              });
              await yieldToUi();
            } catch {
              /* ignore */
            }
          }
        }
        const tail = buffer.trim();
        if (tail) {
          try {
            const ev = JSON.parse(tail) as AgentEvent;
            flushSync(() => {
              handleAgentEvent(ev);
            });
            await new Promise<void>(r => setTimeout(r, 0));
          } catch {
            /* ignore */
          }
        }
      } catch {
        addMessage(OFFLINE_ASSISTANT_MESSAGE, 'assistant', undefined, cid);
        return;
      } finally {
        setIsAgentRunning(false);
      }

      setAgentTrace(prev => {
        if (prev.some(e => e.kind === 'final')) return prev;
        return [
          ...prev,
          {
            id: generateId(),
            ts: Date.now(),
            kind: 'final',
            text:
              'Agent loop completed. Review terminal output, diffs, and summary cards below.',
          },
        ];
      });
      await refreshWorkspaceTree();
    },
    [
      addMessage,
      agentSessionId,
      chats,
      createNewChat,
      currentChatId,
      currentModel,
      handleAgentEvent,
      refreshWorkspaceTree,
    ]
  );

  const applyDiffProposals = useCallback(
    async (proposalIds: string[]) => {
      if (!agentSessionId || proposalIds.length === 0) return;
      const res = await fetch('/api/agent/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ session_id: agentSessionId, proposal_ids: proposalIds }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { applied?: Array<{ proposal_id: string }> };
      const appliedIds = new Set((data.applied || []).map(x => x.proposal_id));
      if (appliedIds.size === 0) return;
      setDiffProposals(prev =>
        prev.map(d => (appliedIds.has(d.proposalId) ? { ...d, applied: true } : d))
      );
      await refreshWorkspaceTree();
    },
    [agentSessionId, refreshWorkspaceTree]
  );

  return (
    <ChatContext.Provider
      value={{
        chats,
        currentChatId,
        currentModel,
        user,
        isAuthenticated: !!user,
        showAuthModal,
        setShowAuthModal,
        storageHydrated,
        deepResearchActive,
        setDeepResearchActive,
        createNewChat,
        selectChat,
        deleteChat,
        addMessage,
        updateMessageContent,
        updateMessageFields,
        appendResearchTrace,
        setCurrentModel,
        signIn,
        signOut,
        getCurrentChat,
        exportChatsJsonl,
        syncChatsFromBackend,
        pushChatsToBackend,
        sendChatToBackend,
        sendResearchToBackend,
        sendAgentChatToBackend,
        applyDiffProposals,
        refreshWorkspaceTree,
        agentSessionId,
        workspaceSnapshot,
        terminalRuns,
        diffProposals,
        mediaArtifacts,
        summaryReport,
        isAgentRunning,
        agentTrace,
        writeFileStream,
        showAgentComposerUi,
        hideAgentComposerUi,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChatContext() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return context;
}
