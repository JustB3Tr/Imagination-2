/**
 * IndexedDB persistence for chats + settings (local-first).
 * Debounced writes are handled by the caller; this module exposes load/save/delete.
 */

const DB_NAME = 'imagination-chats-v1';
const DB_VERSION = 1;
const STORE_STATE = 'app_state';

export type PersistedMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  attachments?: import('./types').Attachment[];
  createdAt: string;
  researchTrace?: import('./types').ResearchTraceEvent[];
  answerPhase?: 'idle' | 'preliminary' | 'final';
};

export type PersistedChat = {
  id: string;
  title: string;
  messages: PersistedMessage[];
  createdAt: string;
  model: import('./types').ModelType;
  agentMemory?: import('./types').AgentChatMemory | null;
};

export type PersistedAppState = {
  chats: PersistedChat[];
  currentChatId: string | null;
  currentModel: import('./types').ModelType;
  updatedAt: string;
};

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_STATE)) {
        db.createObjectStore(STORE_STATE, { keyPath: 'key' });
      }
    };
  });
}

const STATE_KEY = 'default';

export async function loadAppState(): Promise<PersistedAppState | null> {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_STATE, 'readonly');
      const store = tx.objectStore(STORE_STATE);
      const r = store.get(STATE_KEY);
      r.onerror = () => reject(r.error);
      r.onsuccess = () => {
        const row = r.result as { key: string; value: PersistedAppState } | undefined;
        resolve(row?.value ?? null);
      };
    });
  } catch {
    return null;
  }
}

export async function saveAppState(state: PersistedAppState): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_STATE, 'readwrite');
    const store = tx.objectStore(STORE_STATE);
    store.put({ key: STATE_KEY, value: state });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export function chatsToJsonlLines(chats: PersistedChat[]): string {
  return chats.map(c => JSON.stringify(c)).join('\n') + (chats.length ? '\n' : '');
}
