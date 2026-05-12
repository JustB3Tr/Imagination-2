export interface Attachment {
  id: string;
  name: string;
  type: 'image' | 'document' | 'code';
  mimeType: string;
  size: number;
  url: string; // Data URL or blob URL
  content?: string; // For text/code files
}

/** Structured research / deep-mode trace event (mirrors backend NDJSON trace payloads). */
export interface ResearchTraceEvent {
  id: string;
  ts: number;
  step: string;
  summary: string;
  detail?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  attachments?: Attachment[];
  createdAt: Date;
  /** Populated during deep research streaming */
  researchTrace?: ResearchTraceEvent[];
  answerPhase?: 'idle' | 'preliminary' | 'final';
}

export type ModelType = 'imagination-1.3' | 'imagination-1.3-pro' | 'imagination-1.3-coder' | 'imagination-1.4-ui';

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

export interface WorkspaceTreeNode {
  name: string;
  path: string;
  type: 'file' | 'dir';
  size?: number;
  truncated?: boolean;
  children?: WorkspaceTreeNode[];
}

export interface WorkspaceSnapshot {
  root: string;
  session_id: string;
  tree: WorkspaceTreeNode | null;
  truncated?: boolean;
}

export type TerminalStatus = 'running' | 'success' | 'fail';

export interface TerminalRun {
  id: string;
  command: string;
  cwd?: string;
  stdout?: string;
  stderr?: string;
  exitCode?: number;
  status: TerminalStatus;
}

export interface DiffProposal {
  proposalId: string;
  path: string;
  diff: string;
  applied: boolean;
}

export interface SummaryReport {
  session_id: string;
  workspace_root: string;
  files_modified: Array<{ path: string; why: string }>;
  commands_run: string[];
  captures: Array<{ artifact_id: string; kind: string; path: string }>;
  pending_proposals: Array<{ proposal_id: string; path: string; reason: string }>;
}

export interface MediaArtifact {
  artifactId: string;
  sessionId: string;
  kind: 'screenshot' | 'video';
  mimeType: string;
  src: string;
}

/** Optional structured visual self-review (v1.4 UI Artisan agent). */
export interface VisualReportPayload {
  schema_version?: string;
  viewport?: { width_px?: number; height_px?: number };
  summary?: string;
  alignment_notes?: string[];
  spacing_notes?: string[];
  a11y_flags?: string[];
  regions?: Array<{
    id: string;
    label: string;
    bbox_norm: { x: number; y: number; w: number; h: number };
    role?: string;
  }>;
  expected_after_next_edit?: string;
}

/** Live NDJSON trace row for /agent (thought, tools, errors). */
export interface AgentTraceEntry {
  id: string;
  ts: number;
  kind:
    | 'thought'
    | 'tool_call'
    | 'tool_result'
    | 'error'
    | 'final'
    | 'ui_artisan_revision'
    | 'ui_artisan_final';
  name?: string;
  text?: string;
  callId?: string;
  detail?: string;
  ok?: boolean;
  visualReport?: VisualReportPayload;
  codeRevisionId?: string;
}

/** Live typewriter preview for a write_file tool call (UI only). */
export interface WriteFileStreamState {
  callId: string;
  path: string;
  fullText: string;
  revealed: number;
  complete: boolean;
}

/** Persisted Composer session (trace, workspace, diffs) so refresh / chat switch can restore UI. */
export interface AgentChatMemory {
  sessionId?: string;
  /** Backend workspace root (folder with session files); kept if tree snapshot is omitted. */
  workspaceRoot?: string;
  workspaceSnapshot?: WorkspaceSnapshot | null;
  agentTrace: AgentTraceEntry[];
  diffProposals: DiffProposal[];
  terminalRuns: TerminalRun[];
  mediaArtifacts: MediaArtifact[];
  summaryReport?: SummaryReport | null;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  model: ModelType;
  /** Saved agent / workspace state for this thread (IndexedDB + optional sync). */
  agentMemory?: AgentChatMemory | null;
}
