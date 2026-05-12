# Imagination Composer Agentic Engine (v1.4) - Implementation Handoff

This document explains the exact changes implemented for the **Imagination Composer Agentic Engine** in the v1.4 fullstack target:

- Backend: `versions/v1.4/imagination_runtime`
- Frontend: `versions/v1.4/front&back/frontend`

It is written as a technical handoff you can send to Gemini for continuation/refinement.

---

## 1) What Was Implemented

Implemented end-to-end agentic loop plumbing with:

1. **Persistent session-backed agent loop** (`AgenticLoop`) with tool calling:
   - `read_file`
   - `write_file` (diff proposal staging + optional apply)
   - `run_shell`
   - `web_search`
   - `capture_ui` (Playwright screenshot + optional video metadata)
2. **Self-correction logic** for failing shell commands:
   - Failed `run_shell` appends a system correction prompt and continues loop iterations.
3. **NDJSON trace protocol** from backend to frontend:
   - `session`, `workspace_tree`, `thought`, `tool_call`, `tool_result`, `diff_preview`, `media`, `summary`, `final`, `error`
4. **New FastAPI routes** for agent workflow:
   - `POST /api/chat/agent`
   - `GET /api/agent/workspace`
   - `POST /api/agent/apply`
   - `GET /api/agent/capture/{session_id}/{artifact_id}`
5. **Frontend integration**:
   - Agent stream parser and event state in chat context
   - File tree UI in sidebar
   - Terminal run cards with status badges
   - Diff preview cards with Apply action
   - Summary report card
   - Capture media rendering (image/video)
6. **Dependency update**:
   - Added `playwright` to `versions/v1.4/requirements.txt`

---

## 2) File-by-File Change Map

## Backend

- **Added:** `versions/v1.4/imagination_runtime/agent_loop.py`
  - Core agent runtime + session state + tools + summary generation.
- **Updated:** `versions/v1.4/imagination_runtime/chat_http.py`
  - New request models and API routes for agent orchestration and apply/capture/workspace APIs.
- **Updated:** `versions/v1.4/requirements.txt`
  - Added `playwright`.

## Frontend API routes

- **Added:** `versions/v1.4/front&back/frontend/app/api/chat/agent/route.ts`
  - Proxies NDJSON stream to backend `/api/chat/agent`.
- **Added:** `versions/v1.4/front&back/frontend/app/api/agent/workspace/route.ts`
  - Proxies workspace tree route.
- **Added:** `versions/v1.4/front&back/frontend/app/api/agent/apply/route.ts`
  - Proxies diff proposal apply route.
- **Added:** `versions/v1.4/front&back/frontend/app/api/agent/capture/[sessionId]/[artifactId]/route.ts`
  - Proxies capture artifact binary fetch.

## Frontend state + UI

- **Updated:** `versions/v1.4/front&back/frontend/lib/chat-context.tsx`
  - Added streaming agent event handling and shared state slices.
- **Updated:** `versions/v1.4/front&back/frontend/lib/types.ts`
  - Added types for workspace tree, terminal runs, diffs, media, summary report.
- **Updated:** `versions/v1.4/front&back/frontend/components/chat-sidebar.tsx`
  - Added FileTree panel + manual refresh button.
- **Updated:** `versions/v1.4/front&back/frontend/components/chat-messages.tsx`
  - Added terminal/diff/media/summary rendering blocks.
- **Updated:** `versions/v1.4/front&back/frontend/components/chat-input.tsx`
  - Disabled sending while agent loop is running, updated status text.
- **Updated:** `versions/v1.4/front&back/frontend/components/math-renderer.tsx`
  - Added rendering support for fenced `terminal` blocks.
- **Added:** `versions/v1.4/front&back/frontend/components/file-tree.tsx`
- **Added:** `versions/v1.4/front&back/frontend/components/terminal-card.tsx`
- **Added:** `versions/v1.4/front&back/frontend/components/diff-preview-card.tsx`
- **Added:** `versions/v1.4/front&back/frontend/components/summary-report-card.tsx`
- **Added:** `versions/v1.4/front&back/frontend/components/media-artifacts.tsx`

---

## 3) Backend Design Details

## 3.1 Session Model

`agent_loop.py` keeps in-memory session state keyed by `session_id`:

- `workspace_root`
- pending `proposals` (diffs not applied yet)
- capture `artifacts`
- `modified_files` map with rationale
- `commands_run` list

Session roots default to:

`<IMAGINATION_ROOT>/temp/agent_workspaces/<session_id>`

unless `workspace_root` is explicitly passed.

## 3.2 Tool Behaviors

### `read_file`
- Path normalized and constrained to workspace.
- Returns:
  - `exists`
  - `content`
  - `sha256`
  - `truncated` flag (size-capped read)

### `write_file`
- Produces unified diff (`a/...` vs `b/...`).
- If `confirm_apply=false`:
  - Stores proposal in session and emits `diff_preview` event.
- If `confirm_apply=true`:
  - Writes directly and records modified file in summary state.

### `run_shell`
- Executes with `shell=False` and parsed argv.
- Honors workspace-relative optional `cwd`.
- Captures truncated stdout/stderr.
- Emits status payload:
  - `status: success|fail`
  - `exit_code`
  - `stdout`
  - `stderr`

### `web_search`
- Uses existing `fetch_web_context(...)` from `chat_web_research.py`.
- Guarded by request toggle `allow_network_tools`.

### `capture_ui`
- Uses Playwright Chromium in headless mode.
- Supports screenshot, optional video metadata path.
- Supports URL allowlist via env var.
- Stores artifact metadata in session, emits `media` event.

## 3.3 Self-Correction Loop

When tool call is `run_shell` and result is non-zero:

- A system message is appended with failure diagnostics and fix instruction.
- Loop continues until success, final answer, or `max_tool_iters` reached.

## 3.4 SummaryReport

Final report is built from actual trace/session state (not freeform only):

- `files_modified`
- `commands_run`
- `captures`
- `pending_proposals`
- `session_id`, `workspace_root`

---

## 4) API Contract (Current)

## 4.1 `POST /api/chat/agent`

Request body extends existing chat payload:

```json
{
  "prompt": "string",
  "currentModel": "imagination-1.3",
  "messages": [{"role":"user","content":"..."}],
  "max_new_tokens": 1024,
  "session_id": "optional",
  "workspace_root": "optional",
  "max_tool_iters": 10,
  "confirm_apply": false,
  "allow_network_tools": true
}
```

Response is `application/x-ndjson`, one JSON object per line.

## 4.2 `GET /api/agent/workspace`

Query:

- `session_id` (optional)
- `workspace_root` (optional)

Returns workspace tree snapshot.

## 4.3 `POST /api/agent/apply`

```json
{
  "session_id": "required",
  "proposal_ids": ["id1","id2"]
}
```

Applies staged diff proposals.

## 4.4 `GET /api/agent/capture/{session_id}/{artifact_id}`

Returns binary artifact content for screenshot/video.

---

## 5) NDJSON Event Shapes (Observed/Expected)

Typical stream events:

```json
{"type":"session","session_id":"...","workspace_root":"..."}
{"type":"workspace_tree","root":"...","tree":{...}}
{"type":"thought","text":"..."}
{"type":"tool_call","id":"...","name":"run_shell","args":{"command":"..."}}
{"type":"tool_result","id":"...","name":"run_shell","ok":false,"data":{"exit_code":1,"stderr":"..."}}
{"type":"diff_preview","proposal_id":"...","path":"...","diff":"...","applied":false}
{"type":"media","artifact_id":"...","kind":"screenshot","mime_type":"image/png","base64":"..."}
{"type":"summary","report":{...}}
{"type":"final","text":"..."}
{"type":"error","message":"..."}
```

Note: `chat_http.py` currently emits nested `diff_preview`/`media` from `tool_result.data` as top-level events too, for easier frontend consumption.

---

## 6) Frontend Integration Details

## 6.1 Context state additions (`chat-context.tsx`)

Added:

- `agentSessionId`
- `workspaceSnapshot`
- `terminalRuns`
- `diffProposals`
- `mediaArtifacts`
- `summaryReport`
- `isAgentRunning`

And methods:

- `sendAgentChatToBackend(...)`
- `applyDiffProposals(...)`
- `refreshWorkspaceTree()`

Current behavior routes default chat submit through agent path.

## 6.2 Sidebar FileTree

`chat-sidebar.tsx` now includes:

- Workspace panel with refresh button
- `FileTree` recursive node renderer

## 6.3 Tool windows / cards

`chat-messages.tsx` renders:

- `TerminalCard` list (running/success/fail)
- `DiffPreviewCard` list with Apply button
- `MediaArtifacts` block (image/video)
- `SummaryReportCard` at loop end

## 6.4 Terminal block rendering

`math-renderer.tsx` now recognizes fenced:

```text
```terminal
status: success|fail|running
...
```
```

and renders status badge + `<pre>` payload.

---

## 7) Safety + Runtime Controls

Current controls implemented:

- Workspace path confinement via relative resolution checks.
- Read and output truncation for files and shell logs.
- Shell can be disabled:
  - `IMAGINATION_AGENT_ALLOW_SHELL=0`
- Capture can be disabled:
  - `IMAGINATION_AGENT_ENABLE_CAPTURE=0`
- Capture host allowlist:
  - `IMAGINATION_AGENT_CAPTURE_ALLOW_HOSTS=host1,host2`
- Request-level network gate:
  - `allow_network_tools` in `/api/chat/agent` body.

---

## 8) Verification Performed

Completed:

- `python -m py_compile` on updated backend modules succeeded.
- IDE lints for touched frontend/backend files reported no errors.

Partially blocked in environment:

- Full runtime smoke script failed due to missing dependency chain (`ddgs` missing at runtime import path in this environment).
- `npm run lint` not runnable in this environment because local `eslint` executable was unavailable.

---

## 9) Known Gaps / Follow-Up Recommendations

These are not blockers for understanding current implementation, but are good next steps:

1. **Session persistence** is in-memory; restart clears session map.
2. **Auth/authorization** around apply/capture routes is not added yet.
3. **Large media transport** currently supports base64 inline + artifact route; consider size thresholding policy.
4. **Diff UX** is minimal custom renderer; can be upgraded to richer split view if desired.
5. **Model tool protocol robustness** can be improved with stricter schema validation/retries.
6. **Background job execution** for long shell/capture tasks is currently synchronous per call.
7. **Dependency readiness checks** should be added for Playwright browsers and DDGS.

---

## 10) Quick Runbook for Gemini

If Gemini is continuing from this state, ask it to:

1. Audit and harden `agent_loop.py` security limits and error handling.
2. Add strict JSON schema validator for model tool outputs.
3. Add backend integration tests for:
   - staged write -> apply flow
   - shell fail -> correction cycle
   - capture route artifact retrieval
4. Add frontend tests for NDJSON parser and UI cards.
5. Add install/setup checks:
   - `pip install -r versions/v1.4/requirements.txt`
   - `python -m playwright install chromium`
6. Optionally rewire regular non-agent chat path if both modes should coexist.

---

## 11) Short Summary You Can Paste Elsewhere

Implemented a full agentic coding loop in v1.4 with NDJSON streaming traces, sandboxed file/shell tools, staged diff proposals + apply endpoint, Playwright capture artifacts, workspace tree sync, and frontend cards for terminal/diff/media/summary. Core backend is in `imagination_runtime/agent_loop.py` and wired via new routes in `imagination_runtime/chat_http.py`, with Next proxy + UI integration under `front&back/frontend`.

