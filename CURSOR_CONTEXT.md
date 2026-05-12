# Imagination 2 - Cursor Long Context

Use this file as persistent handoff context in Cursor CLI and Agent sessions:

- `@CURSOR_CONTEXT.md` for full project context
- `@README.md` for quick structure
- `@versions/1/...` for concrete implementation files

This file is intentionally long-form and append-friendly.

---

## 1) Project identity

`Imagination 2` is a standalone repo rooted at `G:\My Drive\Imagination 2`.

It starts from a copied runtime baseline:

- Source baseline: `G:\My Drive\imagination-v1.1.0\versions\v1.4`
- Current runtime copy: `versions/1`

**Naming:** In this repo, **`versions/1` is that v1.4 stack** — the canonical “runtime generation 1” for Imagination 2, regardless of older internal docs that referenced `versions/v1.2` or `imagination_v1_2.py`. On disk you will still see **`imagination_v1_4.py`** and other **v1.4** filenames; that is expected.

The goal is to evolve model-2 releases without rewriting the full stack each time.

---

## 2) Top-level hierarchy

- `versions/1` - runtime + API + frontend baseline copied from old v1.4
- `model` - base model directory used by training and inference
- `training` - top-level model-2 training entrypoints/wrappers
- `scripts/model` - model acquisition and integrity checks
- `scripts/training` - data and training utility scripts
- `adapters/text_adapters` - text LoRA/QLoRA outputs
- `adapters/vision_adapters` - vision adapters/projector bundles

---

## 3) Runtime baseline notes

`versions/1` is treated as the first generation runtime for model-2.

Typical runtime flow in this baseline:

- backend routes and chat orchestration under `versions/1/imagination_runtime`
- optional full-stack orchestration via `versions/1/colab_fullstack_orchestrator.py`
- embedded frontend under `versions/1/front&back/frontend`

When making a new runtime generation, branch by directory:

- `versions/2`, `versions/3`, ...

Do not rename/delete `versions/1`; keep it as historical baseline.

---

## 4) Model strategy

Base model currently targeted:

- `meta-llama/Llama-3.1-8B-Instruct`

Downloaded files are kept in:

- `model/`

Adapters are kept separate:

- text: `adapters/text_adapters/`
- vision: `adapters/vision_adapters/`

This preserves a clean base+adapter workflow:

1. Keep one canonical base model folder.
2. Train adapters per run/release.
3. Load base + chosen adapter at inference time.

---

## 5) Training context

Top-level wrapper:

- `training/train_imagination2.py`

It forwards execution to:

- `versions/1/training/scripts/train_imagination2_coding.py`

Expected dataset patterns in the copied training stack:

- `messages` JSONL chat-style rows
- optional normalized conversion from app export fields

---

## 6) Scripts policy

All scripts live under `scripts/` in labeled subfolders:

- `scripts/model/*` - model download/check
- `scripts/training/*` - training utilities (reserved for future additions)

Current required scripts:

- `scripts/model/download_llama31_8b.ps1` (wrapper)
- `scripts/model/download_llama31_8b.py` (actual `snapshot_download` implementation)
- `scripts/model/verify_model_files.ps1`
- `scripts/hf_build_train_jsonl.py` (HF coding JSONL export; used by `versions/1/training/scripts/download_datasets.py`)
- `scripts/requirements_train_export.txt` (`datasets`, `tqdm` for that script)

---

## 7) Colab and local behavior

For Colab:

- prefer mounting Drive and pointing to this repo root
- keep downloaded base model in `model/` to avoid path drift
- run wrappers from repo root for consistent relative paths

For local Windows:

- PowerShell scripts are primary for model download/validation
- Python wrappers keep training invocation consistent

---

## 8) Session changelog seed

- Bootstrapped standalone repo for model-2 line
- Copied old v1.4 runtime to `versions/1`
- Added dedicated model and adapters hierarchy
- Added model download + validation scripts
- Added `scripts/hf_build_train_jsonl.py` and `scripts/requirements_train_export.txt` at repo root (same as parent monorepo) so `versions/1/training/scripts/download_datasets.py` finds the default builder without `--hf-script`

Append future decisions below this section as the source of truth for Cursor sessions.

---

## 9) Full conversation archive and implementation log (2026-05-07)

This section records the **thread of decisions and explanations** from the parent `imagination-ai` workspace chat plus the **bootstrap work** that created **this** repo. It is meant so you can open this repo cold in Cursor and still have narrative context.

### 9.1 Monorepo context (parent repo: `G:\My Drive\imagination-v1.1.0`)

- **`scripts/hf_build_train_jsonl.py`**: builds `train.jsonl` from streamed HF coding mixes with fact-density filtering and dedupe. A shortfall vs `--target` (e.g. 998 vs 1000) was addressed by a **backfill loop** (relax `--min-fact-score` toward `--backfill-floor`) and larger scan caps in `harvest()`. User saw a warning when under target before that change.
- **`CURSOR_CLI_SESSION_CONTEXT.md`** was added at monorepo root as a long handoff doc for Cursor CLI (`@` references, resume notes).
- **Cursor Agent CLI OOM**: Node heap ~4GB; mitigations include `NODE_OPTIONS=--max-old-space-size=8192`, smaller `@` context, `/compress`, avoid huge review sessions, prefer local disk over Google Drive for heavy CLI work.
- **`versions/v1.4/training/`**: canonical layout for Imagination 2 coding data + QLoRA (`README.md`, `scripts/train_imagination2_coding.py`, `scripts/download_datasets.py` delegating to monorepo `scripts/hf_build_train_jsonl.py`, eval smoke, `config/paths.env.example`).
- **CPT vs SFT for the user’s historical 3B**: repo evidence (`versions/v1.2.1/train_imagination.py`) is **QLoRA + TRL `SFTTrainer`** on chat JSONL = **instruction SFT**, not unstructured CPT. Root `config.json` matches **Llama-class ~3B** architecture. `pretrain-vision-8b/scripts/train_text_cpt.py` is a **separate** CPT-style path. **Weights alone do not prove** no external CPT was ever run outside the repo.
- **Scaling / “10B”**: LoRA/QLoRA trains **adapters on a chosen base**; it does not “convert” 3B tensors into 10B by config edit. Real larger capacity needs **a larger base checkpoint** and/or **distillation / expansion + CPT** (heavy). Honest product story: **disclose base + own training/adapters/system**.
- **Adapter ownership**: the **adapter weights you train are yours**; claiming “I made the whole dense foundation from scratch” is inaccurate if a public base remains underneath. Use precise wording + follow base **license** terms.
- **Colab L4 “dumbed down” path**: mount Drive → install deps → build JSONL → QLoRA train → eval → point backend at base+adapter. Rough times: downloads/install dominate first run; QLoRA often **hours** depending on rows and `max_seq_length`.

### 9.2 User request: standalone “Imagination 2” repo on `G:`

**Asked for:**

- New folder on `G:` named **Imagination 2**
- Copy **`versions/v1.4`** into a **new version folder** that represents **model 2 version 1** (not “v1.4” in name) — agreed as **`versions/1`**
- **`model/`** at repo root with HF model files inside
- **`training/`** with a training script at top level
- **`CURSOR_CONTEXT.md`** at root (long-form, no artificial size cap)
- **`scripts/`** with clearly labeled subfolders
- Two adapter roots: **`adapters/text_adapters`** and **`adapters/vision_adapters`** (normalized names; user rejected literal `{model}` / spaced names)
- **Initialize its own git repo**
- **Run** the download for `meta-llama/Llama-3.1-8B-Instruct` (not script-only)

### 9.3 Filesystem constraint (important)

On this machine, **`G:\` only contained `My Drive`** as a visible root folder; creating **`G:\Imagination 2` failed** (`Test-Path` false, `git init` could not target it).

**Actual repo root used:**

- `G:\My Drive\Imagination 2`

Treat this as the canonical “Imagination 2” workspace unless the user later maps a true `G:\Imagination 2` drive letter path.

### 9.4 What was created / copied (this repo)

**Directories**

- `versions/1/` — **robocopy** mirror of `G:\My Drive\imagination-v1.1.0\versions\v1.4` (includes nested `training/`, `imagination_runtime/`, `front&back/`, etc.)
- `model/` — populated by HF download
- `training/` — top-level training entrypoint
- `scripts/model/` — download + verify
- `scripts/training/` — placeholder README for future helpers
- `adapters/text_adapters/` — empty placeholder for LoRA outputs
- `adapters/vision_adapters/` — empty placeholder for vision bundles

**New / authored files in this repo (not copied from v1.4)**

- `README.md` — layout map + quick start commands
- `CURSOR_CONTEXT.md` — this file (expanded by this section)
- `training/train_imagination2.py` — **wrapper** that forwards argv to `versions/1/training/scripts/train_imagination2_coding.py`
- `scripts/model/download_llama31_8b.ps1` — PowerShell entrypoint
- `scripts/model/download_llama31_8b.py` — Python `huggingface_hub.snapshot_download` implementation (added after initial failure)
- `scripts/model/verify_model_files.ps1` — checks `config.json`, tokenizer files, and weight shards/index
- `scripts/training/README.md` — notes reserved folder

**Git**

- `git init` at `G:\My Drive\Imagination 2`
- **No commit** was made (working tree shows untracked paths as of bootstrap)

### 9.5 Model download execution notes

- Background task **80123** (first attempt) **failed** with a **PowerShell inline Python quoting** / `SyntaxError` in the one-liner approach.
- **Fix**: move download logic into **`scripts/model/download_llama31_8b.py`**; keep **`download_llama31_8b.ps1`** as a thin wrapper calling `python ...`.
- **Re-run succeeded**; `verify_model_files.ps1` reports valid model dir.
- Observed downloaded artifacts under `model/` include at least: `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json`, `model-00001-of-00004.safetensors` … `model-00004-of-00004.safetensors`, plus hub metadata files (e.g. `.gitattributes`, `README.md`, `LICENSE`, `USE_POLICY.md`, `.cache/`).

### 9.6 Operational commands (from this repo root)

PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\model\download_llama31_8b.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\model\verify_model_files.ps1"
python .\training\train_imagination2.py --help
```

### 9.7 Explicit non-changes

- The parent monorepo **`G:\My Drive\imagination-v1.1.0`** was not modified by the bootstrap copy (read-only source for `robocopy`).
- The attached **plan file** in Cursor metadata was **not edited** per user instruction.

### 9.8 Suggested next edits (optional, not done)

- Add a **`.gitignore`** that ignores `model/*.safetensors`, `model/.cache`, adapter outputs, and large artifacts if you intend to commit this repo to git hosting.
- Add an initial **`git commit`** once you decide what should be tracked vs Drive-only.
- If you truly need `G:\Imagination 2` as a path, create a **junction** from `G:\Imagination 2` → `G:\My Drive\Imagination 2` *only if* the OS/drive policy allows creating new entries at `G:\` (it did not in the bootstrap environment).

