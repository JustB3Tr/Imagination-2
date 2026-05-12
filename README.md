# Imagination 2

Standalone workspace for the Imagination 2 model line.

## Layout

- `versions/1/` - copied runtime baseline from the prior `v1.4` stack.
- `model/` - base model files (currently `meta-llama/Llama-3.1-8B-Instruct`).
- `training/` - top-level training entrypoints and wrappers for model-2 iteration.
- `scripts/model/` - model download and validation scripts.
- `scripts/hf_build_train_jsonl.py` (+ `scripts/requirements_train_export.txt`) - HF coding JSONL builder used by training `download_datasets.py`.
- `scripts/training/` - training helpers and data prep scripts.
- `adapters/text_adapters/` - text adapter outputs (LoRA/QLoRA).
- `adapters/vision_adapters/` - vision adapter and projector outputs.
- `CURSOR_CONTEXT.md` - persistent long-form context for Cursor CLI/Agent.

## Quick start

1. Download the base model:
   - `powershell -ExecutionPolicy Bypass -File ".\scripts\model\download_llama31_8b.ps1"`
   - (implementation: `scripts/model/download_llama31_8b.py`)
2. Validate required model files:
   - `powershell -ExecutionPolicy Bypass -File ".\scripts\model\verify_model_files.ps1"`
3. Start training wrapper:
   - `python .\training\train_imagination2.py --help`

## Notes

- This repo was bootstrapped from `versions/v1.4` into `versions/1`.
- Keep new inference/runtime generations under `versions/2`, `versions/3`, etc.
