# Imagination v1.4 — UI Artisan (graphical UI + visual self-review)

This line is a fork of the v1.3 stack focused on **HTML/CSS UI**, **Playwright capture**, and a shared **`visual_report`** JSON schema so users can give feedback in natural language aligned with what the model claims to see.

## Environment flags

| Variable | Effect |
|----------|--------|
| `IMAGINATION_UI_ARTISAN_MODE=1` | Agent loop uses **read_file / write_file / capture_ui** only and the UI Artisan system prompt (can also be set per request: `ui_artisan_mode` on `POST /api/chat/agent`). |
| `IMAGINATION_UI_LOOP=1` | Allows `capture_ui` with **`file:///`** URLs for HTML files **under the session workspace** (even when `allow_network_tools` is false). |
| `IMAGINATION_AGENT_ENABLE_CAPTURE=0` | Disables Playwright capture entirely. |

## Protocol

- **Structured reviews:** `imagination_runtime/ui_artisan_protocol.py` defines `VISUAL_REPORT_SCHEMA_VERSION`, `validate_visual_report()`, and `example_visual_report()`.
- **NDJSON events:** The agent stream may include `ui_artisan_revision` (after each `write_file` in UI mode) and `ui_artisan_final` (when `final_answer` is an object with `visual_report`).

## Training (Imagination 2 coding + outputs)

See **[`training/README.md`](training/README.md)** for dataset download defaults, QLoRA training script, eval layout, and **where to place `projector.pt`** (`training/outputs/vision_projector/`).

## UI Artisan (UI-focused) model training

1. **Stage A (text):** `python train_ui_v14.py --model_path ... --dataset datasets/ui_artisan_text_sft_seed.jsonl --output_dir adapters/0001`
2. **Stage B (vision-aligned JSONL):** Run `scripts/generate_ui_artisan_synthetic.py` then `python train_ui_vision_v14.py --dataset ...` (validates rows + image paths; uses the same QLoRA path as Stage A with screenshot path echoed in the user turn for traceability — swap in your multimodal template when your base model expects `<image>` tokens).
3. **Stage C (preferences):** See `datasets/ui_preference_pair_example.jsonl` and `imagination_runtime/ui_artisan_protocol.preference_jsonl_schema_doc()`.

## Evaluation

Run `python evals/layout_metrics.py --cases evals/fixture_cases.json` after exporting fixtures from the synthetic generator (or hand-written `cases` JSON).

## Embedded frontend

Set `NEXT_PUBLIC_UI_ARTISAN_MODE=1` before `next build` to send `ui_artisan_mode: true` and disable general network tools for the agent request (`allow_network_tools: false`), relying on `file://` capture when `IMAGINATION_UI_LOOP=1` on the backend.
