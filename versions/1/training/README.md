# Imagination 2 — training layout (`versions/v1.4/training`)

This folder is the **canonical place** for coding (and related) **dataset exports**, **QLoRA runs**, **eval tasks**, and **vision bundle artifacts** used while iterating toward **Imagination 2**.

## Directory map

| Path | Purpose |
|------|---------|
| [`datasets/downloaded/`](datasets/downloaded/) | Raw / HF-export JSONL (from `scripts/download_datasets.py` or manual copies). **Not committed** (see `.gitignore`). |
| [`datasets/processed/`](datasets/processed/) | Cleaned / merged / deduped JSONL ready for `train_imagination2_coding.py`. |
| [`outputs/adapters/`](outputs/adapters/) | QLoRA / LoRA checkpoints (`adapter_model.safetensors`, tokenizer, config). Point inference `PEFT` / adapter path here. |
| [`outputs/vision_projector/`](outputs/vision_projector/) | **CLIP + projector bundle** for multimodal: place `projector.pt` + `attach_vision_multimodal_meta.json` here (or symlink). At runtime set `IMAGINATION_VISION_PROJECTOR_DIR` to this directory (or copy the bundle next to your HF LM). |
| [`outputs/logs/`](outputs/logs/) | Optional trainer logs / copies of stdout. |
| [`eval/`](eval/) | Task files + `run_coding_eval.py` smoke / regression checks. |
| [`config/`](config/) | Example env / path templates. |

## Hugging Face “it’s only downloading README.md”

The `datasets` library often prints **dataset card / README** files into the cache the first time you touch a hub dataset. That is **normal** and **not** the same as your output JSONL.

Training lines are built from the **`train` split** of the coding datasets inside `scripts/hf_build_train_jsonl.py`.

If your **JSONL** content looks like distilled documentation instead of coding Q&A, you were probably ingesting local markdown via **`--docs-dir` / `--docs-glob`**. The wrapper `scripts/download_datasets.py` now passes **`--doc-quota 0`** so local README harvesting is off unless you override after `--`.


1. **Mount Drive** and ensure this repo (or a copy) is under e.g. `/content/imagination-v1.1.0`.
2. **Base model** on disk: a Hugging Face folder with `config.json` + weights (often under `IMAGINATION_ROOT` / `modules/reasoning/...`).
3. **Install** (after Colab’s `torch`):

   ```bash
   cd /content/imagination-v1.1.0/versions/v1.4/training
   pip install -q -r requirements.txt
   pip install -q -r ../../scripts/requirements_train_export.txt   # HF dataset export script deps
   ```

4. **Download / build coding JSONL** (delegates to monorepo `scripts/hf_build_train_jsonl.py`):

   ```bash
   python scripts/download_datasets.py --target 5000 -- \
     --min-fact-score 2.0 --min-user-chars 40 --min-assistant-chars 80
   ```

   Output default: `datasets/downloaded/coding_hf_export.jsonl`.

5. **(Optional)** Copy or symlink into `datasets/processed/coding_sft.jsonl` and edit / filter.

6. **Train Imagination 2 (coding QLoRA)**:

   ```bash
   export IMAGINATION2_BASE_MODEL="/content/drive/MyDrive/models/your-hf-model"
   python scripts/train_imagination2_coding.py \
     --model_path "$IMAGINATION2_BASE_MODEL" \
     --dataset datasets/processed/coding_sft.jsonl \
     --output_dir outputs/adapters/imagination2-pass01
   ```

   If **bitsandbytes** fails on Colab, add `--no_4bit` (uses more VRAM; L4 24GB is often enough for 7B-class bf16 LoRA).

7. **Eval / smoke**:

   ```bash
   python scripts/smoke_training_env.py
   python eval/run_coding_eval.py --tasks eval/coding_tasks_smoke.jsonl
   ```

## Vision projector (separate from coding LoRA)

- **Coding SFT** writes to **`outputs/adapters/`**.
- **Projector CPT** (CLIP → LLM) is trained with the repo’s **`pretrain-vision-8b/`** pipeline, not reimplemented here. After CPT, copy **`projector.pt`** + **`attach_vision_multimodal_meta.json`** into **`outputs/vision_projector/`** (or keep in `vision_cpt_out_real/` at repo root and set `IMAGINATION_VISION_PROJECTOR_DIR`).
- Run bundle smoke from repo root: `python pretrain-vision-8b/scripts/smoke_test_clip_projector_bundle.py` (see that script for token/layout checks).

## Environment variables (reference)

| Variable | Role |
|----------|------|
| `IMAGINATION_ROOT` | Root containing `modules/` layout for inference (optional for training if `--model_path` is explicit). |
| `IMAGINATION_VISION_PROJECTOR_DIR` | Absolute path to folder with `projector.pt` + `attach_vision_multimodal_meta.json`. Point at `training/outputs/vision_projector` if you store the bundle there. |
| `HF_TOKEN` | Only if you pull **gated** datasets from Hugging Face. |

## Naming

- **Imagination 2** = your **released** combination of base weights + adapter(s) + optional vision bundle + eval sign-off. Intermediate folders can use `outputs/adapters/imagination2-passNN`.
