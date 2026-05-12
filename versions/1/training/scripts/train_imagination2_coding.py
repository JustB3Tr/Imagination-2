#!/usr/bin/env python3
"""
QLoRA supervised fine-tuning for **Imagination 2** (coding / instruction JSONL).

Defaults assume you run from ``versions/1/training``:

  cd versions/1/training
  export IMAGINATION2_BASE_MODEL=/path/to/hf_model_dir
  python scripts/train_imagination2_coding.py

Dataset: ``datasets/processed/coding_sft.jsonl`` (prepare by copying from ``datasets/downloaded/``).
Output: ``outputs/adapters/imagination2-latest``.

Same stack as ``versions/v1.4/train_ui_v14.py`` / v1.2.1 ``train_imagination.py`` (TRL SFTTrainer + PEFT).

**Weights & Biases:** ``pip install wandb``, then ``--wandb_project my-proj`` (optional ``--wandb_run_name``,
``--wandb_entity``). Set ``WANDB_API_KEY`` or run ``wandb login``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def _training_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _messages_from_row(obj: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        out: List[Dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            content = str(m.get("content") or "")
            if role in ("system", "user", "assistant") and content.strip():
                out.append({"role": role, "content": content})
        return out or None

    user_message = (obj.get("user_message") or "").strip()
    answer = (obj.get("answer") or "").strip()
    if not user_message or not answer:
        return None

    task_label = str(obj.get("task_label") or "").strip()
    reasoning = obj.get("reasoning_trace")
    sys_chunks: List[str] = []
    if task_label:
        sys_chunks.append(f"Task / model: {task_label}")
    if isinstance(reasoning, list) and reasoning:
        sys_chunks.append("Reasoning trace (reference):\n" + "\n".join(str(x) for x in reasoning if str(x).strip()))
    elif isinstance(reasoning, str) and reasoning.strip():
        sys_chunks.append("Reasoning trace (reference):\n" + reasoning.strip())

    trace = obj.get("trace")
    if isinstance(trace, dict):
        sl = trace.get("step_lines")
        if isinstance(sl, list) and sl:
            sys_chunks.append("Step trace (reference):\n" + "\n".join(str(x) for x in sl if str(x).strip()))

    system = "\n\n".join(sys_chunks).strip() or "You are Imagination, a helpful assistant."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]


def parse_args() -> argparse.Namespace:
    root = _training_root()
    p = argparse.ArgumentParser(description="QLoRA fine-tune Imagination 2 (coding JSONL)")
    p.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("IMAGINATION2_BASE_MODEL") or os.environ.get("IMAGINATION_ROOT", "."),
        help="HF model directory (env IMAGINATION2_BASE_MODEL overrides IMAGINATION_ROOT default).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=str(root / "datasets" / "processed" / "coding_sft.jsonl"),
        help="JSONL path (relative paths resolve from cwd, then from training/ root).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(root / "outputs" / "adapters" / "imagination2-latest"),
        help="Where to save adapter + tokenizer. Absolute paths are used as-is; "
        "relative paths are under this training folder (versions/1/training/). "
        "On Colab prefer e.g. --output_dir /content/out/my-run",
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help='Comma-separated LoRA targets, or "auto".',
    )
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=("adamw_torch", "paged_adamw_8bit", "adamw_bnb_8bit"),
    )
    p.add_argument(
        "--no_4bit",
        action="store_true",
        help="bf16/fp16 full weights on GPU (no bitsandbytes 4-bit).",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="If set, logs to Weights & Biases (pip install wandb; wandb login or WANDB_API_KEY).",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Optional W&B run name (TrainingArguments.run_name).",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Optional W&B entity/team (or set WANDB_ENTITY).",
    )
    return p.parse_args()


def _resolve_existing_file(raw: str, training_root: Path) -> Path:
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    q = training_root / raw
    if q.is_file():
        return q.resolve()
    return Path(raw).expanduser().resolve()


def _resolve_output_dir(raw: str, training_root: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (training_root / raw).resolve()


def _infer_lora_target_modules(model: Any) -> List[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv_proj",
        "W_pack",
        "query_key_value",
        "c_attn",
        "c_proj",
        "c_fc",
    }
    found: set[str] = set()
    for name, _mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    out = sorted(found)
    return out if out else ["q_proj", "v_proj", "o_proj"]


def _format_supervised_text(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    blocks: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip().upper()
        content = (m.get("content") or "").strip()
        blocks.append(f"### {role}\n{content}\n")
    blocks.append("### ASSISTANT\n")
    return "\n".join(blocks)


def main() -> None:
    args = parse_args()
    training_root = _training_root()
    model_path = os.path.abspath(args.model_path)
    dataset_path = str(_resolve_existing_file(args.dataset, training_root))
    output_dir = str(_resolve_output_dir(args.output_dir, training_root))
    out_p = Path(output_dir)
    try:
        out_p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(
            f"Cannot create output_dir:\n  {out_p.resolve()}\n{e}\n"
            "On Colab use an absolute path, e.g. --output_dir /content/out/imagination2-qlora "
            "(create parent first: os.makedirs('/content/out', exist_ok=True))."
        ) from e
    if not out_p.is_dir():
        raise SystemExit(f"output_dir is not a directory after mkdir: {out_p.resolve()}")

    print(f"[imagination2 train] model_path={model_path}", flush=True)
    print(f"[imagination2 train] dataset_path={dataset_path}", flush=True)
    print(f"[imagination2 train] output_dir={out_p.resolve()}", flush=True)

    if not os.path.isdir(model_path):
        raise SystemExit(f"model_path is not a directory:\n  {model_path}")
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise SystemExit(f"No config.json under model_path:\n  {model_path}")
    if not os.path.isfile(dataset_path):
        raise SystemExit(
            f"Dataset not found: {dataset_path}\n"
            "Copy e.g. datasets/downloaded/coding_hf_export.jsonl → datasets/processed/coding_sft.jsonl "
            "or pass --dataset explicitly."
        )

    wb_proj = (args.wandb_project or "").strip()
    if wb_proj:
        os.environ.setdefault("WANDB_PROJECT", wb_proj)
    wb_ent = (args.wandb_entity or os.environ.get("WANDB_ENTITY") or "").strip()
    if wb_ent:
        os.environ.setdefault("WANDB_ENTITY", wb_ent)
    run_name_wb = (args.wandb_run_name or "").strip() or None

    rows = _load_jsonl_rows(dataset_path)
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        m = _messages_from_row(r)
        if m:
            normalized.append({"messages": m})

    if not normalized:
        raise SystemExit("No valid training examples after parsing JSONL.")

    ds = Dataset.from_list(normalized)
    print(f"[imagination2 train] Loaded {len(ds)} examples from {dataset_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16_weights = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if args.no_4bit:
        dtype = torch.bfloat16 if use_bf16_weights else torch.float16
        print(f"[imagination2 train] Loading base in {dtype} (--no_4bit)", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

    raw_tm = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    if len(raw_tm) == 1 and raw_tm[0].lower() == "auto":
        target_modules = _infer_lora_target_modules(model)
        print(f"[imagination2 train] Inferred LoRA targets: {target_modules}", flush=True)
    else:
        target_modules = raw_tm

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def formatting_func(example: Dict[str, Any]) -> str:
        return _format_supervised_text(tokenizer, example["messages"])

    try:
        _probe = formatting_func(normalized[0])
        if not (_probe or "").strip():
            raise ValueError("empty formatting")
    except Exception as exc:
        raise RuntimeError(
            "Could not format the first training example (chat_template missing or bad messages)."
        ) from exc

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    _base_sft_kw = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim=args.optim,
        report_to="wandb" if wb_proj else "none",
        run_name=run_name_wb,
    )
    sft_config = None
    for extra in (
        {"max_seq_length": args.max_seq_length, "packing": False},
        {"max_seq_length": args.max_seq_length},
        {"max_length": args.max_seq_length, "packing": False},
        {"max_length": args.max_seq_length},
    ):
        try:
            sft_config = SFTConfig(**{**_base_sft_kw, **extra})
            break
        except TypeError:
            continue
    if sft_config is None:
        raise RuntimeError("Could not construct SFTConfig for this trl version.")

    if wb_proj:
        try:
            import wandb  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Install wandb for W&B logging: pip install wandb\n" + str(e)
            ) from e
        print(
            "[imagination2 train] W&B → project="
            f"{wb_proj!r}"
            + (f" run_name={run_name_wb!r}" if run_name_wb else "")
            + " (WANDB_API_KEY or `wandb login`)",
            flush=True,
        )

    trainer = None
    last_err: Optional[BaseException] = None
    for kwargs in (dict(processing_class=tokenizer), dict(tokenizer=tokenizer)):
        try:
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=ds,
                formatting_func=formatting_func,
                **kwargs,
            )
            break
        except TypeError as e:
            last_err = e
            continue
    if trainer is None:
        raise RuntimeError(f"SFTTrainer could not be constructed: {last_err}")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[imagination2 train] Saved adapter to {output_dir}", flush=True)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
