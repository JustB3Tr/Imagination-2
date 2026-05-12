"""
Multimodal main-model load + generation for Imagination v1.3 (HF-style VLM).

Falls back to text-only causal LM when the checkpoint is not a vision model or
IMAGINATION_TEXT_MAIN_ONLY=1 is set.
"""
from __future__ import annotations

import json
import os
from threading import Lock, Thread
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoTokenizer, TextIteratorStreamer, AutoProcessor
from transformers import AutoModelForCausalLM

from imagination_runtime.main_model_quant import main_lm_from_pretrained_kwargs, print_main_lm_load_banner
from imagination_runtime.paths import resolve_vision_projector_bundle_dir


def _read_config_model_type(path: str) -> str:
    p = os.path.join(path, "config.json")
    if not os.path.isfile(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        return str(j.get("model_type") or "").lower()
    except (OSError, json.JSONDecodeError):
        return ""


def checkpoint_looks_like_vlm(path: str) -> bool:
    mt = _read_config_model_type(path)
    if not mt:
        return False
    if "llava" in mt or "qwen2_vl" in mt or "qwen3_vl" in mt or "mllama" in mt:
        return True
    try:
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
        if j.get("vision_config") or j.get("image_token_index") is not None:
            return True
    except (OSError, json.JSONDecodeError):
        pass
    return False


def load_main_model_auto(path: str) -> tuple[Any, Any, Any, bool]:
    """
    Returns (tokenizer_or_none, model, processor_or_none, is_vlm).
    For text-only: processor is None, tokenizer is AutoTokenizer, is_vlm False.
    For VLM: processor is AutoProcessor; tokenizer is processor.tokenizer if present else AutoTokenizer.
    """
    text_only = (os.getenv("IMAGINATION_TEXT_MAIN_ONLY") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    bundle_dir = resolve_vision_projector_bundle_dir(path)
    if bundle_dir and not text_only:
        from imagination_runtime.clip_projector_vlm import load_clip_projector_bundle

        print_main_lm_load_banner(path, clip_projector=True)
        print(f"[imagination] Projector bundle directory: {bundle_dir}", flush=True)
        tok, mdl, shim = load_clip_projector_bundle(path, bundle_dir)
        return tok, mdl, shim, True

    if text_only or not checkpoint_looks_like_vlm(path):
        print_main_lm_load_banner(path, clip_projector=False)
        tok = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
        kwargs: Dict[str, Any] = main_lm_from_pretrained_kwargs()
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        model.eval()
        return tok, model, None, False

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tok = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        path, use_fast=True, trust_remote_code=True
    )
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token

    print_main_lm_load_banner(path, clip_projector=False)
    kwargs = main_lm_from_pretrained_kwargs()

    model = None
    mt = _read_config_model_type(path)
    try:
        if "llava" in mt:
            from transformers import LlavaForConditionalGeneration

            model = LlavaForConditionalGeneration.from_pretrained(path, **kwargs)
        else:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(path, **kwargs)
    except Exception:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(path, **kwargs)

    model.eval()
    return tok, model, processor, True


def _vlm_repetition_penalty() -> float:
    raw = (os.getenv("IMAGINATION_VLM_REPETITION_PENALTY") or "").strip()
    if not raw:
        return 1.15
    try:
        return max(1.0, min(float(raw), 2.0))
    except ValueError:
        return 1.15


def _vlm_no_repeat_ngram() -> int:
    raw = (os.getenv("IMAGINATION_VLM_NO_REPEAT_NGRAM") or "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        return 0
    if not raw:
        return 4
    try:
        return max(0, min(int(raw), 16))
    except ValueError:
        return 4


def _vlm_stream_timeout_s() -> float:
    """
    Timeout while waiting for streamed decode chunks.
    Keeps vision requests from hanging silently when generation stalls before first token.

    Default 300 -> 600s (10m); bf16 + long prefill can exceed 5m to first chunk. Max env 1800s.
    """
    raw = (os.getenv("IMAGINATION_VLM_STREAM_TIMEOUT_S") or "").strip()
    if not raw:
        return 600.0
    try:
        return max(10.0, min(float(raw), 1800.0))
    except ValueError:
        return 600.0


def _get_device_for_model(model_obj: Any) -> torch.device:
    if hasattr(model_obj, "hf_device_map") and getattr(model_obj, "hf_device_map", None):
        for _, dev in model_obj.hf_device_map.items():
            if isinstance(dev, str) and dev not in ("cpu", "disk"):
                return torch.device(dev)
    return next(model_obj.parameters()).device


def _messages_to_vlm_chat(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize to HF chat template: content is str or list of {type,text/image}."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if isinstance(content, list):
            out.append({"role": role, "content": content})
        else:
            out.append({"role": role, "content": (content or "").strip()})
    return out


def generate_stream_vlm(
    *,
    processor: Any,
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    lock: Lock,
    image: Optional[Any] = None,
) -> Iterable[str]:
    """Stream decoded text from a VLM; `image` is a PIL Image or None."""
    if getattr(processor, "is_clip_projector", False):
        from imagination_runtime.clip_projector_vlm import generate_stream_clip_projector

        yield from generate_stream_clip_projector(
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            lock=lock,
            image=image,
        )
        return

    device = _get_device_for_model(model)
    chat_msgs = _messages_to_vlm_chat(messages)

    try:
        prompt_text = processor.apply_chat_template(
            chat_msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        # Fallback: single string prompt
        parts = []
        for m in chat_msgs:
            parts.append(f"{m['role']}: {m.get('content', '')}")
        prompt_text = "\n".join(parts) + "\nassistant:"

    if image is not None:
        try:
            inputs = processor(images=[image], text=prompt_text, return_tensors="pt")
        except Exception:
            inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    else:
        inputs = processor(text=prompt_text, return_tensors="pt")

    def _to_dev(x: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(device)
        return x

    model_inputs = {k: _to_dev(v) for k, v in inputs.items() if v is not None}
    prompt_len = int(model_inputs["input_ids"].shape[1])
    max_pos = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "text_config", {}
    )
    if isinstance(max_pos, dict):
        max_pos = max_pos.get("max_position_embeddings", 8192)
    try:
        max_pos = int(max_pos)
    except (TypeError, ValueError):
        max_pos = 8192
    room = max(64, max_pos - prompt_len - 32)
    clamped_max = max(32, min(int(max_new_tokens), room))

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=_vlm_stream_timeout_s(),
    )
    gen_error: Dict[str, Any] = {"exc": None}
    rp = _vlm_repetition_penalty()
    ngram = _vlm_no_repeat_ngram()
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": clamped_max,
        "do_sample": False,
        "use_cache": True,
        "repetition_penalty": rp,
    }
    if ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = ngram
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    def _run() -> None:
        try:
            with lock, torch.inference_mode():
                model.generate(**model_inputs, streamer=streamer, **gen_kwargs)
        except TypeError:
            # Some vision models reject no_repeat_ngram_size on generate().
            gen_kwargs.pop("no_repeat_ngram_size", None)
            try:
                with lock, torch.inference_mode():
                    model.generate(**model_inputs, streamer=streamer, **gen_kwargs)
            except Exception as e:
                gen_error["exc"] = e
        except Exception as e:
            gen_error["exc"] = e

    t = Thread(target=_run, daemon=True)
    t.start()
    partial = ""
    for chunk in streamer:
        if gen_error["exc"] is not None:
            raise RuntimeError(f"VLM generation failed: {gen_error['exc']}") from gen_error["exc"]
        partial += chunk
        yield partial
    t.join(timeout=0.2)
    if gen_error["exc"] is not None:
        raise RuntimeError(f"VLM generation failed: {gen_error['exc']}") from gen_error["exc"]
