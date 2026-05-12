"""
CLIP vision tower (e.g. openai/clip-vit-large-patch14-336) + trained projector + causal LM.

Loads `projector.pt` + `attach_vision_multimodal_meta.json` from a bundle directory
(produced by pretrain-vision-8b/scripts/attach_vision_multimodal.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    TextIteratorStreamer,
)
from transformers.activations import ACT2FN

from imagination_runtime.main_model_quant import main_lm_from_pretrained_kwargs


def _vlm_stream_timeout_s() -> float:
    """Timeout while waiting for streamed decode chunks.

    Covers the wait between ``TextIteratorStreamer`` chunks; on Colab the first chunk can take
    30-90s extra the *first* time ``model.generate`` runs (lazy BnB 4-bit kernel compile, prefill
    on a ~600-token image-expanded prompt). Default 300 -> 600s: bf16 main LM + first ``generate``
    on Colab can exceed 5 minutes to the first streamed token; 5m was often a silent streamer timeout.
    """
    raw = (os.getenv("IMAGINATION_VLM_STREAM_TIMEOUT_S") or "").strip()
    if not raw:
        return 600.0
    try:
        return max(10.0, min(float(raw), 1800.0))
    except ValueError:
        return 600.0


class MultimodalProjector(nn.Module):
    """LLaVA-style 2-layer MLP: vision_dim -> llm_dim -> llm_dim."""

    def __init__(
        self,
        vision_hidden: int,
        text_hidden: int,
        hidden_act: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden, text_hidden, bias=bias)
        self.act = ACT2FN[hidden_act]
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


def vision_patch_embeddings(
    vision: CLIPVisionModel,
    pixel_values: torch.Tensor,
    feature_layer: int,
    patch_count: int,
) -> torch.Tensor:
    out = vision(
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    idx = feature_layer if feature_layer >= 0 else len(hs) + feature_layer
    hidden = hs[idx]
    if hidden.shape[1] == patch_count + 1:
        hidden = hidden[:, 1:, :]
    elif hidden.shape[1] != patch_count:
        raise ValueError(
            f"Unexpected vision seq len {hidden.shape[1]} for patch_count={patch_count}"
        )
    return hidden


def _rescale_projection_rms_to_token_table(
    projected: torch.Tensor,
    embed_layer: Any,
    *,
    max_tokens_sample: int = 4096,
) -> torch.Tensor:
    """
    Scale projector outputs so their per-token L2 norm matches the average norm of (sampled) rows
    of the LM's token embedding table. Use when the checkpoint was trained with a different
    activation scale than your current vision dtype (FP16 noise) or a mismatched final layer.
    Enable with IMAGINATION_VISION_MATCH_EMBED_RMS=1.
    """
    w = embed_layer.weight.float()
    n = min(int(max_tokens_sample), int(w.shape[0]))
    if n <= 0:
        return projected
    idx = torch.arange(n, device=w.device, dtype=torch.long)
    ref = w[idx].pow(2).sum(-1).sqrt().mean().clamp(min=1e-8)
    p = projected.float()
    cur = p.pow(2).sum(-1).sqrt().mean().clamp(min=1e-8)
    scale = (ref / cur).clamp(0.05, 25.0)
    return (p * scale).to(projected.dtype)


def merge_image_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    projected: torch.Tensor,
) -> torch.Tensor:
    out = inputs_embeds.clone()
    b, _t, _h = out.shape
    for i in range(b):
        mask = input_ids[i] == image_token_id
        n_img = int(mask.sum().item())
        if n_img != projected.shape[1]:
            raise ValueError(
                f"Batch {i}: {n_img} image tokens but projector seq={projected.shape[1]}"
            )
        out[i, mask] = projected[i].to(dtype=out.dtype)
    return out


def expand_image_token_ids(ids: List[int], image_token_id: int, patch_count: int) -> List[int]:
    out: List[int] = []
    for tid in ids:
        if tid == image_token_id:
            out.extend([image_token_id] * patch_count)
        else:
            out.append(tid)
    return out


def _flatten_message_content(msg: Dict[str, Any], image_token: str) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return (c or "").strip()
    if not isinstance(c, list):
        return ""
    parts: List[str] = []
    for block in c:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "image":
            parts.append(image_token)
        elif t == "text":
            parts.append((block.get("text") or "").strip())
    return "\n\n".join(p for p in parts if p)


def resolve_image_placeholder_string(tokenizer: Any, image_token: str, image_token_id: int) -> str:
    """
    Find a string that encodes to exactly [image_token_id].

    Plain '<image>' often splits into multiple BPE tokens unless it was added as a special
    token during training (--add_image_token). We try added_tokens_encoder, meta string,
    convert_ids_to_tokens, and decode([id]).
    """
    candidates: List[str] = []
    enc = getattr(tokenizer, "added_tokens_encoder", None) or {}
    for s, tid in enc.items():
        if int(tid) == int(image_token_id):
            candidates.append(str(s))
    if image_token:
        candidates.append(image_token.strip())
    try:
        t0 = tokenizer.convert_ids_to_tokens([image_token_id])
        if t0 and t0[0] is not None:
            piece = t0[0]
            if not isinstance(piece, str):
                piece = str(piece)
            if piece:
                candidates.append(piece)
    except Exception:
        pass
    try:
        dec = tokenizer.decode([image_token_id], skip_special_tokens=False)
        if dec:
            candidates.append(dec)
    except Exception:
        pass

    seen: set[str] = set()
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1 and ids[0] == image_token_id:
            return cand

    got = tokenizer.encode(image_token, add_special_tokens=False)
    raise ValueError(
        f"No string encodes to a single id {image_token_id} (needed for the projector bundle). "
        f"Encoding meta image_token {image_token!r} gives {got!r} (likely split into multiple BPE tokens). "
        "Fix: deploy the tokenizer files from the same checkpoint where you ran --add_image_token "
        "(tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json). "
        f"Vocab size is {len(tokenizer)}; ensure id {image_token_id} is your added <image> token."
    )


def messages_to_string_contents(messages: List[Dict[str, Any]], image_token: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages or []:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        out.append({"role": str(role), "content": _flatten_message_content(m, image_token)})
    return out


def _get_device_for_model(model_obj: Any) -> torch.device:
    if hasattr(model_obj, "hf_device_map") and getattr(model_obj, "hf_device_map", None):
        for _, dev in model_obj.hf_device_map.items():
            if isinstance(dev, str) and dev not in ("cpu", "disk"):
                return torch.device(dev)
    return next(model_obj.parameters()).device


@dataclass
class ClipProjectorBundle:
    vision: CLIPVisionModel
    projector: MultimodalProjector
    image_processor: CLIPImageProcessor
    meta: Dict[str, Any]


class ClipProjectorProcessorShim:
    """Hung on RUNTIME.main_processor so generate_stream_vlm can branch."""

    is_clip_projector = True

    def __init__(self, tokenizer: Any, bundle: ClipProjectorBundle) -> None:
        self.tokenizer = tokenizer
        self.bundle = bundle


def load_clip_projector_bundle(llm_path: str, bundle_dir: str) -> tuple[Any, Any, ClipProjectorProcessorShim]:
    meta_path = os.path.join(bundle_dir, "attach_vision_multimodal_meta.json")
    proj_path = os.path.join(bundle_dir, "projector.pt")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vision_model_id = (meta.get("vision_model") or "openai/clip-vit-large-patch14-336").strip()
    image_token = (meta.get("image_token") or "<image>").strip()
    image_token_id = int(meta["image_token_id"])
    patch_count = int(meta["patch_count"])
    vision_feature_layer = int(meta.get("vision_feature_layer", -2))
    llm_hidden = int(meta["llm_hidden"])
    vision_hidden = int(meta["vision_hidden"])
    projector_act = (meta.get("projector_act") or "gelu").strip().lower()

    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training adds one special token at id == old vocab size (e.g. 128256 when base had 128256 ids 0..128255).
    # Stale tokenizer files often omit it, so "<image>" splits into multiple BPE ids.
    patched_image_vocab = False
    if image_token_id > len(tokenizer):
        raise ValueError(
            f"Bundle image_token_id={image_token_id} is past tokenizer length={len(tokenizer)} "
            "(cannot infer how many special tokens to add). Use tokenizer files from projector training."
        )
    if image_token_id == len(tokenizer):
        print(
            f"[imagination] Adding special token {image_token!r} as id {image_token_id} "
            f"(tokenizer had len={len(tokenizer)}; matches projector training). "
            "Save tokenizer after first run if you want to skip this step.",
            flush=True,
        )
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        patched_image_vocab = True

    llm_kwargs: Dict[str, Any] = main_lm_from_pretrained_kwargs()
    print(
        "[imagination] CLIP+projector: loading main causal LM weights (can take several minutes on Colab)…",
        flush=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(llm_path, **llm_kwargs)
    print("[imagination] CLIP+projector: main LM weights loaded.", flush=True)
    llm.eval()

    emb_n = int(llm.get_input_embeddings().weight.shape[0])
    if len(tokenizer) > emb_n:
        try:
            llm.resize_token_embeddings(len(tokenizer))
            print(
                f"[imagination] Resized LM embeddings {emb_n} -> {len(tokenizer)} for <image> token.",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Tokenizer len={len(tokenizer)} but model embeddings={emb_n}; resize failed ({e}). "
                "Try full-precision load or update transformers; 4-bit resize can be finicky."
            ) from e

    image_token_literal = resolve_image_placeholder_string(tokenizer, image_token, image_token_id)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(
        f"[imagination] CLIP+projector: loading CLIP vision tower {vision_model_id!r}…",
        flush=True,
    )
    vision = CLIPVisionModel.from_pretrained(
        vision_model_id,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
    )
    vision.eval()
    vision.requires_grad_(False)
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)
    print("[imagination] CLIP+projector: CLIP + image processor ready.", flush=True)

    projector = MultimodalProjector(
        vision_hidden=vision_hidden,
        text_hidden=llm_hidden,
        hidden_act=projector_act,
        bias=True,
    )
    print("[imagination] CLIP+projector: loading trained projector MLP (projector.pt)…", flush=True)
    try:
        state = torch.load(proj_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(proj_path, map_location="cpu")
    projector.load_state_dict(state, strict=True)
    projector.eval()
    projector.requires_grad_(False)
    print("[imagination] CLIP+projector: bundle ready (vision + projector + LLM).", flush=True)

    device = _get_device_for_model(llm)
    if device.type == "cuda":
        vision = vision.to(device=device, dtype=dtype)
        projector = projector.to(device=device, dtype=dtype)
    else:
        vision = vision.to(device=device)
        projector = projector.to(device=device)

    bundle = ClipProjectorBundle(
        vision=vision,
        projector=projector,
        image_processor=image_processor,
        meta={
            "image_token": image_token,
            "image_token_literal": image_token_literal,
            "image_token_id": image_token_id,
            "patch_count": patch_count,
            "vision_feature_layer": vision_feature_layer,
            "vision_model": vision_model_id,
        },
    )
    return tokenizer, llm, ClipProjectorProcessorShim(tokenizer, bundle)


def generate_stream_clip_projector(
    *,
    processor: ClipProjectorProcessorShim,
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    lock: Lock,
    image: Optional[Any] = None,
) -> Iterable[str]:
    device = _get_device_for_model(model)
    meta = processor.bundle.meta
    image_token = str(meta.get("image_token_literal") or meta["image_token"])
    image_token_id = int(meta["image_token_id"])
    patch_count = int(meta["patch_count"])
    vision_feature_layer = int(meta["vision_feature_layer"])

    string_msgs = messages_to_string_contents(messages, image_token)
    try:
        prompt_text = tokenizer.apply_chat_template(
            string_msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        parts: List[str] = []
        for m in string_msgs:
            parts.append(f"{m['role']}: {m.get('content', '')}")
        prompt_text = "\n".join(parts) + "\nassistant:"

    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if image is not None:
        if ids.count(image_token_id) != 1:
            raise ValueError(
                f"Expected exactly one image token id {image_token_id} in prompt when an image is set; "
                f"got count={ids.count(image_token_id)}. Ensure the chat template preserves {image_token!r}."
            )
        ids = expand_image_token_ids(ids, image_token_id, patch_count)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=_vlm_stream_timeout_s(),
    )
    gen_error: Dict[str, Any] = {"exc": None}

    max_pos = getattr(model.config, "max_position_embeddings", None) or 8192
    try:
        max_pos = int(max_pos)
    except (TypeError, ValueError):
        max_pos = 8192
    prompt_len = int(input_ids.shape[1])
    room = max(64, max_pos - prompt_len - 32)
    clamped_max = max(32, min(int(max_new_tokens), room))

    try:
        rp = max(1.0, min(float((os.getenv("IMAGINATION_VLM_REPETITION_PENALTY") or "1.15").strip()), 2.0))
    except ValueError:
        rp = 1.15
    ngram_raw = (os.getenv("IMAGINATION_VLM_NO_REPEAT_NGRAM") or "4").strip().lower()
    if ngram_raw in ("0", "false", "no", "off"):
        ngram = 0
    else:
        try:
            ngram = max(0, min(int(ngram_raw), 16))
        except ValueError:
            ngram = 4

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": clamped_max,
        "do_sample": False,
        "use_cache": True,
        "attention_mask": attention_mask,
        "repetition_penalty": rp,
    }
    if ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = ngram
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    def _run() -> None:
        def _gen(gkw: Dict[str, Any]) -> None:
            with lock, torch.inference_mode():
                if image is not None:
                    proc = processor.bundle.image_processor(images=image, return_tensors="pt")
                    vision = processor.bundle.vision
                    vision_dtype = next(vision.parameters()).dtype
                    pixel_values = proc["pixel_values"].to(device=device, dtype=vision_dtype)
                    img_hidden = vision_patch_embeddings(
                        vision,
                        pixel_values,
                        vision_feature_layer,
                        patch_count,
                    )
                    embed_layer = model.get_input_embeddings()
                    embed_dtype = embed_layer.weight.dtype
                    # FP32 matmul through the projector avoids HF16 blow-ups that look like "the the the…" loops
                    # in the LM even when text-only decoding is fine.
                    img_h32 = img_hidden.detach().float()
                    with torch.autocast(device_type=device.type, enabled=False):
                        proj = processor.bundle.projector(img_h32).to(dtype=embed_dtype)
                    if (os.getenv("IMAGINATION_VISION_MATCH_EMBED_RMS") or "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    ):
                        proj = _rescale_projection_rms_to_token_table(proj, embed_layer)
                    if (os.getenv("IMAGINATION_DEBUG_VISION") or "").strip() == "1":
                        prms = proj.detach().float().pow(2).mean().sqrt().item()
                        print(
                            f"[imagination][vision] proj shape={tuple(proj.shape)} dtype={proj.dtype} "
                            f"rms={prms:.5g} finite={bool(torch.isfinite(proj).all().item())}",
                            flush=True,
                        )
                    if not torch.isfinite(proj).all():
                        raise RuntimeError(
                            "Vision projector produced NaN/Inf. Check CLIP preprocessing, "
                            "vision_model in attach_vision_multimodal_meta.json, and projector.pt."
                        )
                    inputs_embeds = embed_layer(input_ids)
                    inputs_embeds = merge_image_embeddings(
                        inputs_embeds, input_ids, image_token_id, proj
                    )
                    if not torch.isfinite(inputs_embeds).all():
                        raise RuntimeError("Merged vision+text embeddings are NaN/Inf after image injection.")
                    model.generate(inputs_embeds=inputs_embeds, streamer=streamer, **gkw)
                else:
                    model.generate(input_ids=input_ids, streamer=streamer, **gkw)

        try:
            _gen(gen_kwargs)
        except TypeError:
            gen_kwargs.pop("no_repeat_ngram_size", None)
            try:
                _gen(gen_kwargs)
            except Exception as e:
                gen_error["exc"] = e
        except Exception as e:
            gen_error["exc"] = e

    t = Thread(target=_run, daemon=True)
    t.start()
    partial = ""
    for chunk in streamer:
        if gen_error["exc"] is not None:
            raise RuntimeError(f"CLIP+projector generation failed: {gen_error['exc']}") from gen_error["exc"]
        partial += chunk
        yield partial
    t.join(timeout=0.2)
    if gen_error["exc"] is not None:
        raise RuntimeError(f"CLIP+projector generation failed: {gen_error['exc']}") from gen_error["exc"]
