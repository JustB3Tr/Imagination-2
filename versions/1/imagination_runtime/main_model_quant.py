"""
How the main causal LM is loaded: bitsandbytes 4-bit (NF4) vs bfloat16 (no 4-bit).

Colab: set a single variable before starting the app::

    import os
    os.environ["IMAGINATION_MAIN_4BIT"] = "1"   # 4-bit (saves VRAM, slower first-token on cold start)
    os.environ["IMAGINATION_MAIN_4BIT"] = "0"   # bfloat16 full main weights (needs more VRAM, often faster)

If ``IMAGINATION_MAIN_4BIT`` is **unset**, legacy env ``IMAGINATION_MAIN_LOAD_BF16=1`` means bfloat16;
otherwise on CUDA the default is 4-bit. CLIP + projector are always loaded in fp16/bf16 (not 4-bit).

See ``print_main_lm_load_banner`` for a one-line console summary.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch

from transformers import BitsAndBytesConfig

BNB_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def _truthy(raw: str) -> bool:
    s = (raw or "").strip().lower()
    return s in ("1", "true", "yes", "on")


def _falsy(raw: str) -> bool:
    s = (raw or "").strip().lower()
    return s in ("0", "false", "no", "off")


def _legacy_main_load_bf16() -> bool:
    """``IMAGINATION_MAIN_LOAD_BF16=1`` → bfloat16 main LM (old name, still supported)."""
    return _truthy(os.getenv("IMAGINATION_MAIN_LOAD_BF16") or "")


def main_lm_use_4bit() -> bool:
    """
    Whether to load the *main* causal LM with bitsandbytes 4-bit (NF4).

    Priority:
    1. **IMAGINATION_MAIN_4BIT** (recommended): true/1/yes = 4-bit; false/0 = no 4-bit (bfloat16 on GPU).
    2. If unset: **IMAGINATION_MAIN_LOAD_BF16** = 1 → not 4-bit; else on CUDA use 4-bit.
    3. No CUDA → never 4-bit.
    """
    ex = (os.getenv("IMAGINATION_MAIN_4BIT") or "").strip()
    if ex:
        if _truthy(ex):
            return bool(torch.cuda.is_available())
        if _falsy(ex):
            return False
    if _legacy_main_load_bf16():
        return False
    return bool(torch.cuda.is_available())


def main_lm_use_bf16_on_gpu() -> bool:
    """bfloat16 main LM weights (no quant), only meaningful when not using 4-bit and CUDA is used."""
    if main_lm_use_4bit():
        return False
    ex = (os.getenv("IMAGINATION_MAIN_4BIT") or "").strip()
    if ex and _falsy(ex) and torch.cuda.is_available():
        return True
    if _legacy_main_load_bf16() and torch.cuda.is_available():
        return True
    return False


def main_lm_load_mode_label() -> str:
    """
    Human-readable label for the main LM: ``"4bit"``, ``"bf16"``, or ``"auto"`` (e.g. CPU or dtype auto).
    """
    if main_lm_use_4bit():
        return "4bit"
    if main_lm_use_bf16_on_gpu():
        return "bf16"
    return "auto"


def main_lm_from_pretrained_kwargs() -> Dict[str, Any]:
    """
    Keyword args to pass to ``AutoModelForCausalLM.from_pretrained`` for the **main** LM
    (CLIP path and text-only path).
    """
    k: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if main_lm_use_4bit():
        k["torch_dtype"] = "auto"
        k["quantization_config"] = BNB_4BIT
    elif main_lm_use_bf16_on_gpu():
        k["torch_dtype"] = torch.bfloat16
    else:
        k["torch_dtype"] = "auto"
    return k


def print_main_lm_load_banner(path: str, *, clip_projector: bool = False) -> None:
    """
    Print a single [imagination] line describing how the main LM will be loaded, then load proceeds.

    ``path`` is the model directory (shortened in the log if very long).
    """
    p = (path or "").strip()
    if len(p) > 72:
        p = f"...{p[-64:]}"

    on_gpu = bool(torch.cuda.is_available())
    mode = main_lm_load_mode_label()
    if mode == "4bit" and on_gpu:
        line = (
            f"[imagination] Main LM: bitsandbytes 4-bit (NF4, compute fp16) — {p}"
        )
    elif mode == "bf16" and on_gpu:
        line = f"[imagination] Main LM: bfloat16 (no 4-bit quant) — {p}"
    else:
        if not on_gpu:
            line = f"[imagination] Main LM: torch.float32 / auto (CPU, no 4-bit) — {p}"
        else:
            line = f"[imagination] Main LM: torch_dtype=auto (no 4-bit / no bf16 override) — {p}"

    if clip_projector:
        line = (
            line
            + "\n[imagination] Vision: CLIP tower + MLP projector (float; not 4-bit); loading CLIP+projector+LM…"
        )
    else:
        line = line + "\n[imagination] Loading main LM…"
    print(line, flush=True)


def resolve_quant_env_summary() -> Tuple[str, str]:
    """
    (effective_main_4bit, note) for docs / one-line env dump.
    ``IMAGINATION_MAIN_4BIT`` if set takes precedence over ``IMAGINATION_MAIN_LOAD_BF16``.
    """
    ex = (os.getenv("IMAGINATION_MAIN_4BIT") or "").strip()
    if ex:
        s = f"IMAGINATION_MAIN_4BIT={ex!r}"
    else:
        s = f"IMAGINATION_MAIN_4BIT unset, IMAGINATION_MAIN_LOAD_BF16={(os.getenv('IMAGINATION_MAIN_LOAD_BF16') or '')!r}"
    eff = "4-bit main LM" if main_lm_use_4bit() else ("bf16 main LM" if main_lm_use_bf16_on_gpu() else "auto/CPU")
    return eff, s
