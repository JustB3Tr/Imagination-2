from __future__ import annotations

import re
from typing import Any, Iterable, Optional

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

THINK_FORMAT_INSTRUCTION = (
    "You must format your internal safety considerations inside strict "
    "<think>...</think> tags before providing your final response."
)


def _iter_resize_candidates(model: Any) -> Iterable[Any]:
    yield model
    for attr in ("llm", "model", "base_model", "language_model"):
        cand = getattr(model, attr, None)
        if cand is not None:
            yield cand


def _try_resize_token_embeddings(model: Any, target_size: int) -> bool:
    for cand in _iter_resize_candidates(model):
        resize = getattr(cand, "resize_token_embeddings", None)
        if callable(resize):
            resize(int(target_size))
            return True
    return False


def add_think_structure_tokens(
    tokenizer: Any,
    model: Optional[Any],
    *,
    label: str,
) -> int:
    """
    Ensure ``<think>`` and ``</think>`` exist in tokenizer vocab.
    Resize token embeddings when vocab grows.
    """
    if tokenizer is None:
        return 0
    added = int(tokenizer.add_tokens([THINK_OPEN_TAG, THINK_CLOSE_TAG]))
    if added <= 0:
        return 0
    if model is None:
        print(
            f"[imagination] Added {added} think tokens to {label} tokenizer; model unavailable for resize.",
            flush=True,
        )
        return added
    try:
        resized = _try_resize_token_embeddings(model, len(tokenizer))
    except Exception as exc:
        print(
            f"[imagination] Added think tokens for {label}, but resize_token_embeddings failed: {exc}",
            flush=True,
        )
        return added
    if resized:
        print(
            f"[imagination] Added {added} think tokens and resized {label} embeddings to {len(tokenizer)}.",
            flush=True,
        )
    else:
        print(
            f"[imagination] Added {added} think tokens for {label}; no resize_token_embeddings method found.",
            flush=True,
        )
    return added


def preserve_think_tags(text: str) -> str:
    """
    Normalize common escaped variants so think tags survive transport/rendering.
    """
    out = text or ""
    out = re.sub(r"(?i)&lt;\s*think\s*&gt;", THINK_OPEN_TAG, out)
    out = re.sub(r"(?i)&lt;\s*/\s*think\s*&gt;", THINK_CLOSE_TAG, out)
    return out
