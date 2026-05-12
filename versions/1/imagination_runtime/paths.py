from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def resolve_root_path(root_path: str | None) -> str:
    """
    Resolve the mounted imagination root path for local or Colab.
    Priority:
      1) explicit function argument
      2) IMAGINATION_ROOT env var
      3) default Colab mount path
    """
    p = (root_path or "").strip()
    if not p:
        p = (os.getenv("IMAGINATION_ROOT") or "").strip()
    if not p:
        p = "/content/imagination-v1.1.0"
    return p.rstrip("/\\")


def _checkout_repo_root_from_imagination_runtime() -> str | None:
    """
    Repo root containing ``versions/v1.x/imagination_runtime/`` (this package; e.g. v1.4).

    Used so ``<repo>/vision_cpt_out_real`` resolves on local Windows/Drive checkouts when
    ``IMAGINATION_ROOT`` is unset and would otherwise default to the Colab path.
    """
    try:
        here = Path(__file__).resolve().parent
        if here.name != "imagination_runtime":
            return None
        root = here.parents[2]
        if not root.is_dir():
            return None
        return str(root)
    except IndexError:
        return None


def _dedupe_paths(*paths: str) -> list[str]:
    out: list[str] = []
    for raw in paths:
        p = (raw or "").strip()
        if not p:
            continue
        n = os.path.normpath(p)
        if n not in out:
            out.append(n)
    return out


def _vision_projector_bundle_at(dir_path: str) -> str | None:
    """Return dir_path if it contains projector.pt + attach_vision_multimodal_meta.json."""
    proj = os.path.join(dir_path, "projector.pt")
    meta = os.path.join(dir_path, "attach_vision_multimodal_meta.json")
    if os.path.isfile(proj) and os.path.isfile(meta):
        return dir_path
    return None


def resolve_vision_projector_bundle_dir(llm_root: str) -> str | None:
    """
    Directory containing projector.pt + attach_vision_multimodal_meta.json (CLIP + projector CPT).

    Search order (first hit wins):

    1. ``IMAGINATION_VISION_PROJECTOR_DIR`` if set: absolute path as-is; otherwise each of
       ``<IMAGINATION_ROOT>/<dir>`` and ``<llm_root>/<dir>`` (so bundles can live next to weights
       or under the repo root, matching the doc).
    2. Default: ``<llm_root>/vision_cpt_out_real`` (bundle colocated with the HF model folder).
    3. Default fallback: ``<IMAGINATION_ROOT>/vision_cpt_out_real`` (env or explicit root).
    4. Default fallback: checkout root inferred from this file (``.../imagination-v1.1.0``) so
       ``G:\\...\\imagination-v1.1.0\\vision_cpt_out_real`` works without setting ``IMAGINATION_ROOT``.

    Set IMAGINATION_USE_CLIP_PROJECTOR=0 to disable and use a native HF VLM folder instead.
    """
    if (os.getenv("IMAGINATION_USE_CLIP_PROJECTOR") or "").strip().lower() in ("0", "false", "no"):
        return None
    repo_root = resolve_root_path(None)
    checkout_root = _checkout_repo_root_from_imagination_runtime()
    llm = os.path.normpath((llm_root or "").strip() or repo_root)

    override = (os.getenv("IMAGINATION_VISION_PROJECTOR_DIR") or "").strip()
    if override:
        if os.path.isabs(override):
            return _vision_projector_bundle_at(os.path.normpath(override))
        for base in _dedupe_paths(repo_root, llm, checkout_root or ""):
            hit = _vision_projector_bundle_at(os.path.normpath(os.path.join(base, override)))
            if hit:
                return hit
        return None

    for base in _dedupe_paths(llm, repo_root, checkout_root or ""):
        hit = _vision_projector_bundle_at(os.path.join(base, "vision_cpt_out_real"))
        if hit:
            return hit
    return None


def resolve_cad_coder_dir(root: str) -> str:
    """
    Directory for the coder module weights.
    Set IMAGINATION_CAD_CODER_PATH to an absolute path, or a path relative to IMAGINATION_ROOT,
    if your folder name differs from the default (e.g. after swapping 3B → 7B weights).
    """
    override = (os.getenv("IMAGINATION_CAD_CODER_PATH") or "").strip()
    if override:
        p = override
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        return os.path.normpath(p)
    return os.path.join(root, "modules", "cad", "qwen finetuned coder(3b)")


@dataclass(frozen=True)
class ModelPaths:
    root: str

    @property
    def main_llm(self) -> str:
        return self.root

    @property
    def main_vlm(self) -> str:
        """HF folder for 8B VLM; override with IMAGINATION_MAIN_VLM_PATH (abs or relative to root)."""
        override = (os.getenv("IMAGINATION_MAIN_VLM_PATH") or "").strip()
        if override:
            p = override
            if not os.path.isabs(p):
                p = os.path.join(self.root, p)
            return os.path.normpath(p)
        return self.root

    @property
    def cad_coder(self) -> str:
        return resolve_cad_coder_dir(self.root)

    @property
    def reasoning_llm(self) -> str:
        return os.path.join(self.root, "modules", "reasoning", "7b qwen reasoning finetuned")

    @property
    def embeddings(self) -> str:
        return os.path.join(self.root, "modules", "research", "embeddings", "bge-small-en-v1.5")

    @property
    def reranker(self) -> str:
        return os.path.join(self.root, "modules", "research", "reranker", "bge-reranker-v2-m3")

    @property
    def tiny_sd(self) -> str:
        return os.path.join(self.root, "modules", "image", "tiny image dev model")
