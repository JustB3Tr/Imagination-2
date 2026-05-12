#!/usr/bin/env python3
# ruff: noqa: E501
"""
Download high-quality coding instruction data from Hugging Face (no RAG at inference),
filter for fact-dense rows (numbers, API-like tokens, code fences, troubleshooting cues),
and write ``train.jsonl`` in chat format::

  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Default sources (all public, instruction-style, multi-language where applicable):
  - ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response
  - bigcode/self-oss-instruct-sc2-exec-filter-50k
  - ise-uiuc/Magicoder_evol_instruct_110k
  - sahil2801/CodeAlpaca-20k

Optional: ``--docs-glob`` / ``--docs-dir`` to fold in local Markdown/text as memorization-style pairs.

Usage::

  pip install -r scripts/requirements_train_export.txt
  python scripts/hf_build_train_jsonl.py --output train.jsonl --target 1000

HF_TOKEN in the environment is only needed for gated datasets (defaults are open).

**Note (Hugging Face cache):** the first run may log ``Downloading ... README.md`` (and similar) for each
dataset *card* — that is metadata going into the HF cache, **not** your training JSONL. Training rows
come from streaming dataset *splits* (e.g. ``train``) into ``--output``.

**Local README / docs in JSONL:** rows that look like "documentation excerpt" come from
``--docs-dir`` / ``--docs-glob`` + ``--doc-quota``. The v1.4 ``download_datasets.py`` wrapper passes
``--doc-quota 0`` by default so you only get HF coding instruction pairs unless you add e.g.
``-- --docs-dir ./docs --doc-quota -1`` after the wrapper flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import glob as globmod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Heuristics: prefer rows that look "searchable" / factual / API-ish
# ---------------------------------------------------------------------------
_DIGIT_RE = re.compile(r"\d")
_CODEISH_RE = re.compile(
    r"(`[^`]+`|\[[\w.-]+\]|::|->|=>|\bdef\b|\bfn\b|\bfunc\b|\bimport\b|\bSELECT\b|\bFROM\b|"
    r"\bHTTP/\d|\b\d{3}\b\s+(OK|Error)|\b0x[0-9a-fA-F]+\b|\b\d+\.\d+\.\d+\b)"
)
_TROUBLE_RE = re.compile(
    r"\b(error|exception|traceback|fix|debug|timeout|segfault|panic|leak|"
    r"deadlock|race|OOM|stack overflow|undefined|NaN|null pointer|"
    r"compiler|linker|syntax|deprecat|migrate|workaround|CVE-\d+)\b",
    re.I,
)


def fact_density_score(user: str, assistant: str) -> float:
    text = f"{user}\n{assistant}"
    score = 0.0
    score += min(8.0, len(_DIGIT_RE.findall(text)) * 0.35)
    score += min(10.0, len(_CODEISH_RE.findall(text)) * 1.2)
    if _TROUBLE_RE.search(text):
        score += 3.0
    if "```" in assistant or "```" in user:
        score += 2.5
    # modest boost for longer answers (more room for concrete steps)
    score += min(4.0, max(0, len(assistant) - 200) / 800)
    return score


def normalize_user(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def user_key(user: str) -> str:
    h = hashlib.sha256(normalize_user(user)[:2000].encode("utf-8", errors="replace")).hexdigest()
    return h


def passes_filters(
    user: str,
    assistant: str,
    *,
    min_user: int,
    min_asst: int,
    max_user: int,
    max_asst: int,
    min_score: float,
) -> bool:
    u, a = user.strip(), assistant.strip()
    if len(u) < min_user or len(a) < min_asst:
        return False
    if len(u) > max_user or len(a) > max_asst:
        return False
    if fact_density_score(u, a) < min_score:
        return False
    return True


def to_record(user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
    }


# ---------------------------------------------------------------------------
# Optional local documentation -> synthetic memorization pairs
# ---------------------------------------------------------------------------
def iter_doc_pairs(paths: list[Path], *, max_chars: int = 6000, stride: int = 3500) -> Iterator[tuple[str, str]]:
    for p in paths:
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in {".md", ".txt", ".rst"}:
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            continue
        # sliding windows to avoid dumping entire books in one example
        i = 0
        while i < len(raw):
            chunk = raw[i : i + max_chars].strip()
            i += stride
            if len(chunk) < 400:
                continue
            user = (
                "You are distilling built-in reference knowledge (no web search). "
                "From the following documentation excerpt, extract hard facts: exact flags, "
                "defaults, version constraints, numeric limits, error codes, and troubleshooting steps. "
                "Use tight bullets and inline code where appropriate.\n\n"
                f"---\nFile: {p.name}\n---\n\n{chunk}"
            )
            assistant = (
                "Below is a compact fact sheet derived only from the excerpt. "
                "If the excerpt is silent on a topic, it is not listed.\n\n"
                + _summarize_chunk_heuristic(chunk)
            )
            yield user, assistant


def _summarize_chunk_heuristic(chunk: str) -> str:
    """Cheap deterministic pseudo-summary: headings + lines with digits or `code`."""
    lines_out: list[str] = []
    for line in chunk.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("#"):
            lines_out.append(t)
            continue
        if _DIGIT_RE.search(t) or "`" in t or "::" in t or "http" in t.lower():
            if len(t) > 240:
                t = t[:237] + "…"
            lines_out.append(f"- {t}")
        if len(lines_out) >= 40:
            break
    if not lines_out:
        lines_out.append("- (No high-signal lines auto-extracted; re-chunk or edit manually.)")
    return "\n".join(lines_out[:60])


# ---------------------------------------------------------------------------
# HF streaming harvesters
# ---------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    path: str
    split: str
    quota: int


def _pair_magicoder_oss(row: dict[str, Any]) -> tuple[str, str] | None:
    ins = str(row.get("instruction") or "").strip()
    out = str(row.get("response") or "").strip()
    if not ins or not out:
        return None
    lang = str(row.get("lang") or "").strip()
    if lang:
        ins = f"[Language tag: {lang}]\n\n{ins}"
    return ins, out


def _pair_bigcode_self_oss(row: dict[str, Any]) -> tuple[str, str] | None:
    ins = str(row.get("instruction") or "").strip()
    out = str(row.get("response") or "").strip()
    if not ins or not out:
        return None
    concepts = str(row.get("concepts") or "").strip()
    if concepts and len(concepts) < 1200:
        ins = f"{ins}\n\n(Concepts used in synthesis: {concepts})"
    return ins, out


def _pair_magicoder_evol(row: dict[str, Any]) -> tuple[str, str] | None:
    ins = str(row.get("instruction") or "").strip()
    out = str(row.get("response") or "").strip()
    if not ins or not out:
        return None
    return ins, out


def _pair_code_alpaca(row: dict[str, Any]) -> tuple[str, str] | None:
    ins = str(row.get("instruction") or "").strip()
    inp = str(row.get("input") or "").strip()
    out = str(row.get("output") or "").strip()
    if not ins or not out:
        return None
    if inp:
        ins = f"{ins}\n\nAdditional context / constraints:\n{inp}"
    return ins, out


def stream_pairs(
    path: str,
    split: str,
    mapper: Any,
) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(path, split=split, streaming=True)
    for row in ds:
        p = mapper(row)
        if p:
            yield p


def harvest(
    source: Source,
    mapper: Any,
    *,
    min_user: int,
    min_asst: int,
    max_user: int,
    max_asst: int,
    min_score: float,
    seen: set[str],
    rng: random.Random,
    max_scan: int | None = None,
    progress: bool = True,
    pass_tag: str = "",
) -> list[tuple[str, str, float]]:
    """Collect up to ``source.quota`` passing rows; scan up to ``max_scan`` candidates."""
    out: list[tuple[str, str, float]] = []
    cap = max_scan if max_scan is not None else min(25_000, max(source.quota * 50, 800))
    if progress:
        tag = f" {pass_tag}".rstrip()
        print(
            f"[hf_build_train_jsonl] stream{tag}: {source.name} ← {source.path} "
            f"(scan≤{cap}, quota={source.quota}, min_score={min_score:.2f})",
            flush=True,
        )
    stream = stream_pairs(source.path, source.split, mapper)
    pbar = None
    if progress:
        desc = source.name[:32]
        if pass_tag:
            desc = f"{desc} [{pass_tag}]"
        pbar = tqdm(
            unit="pairs",
            desc=desc,
            dynamic_ncols=True,
            mininterval=0.35,
            smoothing=0.05,
            file=sys.stderr,
        )
    n_seen = 0
    postfix_every = 48
    try:
        for user, assistant in stream:
            n_seen += 1
            if pbar is not None:
                pbar.update(1)
                if n_seen % postfix_every == 0:
                    pbar.set_postfix_str(f"seen={n_seen} kept_here={len(out)}/{cap}", refresh=False)

            if not passes_filters(
                user,
                assistant,
                min_user=min_user,
                min_asst=min_asst,
                max_user=max_user,
                max_asst=max_asst,
                min_score=min_score,
            ):
                continue
            k = user_key(user)
            if k in seen:
                continue
            seen.add(k)
            sc = fact_density_score(user, assistant)
            out.append((user, assistant, sc))
            if len(out) >= cap:
                break
    finally:
        if pbar is not None:
            pbar.close()
            if progress and n_seen > 0:
                print(
                    f"[hf_build_train_jsonl]   done {source.name}: scanned≈{n_seen} passing_buffer={len(out)}",
                    flush=True,
                )

    out.sort(key=lambda t: t[2], reverse=True)
    out = out[: source.quota]
    rng.shuffle(out)
    return out


def expand_globs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        for p in globmod.glob(pat, recursive=True):
            pp = Path(p)
            if pp.is_file():
                out.append(pp)
    uniq: dict[str, Path] = {}
    for p in out:
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def collect_doc_paths(dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            continue
        for pat in ("**/*.md", "**/*.txt", "**/*.rst"):
            paths.extend(p for p in root.glob(pat) if p.is_file())
    uniq: dict[str, Path] = {}
    for p in paths:
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("train.jsonl"), help="Output JSONL path")
    ap.add_argument("--target", type=int, default=1000, help="Total rows to write")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-user-chars", type=int, default=40)
    ap.add_argument("--min-assistant-chars", type=int, default=180)
    ap.add_argument("--max-user-chars", type=int, default=8000)
    ap.add_argument("--max-assistant-chars", type=int, default=24000)
    ap.add_argument("--min-fact-score", type=float, default=4.0, help="Raise to 5–7 for stricter density")
    ap.add_argument(
        "--docs-glob",
        action="append",
        default=[],
        help="Glob for local docs (repeatable), e.g. --docs-glob 'docs/**/*.md'",
    )
    ap.add_argument(
        "--docs-dir",
        action="append",
        default=[],
        help="Directory to scan recursively for .md/.txt/.rst (repeatable)",
    )
    ap.add_argument(
        "--doc-quota",
        type=int,
        default=-1,
        help="Max rows synthesized from local .md/.txt/.rst. "
        "-1 = auto (~15%% of --target, capped at 200) only if --docs-glob or --docs-dir match files; "
        "0 = never use local docs (recommended for pure HF coding data). "
        "N>0 = cap at N rows from local docs.",
    )
    ap.add_argument(
        "--backfill-floor",
        type=float,
        default=0.9,
        help="When under --target, lower min fact score down to this floor while re-scanning HF streams",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars and stream status lines (for logs / non-TTY).",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    doc_paths: list[Path] = []
    for g in args.docs_glob:
        doc_paths.extend(expand_globs([g]))
    doc_paths.extend(collect_doc_paths(args.docs_dir))

    if args.doc_quota > 0:
        doc_quota = int(args.doc_quota)
    elif args.doc_quota == 0:
        doc_quota = 0
    else:
        # auto quota only when we actually have local doc files to read
        if doc_paths:
            doc_quota = max(0, min(args.target // 6, 200))
        else:
            doc_quota = 0

    print(
        f"[hf_build_train_jsonl] target={args.target} doc_quota={doc_quota} "
        f"local_doc_files={len(doc_paths)} hf_sources=4",
        flush=True,
    )
    show_progress = (not args.no_progress) and sys.stderr.isatty()
    if not show_progress:
        print(
            "[hf_build_train_jsonl] Tip: tqdm progress prints to stderr; "
            "use default terminal or drop --no-progress. "
            "Streaming has unknown length — bars count rows scanned, not % of a full download.",
            flush=True,
        )

    sources = [
        Source("magicoder_oss", "ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response", "train", 420),
        Source("bigcode_self_oss", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "train", 320),
        Source("magicoder_evol", "ise-uiuc/Magicoder_evol_instruct_110k", "train", 200),
        Source("code_alpaca", "sahil2801/CodeAlpaca-20k", "train", 260),
    ]
    total_q = sum(s.quota for s in sources)
    if total_q > args.target:
        factor = total_q / args.target
        sources = [Source(s.name, s.path, s.split, max(25, int(s.quota / factor))) for s in sources]

    source_mappers: list[tuple[Source, Any]] = [
        (sources[0], _pair_magicoder_oss),
        (sources[1], _pair_bigcode_self_oss),
        (sources[2], _pair_magicoder_evol),
        (sources[3], _pair_code_alpaca),
    ]

    seen: set[str] = set()
    scored: list[tuple[str, str, float]] = []

    pass_idx = [0]

    def harvest_pass(min_score: float, max_scan: int | None, *, pass_tag: str) -> None:
        pass_idx[0] += 1
        if show_progress:
            print(
                f"[hf_build_train_jsonl] ── pass #{pass_idx[0]} pooled={len(scored)}/{args.target} "
                f"tag={pass_tag}",
                flush=True,
            )
        for src, mapper in source_mappers:
            if len(scored) >= args.target:
                break
            scored.extend(
                harvest(
                    src,
                    mapper,
                    min_user=args.min_user_chars,
                    min_asst=args.min_assistant_chars,
                    max_user=args.max_user_chars,
                    max_asst=args.max_assistant_chars,
                    min_score=min_score,
                    seen=seen,
                    rng=rng,
                    max_scan=max_scan,
                    progress=show_progress,
                    pass_tag=pass_tag,
                )
            )

    harvest_pass(args.min_fact_score, None, pass_tag=f"first min={args.min_fact_score:.2f}")

    if doc_quota > 0 and doc_paths:
        rng.shuffle(doc_paths)
        n_doc = 0
        doc_iter = iter_doc_pairs(doc_paths)
        if show_progress:
            doc_iter = tqdm(
                doc_iter,
                desc="local_docs",
                unit="chunk",
                dynamic_ncols=True,
                file=sys.stderr,
            )
        for user, assistant in doc_iter:
            if not passes_filters(
                user,
                assistant,
                min_user=max(30, args.min_user_chars // 2),
                min_asst=max(120, args.min_assistant_chars // 2),
                max_user=args.max_user_chars,
                max_asst=args.max_assistant_chars,
                min_score=max(2.0, args.min_fact_score - 1.0),
            ):
                continue
            k = user_key(user)
            if k in seen:
                continue
            seen.add(k)
            scored.append((user, assistant, fact_density_score(user, assistant)))
            n_doc += 1
            if n_doc >= doc_quota:
                break

    # Re-stream HF sources with a slightly lower fact threshold until we have enough unique rows.
    relax = float(args.min_fact_score)
    stall = 0
    while len(scored) < args.target:
        if relax > args.backfill_floor:
            relax = max(args.backfill_floor, relax - 0.25)
        else:
            relax = float(args.backfill_floor)
        before = len(scored)
        cap = min(200_000, 28_000 + stall * 22_000)
        harvest_pass(relax, cap, pass_tag=f"backfill relax={relax:.2f} scan_cap={cap}")
        if len(scored) == before:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0

    scored.sort(key=lambda t: t[2], reverse=True)
    # trim / pad to target
    picked = scored[: args.target]
    if len(picked) < args.target:
        print(
            f"WARNING: only collected {len(picked)} rows after filters/backfill. "
            f"Lower --min-fact-score (now {args.min_fact_score}), "
            f"--backfill-floor (now {args.backfill_floor}), or shorten min lengths.",
            flush=True,
        )
    rng.shuffle(picked)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for user, assistant, _ in picked:
            f.write(json.dumps(to_record(user, assistant), ensure_ascii=False) + "\n")

    print(f"Wrote {len(picked)} lines to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
