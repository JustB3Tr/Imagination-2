#!/usr/bin/env python3
"""
Smoke-check that Colab / local env has core training imports (torch, transformers, peft, trl).

Run from ``versions/v1.4/training``::

  python scripts/smoke_training_env.py
"""
from __future__ import annotations


def main() -> None:
    import torch  # noqa: F401

    print(f"[smoke] torch {torch.__version__}, cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[smoke] device: {torch.cuda.get_device_name(0)}", flush=True)

    import transformers  # noqa: F401

    import peft  # noqa: F401

    import trl  # noqa: F401

    import datasets  # noqa: F401

    print("[smoke] transformers, peft, trl, datasets import OK", flush=True)


if __name__ == "__main__":
    main()
