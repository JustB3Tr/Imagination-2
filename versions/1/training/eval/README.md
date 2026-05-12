# Coding eval tasks (JSONL)

Each line is a JSON object:

- **``id``** — stable task id.
- **``messages``** — chat turns ending with a **user** message (what you would send at inference).
- **``checks``** — list of substrings that should appear in the model reply (cheap regression test).

Optional (future):

- **``predictions_path``** — not in file; pass ``run_coding_eval.py --predictions`` with JSONL lines ``{"id":"...","text":"..."}``.

See ``coding_tasks_smoke.jsonl`` for examples.
