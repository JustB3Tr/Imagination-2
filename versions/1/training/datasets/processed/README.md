# Place processed training JSONL here (e.g. copy from `datasets/downloaded/coding_hf_export.jsonl`).

Suggested filename for the default trainer: **`coding_sft.jsonl`**

```bash
cp datasets/downloaded/coding_hf_export.jsonl datasets/processed/coding_sft.jsonl
```

Then run `scripts/train_imagination2_coding.py`.
