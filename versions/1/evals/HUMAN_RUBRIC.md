# Human evaluation rubric — UI Artisan v1.4

Use this for small panel reviews alongside `layout_metrics.py`.

## Prompt fidelity (1–5)

- Does the assistant’s **visual_report** mention the same regions the rater notices?
- Are **alignment_notes** specific (left/right, approximate px) vs vague?

## Fix quality (1–5)

- Does the proposed CSS/HTML change **directly** address the reported defect?
- Is the change **minimal** (no unrelated refactors)?

## One-shot after user correction (pass/fail)

Given a short user message (“still off-center”), does the model’s second turn reference the **prior screenshot / revision id** and apply a credible fix?

Record: `rater_id`, `case_id`, scores, free-text notes, date.
