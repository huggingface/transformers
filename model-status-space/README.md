---
title: vibecheck
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
secrets:
  - HF_TOKEN
  - GITHUB_TOKEN
---

# vibecheck

Paste any Hugging Face checkpoint to run two status checks:

1. **Generate check** — loads the model config and tokenizer via `transformers`, then runs
   a generation call through the Inference API (equivalent to `AutoModel.from_pretrained` + `.generate()`).
2. **CI status** — maps the model type to its test file in
   [`huggingface/transformers`](https://github.com/huggingface/transformers) and shows
   recent GitHub Actions job results for that model type.

## Secrets

| Secret | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | For private/gated models | Passed to `from_pretrained` and `InferenceClient` |
| `GITHUB_TOKEN` | Recommended | Raises GitHub API rate limit from 60 to 5,000 req/hr |
