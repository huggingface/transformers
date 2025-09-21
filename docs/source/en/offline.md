---
title: Use Transformers Fully Offline
---

This guide shows how to run 🤗 Transformers in **air-gapped or firewalled** environments: prevent network calls, control caches, and load **only** local files.

## TL;DR
- Set environment variables:
  - `TRANSFORMERS_OFFLINE=1` — make Transformers operate offline.  
  - `HF_HUB_OFFLINE=1` — keep the Hugging Face Hub client offline (no HEAD/GET).  
- Use `local_files_only=True` in every `from_pretrained(...)` call.  
- Pre-populate the cache or point to local folders.

---

## 1) Disable network access

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

These flags ensure the library and Hub client **do not** make network requests (including metadata checks).

---

## 2) Load models/tokenizers only from local files

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./local/my-model", local_files_only=True)
mdl = AutoModel.from_pretrained("./local/my-model", local_files_only=True)
```

`local_files_only=True` guarantees no download attempts; paths can be **directories** or a **cached id**.

---

## 3) Use or control the cache

Transformers & the Hub client store files under a cache directory. Common ways to control it:

```bash
# put cache somewhere predictable
export TRANSFORMERS_CACHE=/opt/transformers-cache
export HF_HOME=/opt/hf-home
```

Populate this cache in a connected machine (once), then copy it to the target environment.

---

## 4) Pre-download artifacts on a connected machine

```python
from huggingface_hub import hf_hub_download

# example: model weights + tokenizer files
hf_hub_download(repo_id="distilbert-base-uncased", filename="pytorch_model.bin")
hf_hub_download(repo_id="distilbert-base-uncased", filename="config.json")
hf_hub_download(repo_id="distilbert-base-uncased", filename="vocab.txt")
```

Copy the resulting cache directory to the offline machine, set the same env vars (Section 3), and load with `local_files_only=True`.

---

## 5) Pipelines & Trainer offline

```python
from transformers import pipeline

qa = pipeline("question-answering",
              model="./local/my-model",
              tokenizer="./local/my-model",
              local_files_only=True)
```

Training follows the same pattern: ensure all datasets/models/tokenizers are local and pass `local_files_only=True`.

---

## 6) Troubleshooting

- **It still tries a HEAD/GET** → Double-check both `TRANSFORMERS_OFFLINE` and `HF_HUB_OFFLINE`. Some code-paths are in the Hub client.  
- **Missing file error** → Your local folder must contain the same files the model card lists (e.g., `config.json`, tokenizer files, weight shards).  
- **Cache not found** → Confirm `TRANSFORMERS_CACHE`/`HF_HOME` on both machines and identical paths after copying.
