---
language: en
library_name: transformers
license: apache-2.0
tags:
  - blueberry
  - decoder-only
  - rope
  - yarn
  - harmony-chat-format
---

## Blueberry

Blueberry is a small, non-MoE, decoder-only Transformer architecture authored by Dustin Loring. It features a hybrid NoPE/RoPE attention scheme inspired by SmolLM3, alternating between sliding-window attention with RoPE, and full attention without positional encoding (NoPE). Rotary position embeddings support YaRN scaling via the `rope_scaling` configuration.

- Model type: `blueberry`
- Attention: hybrid NoPE/RoPE with sliding and full attention controlled by `layer_types`
- RoPE scaling: YaRN via `rope_scaling` dict
- Tokenizer: GPT-2 style BPE with 100k vocab, Harmony Chat Format template

### Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("your/blueberry")
model = AutoModelForCausalLM.from_pretrained("your/blueberry")

prompt = "Hello, Blueberry!"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

### Configuration

`BlueberryConfig` exposes the key hyperparameters, including `layer_types` to select `sliding_attention` or `full_attention` per layer, `rope_scaling` to enable YaRN, and a default vocab size of 100,000.

### Tokenizer and Harmony Chat Format

The tokenizer mirrors GPT-2 BPE mechanics and includes a chat template compatible with the Harmony Chat Format. Special tokens such as `<|start|>`, `<|end|>`, roles (`<|system|>`, `<|developer|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`), and fields (`<|message|>`, `<|channel|>`) are included. The Jinja2 chat template is assigned to `tokenizer.chat_template`.

### Model Card

- Intended use: research and experimentation with small decoder-only models using hybrid attention and Harmony chat formatting.
- Limitations: small capacity; not instruction-tuned out-of-the-box; tokenizer requires external training to produce `vocab.json` and `merges.txt`.
- Bias and safety: outputs may reflect biases present in training data; apply alignment techniques and safety mitigations for production.

