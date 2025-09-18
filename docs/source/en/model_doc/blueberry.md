## Blueberry

Author: Dustin Loring — Website: `https://dustinwloring1988.github.io`

Copyright © September 15 2025

### Overview

Blueberry is a small, non-MoE, decoder-only Transformer for text generation. It features a hybrid attention architecture that alternates between sliding-window attention with Rotary Position Embeddings (RoPE) and full-span attention without positional embeddings (NoPE). The RoPE implementation supports YaRN scaling for extended context handling.

- Hidden size: configurable (default 768)
- Layers: configurable (default 12)
- Heads: configurable (default 12)
- Vocabulary: 100K (GPT-2 BPE-compatible files `vocab.json`, `merges.txt`)
- Attention: Alternating sliding-window RoPE and full NoPE via `layer_types`
- Tokenizer: GPT-2-like with Harmony Chat Format template

### Model Architecture

- Hybrid NoPE/RoPE: Layers marked as `sliding_attention` use RoPE with a sliding context window, while `full_attention` layers use full attention without positional encoding. This mirrors patterns used by models like smollm3.
- RoPE with YaRN: The RoPE mechanism integrates YaRN scaling parameters through `config.rope_scaling`, enabling context extension. See `transformers.modeling_rope_utils` for details.

### Tokenizer and Harmony Chat Format

The `BlueberryTokenizer` and `BlueberryTokenizerFast` are GPT-2 style tokenizers (byte-level BPE). They register Harmony special tokens and expose a Jinja2 chat template under `tokenizer.chat_template` that formats conversations across roles and channels:

```jinja
{% for message in messages %}
    {% if message['role'] == 'system' %}
        <|start|><|system|><|message|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'developer' %}
        <|start|><|developer|><|message|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'user' %}
        <|start|><|user|><|message|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'assistant' %}
        <|start|><|assistant|><|channel|>{{ message['channel'] }}<|message|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'tool' %}
        <|start|><|tool|><|message|>{{ message['content'] }}<|end|>
    {% endif %}
{% endfor %}
```

Special tokens added: `<|start|>`, `<|end|>`, `<|call|>`, `<|system|>`, `<|developer|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|channel|>`, `<|message|>`.

### Model Card

- Intended use: Research and experimentation with hybrid attention mechanisms and Harmony chat formatting for small LMs; text generation and chat.
- Limitations: Not pretrained; behavior depends on training. Sliding window layers limit per-layer receptive field; ensure `layer_types` match use cases.
- Biases: As with all language models, potential to reproduce biases present in training data. Evaluate and mitigate before deployment.

