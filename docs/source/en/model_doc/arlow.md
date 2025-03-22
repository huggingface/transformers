# Arlow: A PyTorch Transformer Architecture for Causal Language Modeling

## Overview

**Arlow** is a decoder-only Transformer tailored for large-scale causal language model (CLM) pretraining. The project includes:

- **`configuration_arlow.py`** – Defines `ArlowConfig`, storing model hyperparameters and HF metadata.
- **`modeling_arlow.py`** – Houses the actual PyTorch modules:
  - `ArlowPreTrainedModel` (base class)
  - `ArlowModel` (the backbone)
  - `ArlowForCausalLM` (the backbone + LM head + loss)
- **Pretraining script** – Demonstrates how to pre-train Arlow from scratch using Hugging Face `Trainer` or a custom loop. Integrates:
  - `flash-attn` for efficient attention
  - gradient checkpointing
  - optional cross-attention
  - large-scale streaming data
- **Naming scheme** – All classes & modules are called **Arlow**. However, the `model_type` in the config is `"arlowgpt"` for Hugging Face registration.

This document walks through these components in detail.

---

## 1. Configuration

Arlow’s config is in `configuration_arlow.py`:

```python
from transformers import PretrainedConfig

class ArlowConfig(PretrainedConfig):
    model_type = "arlowgpt"  # HF registration name

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=2304,
        intermediate_size=9216,
        max_position_embeddings=2048,
        num_attention_heads=12,
        num_key_value_heads=12,
        num_hidden_layers=28,
        attention_dropout=0.1,
        initializer_range=0.02,
        hidden_act="silu",
        use_cache=True,
        rms_norm_eps=1e-6,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        pad_token_id=None,
        cross_attention=True,
        use_cross_attention=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.cross_attention = cross_attention
        self.use_cross_attention = use_cross_attention
```

**Key Points**:
- **`model_type = "arlowgpt"`** – This is the HF key for your config. Even though everything else is named “Arlow”, `model_type` remains `"arlowgpt"` so that Hugging Face can auto-map your config in e.g. `AutoModelForCausalLM`.
- **`cross_attention`** / **`use_cross_attention`** toggles if you want a seq2seq scenario or multi-turn pipeline. 
- **`tie_word_embeddings`** ensures the LM head shares weights with the input embeddings.
- **`rms_norm_eps`**, **`rope_theta`**, and other fields reflect LLaMA-like parameterization.

---

## 2. Modeling

Located in `modeling_arlow.py`, we have:

1. **`ArlowPreTrainedModel`** – The base class that inherits from `PreTrainedModel`. It sets `_init_weights`, `_set_gradient_checkpointing`, etc.
2. **`ArlowModel`** – A “bare” decoder architecture using:
   - Flash-based multi-head attention (`flash_attn`) 
   - Rotary position embeddings 
   - RMSNorm
   It outputs **hidden states** but doesn’t compute the LM logits or loss.
3. **`ArlowForCausalLM`** – Wraps `ArlowModel` and adds:
   - A final `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`
   - A next-token prediction loss if `labels` are provided.

### 2.1 The Cross-Attention Toggle

By default, `ArlowModel` can have cross-attention blocks if `config.cross_attention=True`. This lets you pass `encoder_hidden_states` to the forward pass. If no encoder states are provided, the model just runs standard self-attention.

### 2.2 FlashAttention & Grouped Q-Attention

We rely on `flash_attn.modules.mha.MHA` to handle the Q/K/V projection more efficiently. We define:

```python
class ArlowGroupedQueryAttention(nn.Module):
    ...
    self.mha = MHA(...)
```

If `flash_attn` isn’t installed, a warning is raised and you cannot instantiate the model.

### 2.3 RMS Norm & Rotary Embeddings

We define:
- **`ArlowRMSNorm`** – used in each layer’s pre- or post-attention steps.
- **`ArlowRotaryEmbedding`** – a method to create cos/sin for RoPE. Integration with flash-attn is manual (some code examples show how to call `apply_rotary_pos_emb` for Q/K).

---

## 3. Pretraining Script

Our **pretraining script** demonstrates how to train Arlow from scratch with Hugging Face’s `Trainer`. Key points:

- **FlashAttention** is used in the model for better performance on large seq lengths.
- **Gradient Checkpointing**: Each layer is optionally wrapped in `torch.utils.checkpoint.checkpoint(...)` if `model.gradient_checkpointing=True`.
- **IterableDataset** approach**: We feed data from a large streaming dataset (like a multi-GB web dataset).
- **`ArlowForCausalLM`** is wrapped in a `Trainer` with a custom data collator, handle for `labels`, etc.

A snippet:

```python
config = ArlowConfig(
    vocab_size=len(tokenizer),
    hidden_size=2304,
    intermediate_size=9216,
    max_position_embeddings=2048,
    num_attention_heads=12,
    num_key_value_heads=12,
    num_hidden_layers=28,
    attention_dropout=0.1,
    initializer_range=0.02,
    hidden_act="silu",
    use_cache=True,
    rms_norm_eps=1e-6,
    rope_theta=100000.0,
    tie_word_embeddings=True,
    pad_token_id=tokenizer.pad_token_id,
    cross_attention=True,
    use_cross_attention=True
)

model = ArlowForCausalLM(config)
model.gradient_checkpointing_enable()

trainer = Trainer(
    model=model,
    ...
)
trainer.train()
```

We rely on standard HF training arguments like `save_steps`, `logging_steps`, `report_to=["wandb"]`, plus a local or remote dataset pipeline.

---

## 4. Naming Scheme

- **Classes**: `ArlowConfig`, `ArlowModel`, `ArlowForCausalLM`, etc.
- **`model_type`** in `ArlowConfig`: `"arlowgpt"`.
- All references in your code say `Arlow` for the brand, but we keep `"arlowgpt"` as the HF registration string.

**Why?**
It’s a standard approach to keep your internal brand name “Arlow” while telling HF that your config is recognized under `"arlowgpt"`. This ensures `AutoModelForCausalLM.from_pretrained("your-arlow-checkpoint")` works properly.

---

## 5. Testing

We have two main test files:

### 5.1 `test_tokenization_arlow.py`

This file uses `unittest` + `TokenizerTesterMixin`. For example:

```python
from transformers.testing_utils import require_tokenizers
from transformers import TokenizerTesterMixin
import unittest

from transformers.models.arlow.tokenization_arlow import ArlowTokenizer

@require_tokenizers
class ArlowTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = ArlowTokenizer
    test_slow_tokenizer = False  # We only have a fast tokenizer

    def setUp(self):
        self.tokenizer = ArlowTokenizer(...)

    def test_full_tokenizer(self):
        input_text = "Hello world!"
        enc = self.tokenizer.encode(input_text)
        self.assertEqual(self.tokenizer.decode(enc), input_text)

```

It checks basic encode/decode, special tokens, added tokens, etc. Hugging Face’s doc testers look for `test_` files with `unittest.TestCase` classes and run them.

### 5.2 `test_modeling_arlow.py`

A test suite for `ArlowModel` and `ArlowForCausalLM`. We define:

- A *tester* class with a small config (`vocab_size=32`, `hidden_size=16`, etc.).
- `all_model_classes = (ArlowModel, ArlowForCausalLM)` so that the common HF test framework can identify them.
- Tests that create a model, do a forward pass (with or without labels), shape checks, and optionally `.generate()` if you want to confirm text generation.

Example:

```python
class ArlowModelTest(unittest.TestCase):
    all_model_classes = (ArlowModel, ArlowForCausalLM)

    def setUp(self):
        self.model_tester = ArlowModelTester()

    def test_model(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, attention_mask, labels)

    def test_model_for_causal_lm(self):
        
```

This ensures your architecture can be tested automatically.

---

## 6. Additional Notes

1. **Modeling and Training**
   - If your environment does not have `flash_attn` installed, you’ll see a warning. Typically, you need a custom PyTorch extension or suitable GPU for flash-attn.
   - Cross-attention remains optional. If `encoder_hidden_states=None`, the cross-attn block is skipped.
   - For multi-turn conversation, you can simply feed a long prompt with the conversation. Or, if you want a formal seq2seq approach, pass an actual `encoder_hidden_states` from an encoder.

2. **Saving & Loading**
   - Once trained, you can do `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)` to store your final checkpoint.
   - Reloading is done with `from_pretrained(...)`, which will look at `ArlowConfig` and recognized `model_type = "arlowgpt"`.

---

## Conclusion

**Arlow** is a flexible LLaMA-like or GPT-like architecture that you can pretrain from scratch. It integrates:

- **FlashAttention** for speed at large sequence lengths
- **RoPE** (rotary embeddings)
- **RMSNorm**
- **Cross-attention** toggles (for seq2seq or multi-turn scenarios)

With the provided scripts and tests, you can:

1. Define your config in `configuration_arlow.py`
2. Implement the model in `modeling_arlow.py`
3. Train with the `Trainer` in your pretraining script (using `flash_attn`, a large dataset, gradient checkpointing, etc.)
4. Test everything end-to-end with `test_tokenization_arlow.py` and `test_modeling_arlow.py`.


## ArlowTokenizer
[[autodoc]] ArlowTokenizer

## ArlowConfig
[[autodoc]] ArlowConfig

## ArlowForCausalLM
[[autodoc]] ArlowForCausalLM

## ArlowModel
[[autodoc]] ArlowModel
