<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Rotary embeddings utilities

This page explains how the Rotary Embedding is computed and applied in Transformers and what types of RoPE are supported.

## Overview

Rotary Position Embeddings are a technique used to inject positional information into attention mechanisms without relying on explicit position encodings.  
Instead of adding position vectors to token embeddings, RoPE rotates query and key vectors in the complex plane according to their positions enabling relative positional awareness and better extrapolation to unseen sequence lengths.

The Transformers library provides a flexible and extensible implementation of various RoPE types defined in `[`~modeling_rope_utils.ROPE_VALIDATION_FUNCTIONS`]`, including both the default and scaled variants:

| Rope Type | Description |
|------------|-------------|
| `"default"` | Standard rotary embedding as in LLaMA. |
| `"linear"` | Linear-scaled RoPE which allows longer context windows. |
| `"dynamic"` | NTK-aware scaling computed by rescaling frequency base (`θ`) for longer context. |
| `"yarn"` | YaRN scaling variant providing smoother extrapolation and stability. |
| `"longrope"` | [LongRoPE](https://github.com/microsoft/LongRoPE) scaling as in Phi-2 model series. |
| `"llama3"` | RoPE scaling as in Llama3.1. |
| `"proportional"` | Frequency base scaled by sequence length relative to the original max position. |

These `rope_type` values map to callables in `[`~modeling_rope_utils.ROPE_INIT_FUNCTIONS`]` (except `"default"`, which uses the model’s default inverse-frequency computation).

## Multidimensional RoPE (MRoPE)

MRoPE (also called axial RoPE) is used by many vision-language models so that a token can carry **independent** rotary positions along several axes — typically temporal, height, and width — instead of a single 1D sequence index. See [Qwen2-VL](https://arxiv.org/abs/2405.14599) for the formulation popularized in open VLMs.

In Transformers, MRoPE is **not** a separate `rope_type` in `ROPE_INIT_FUNCTIONS`. Models keep a normal scaling type (usually `"default"`) and add multimodal keys on `rope_parameters`:

| Key | Meaning |
|-----|---------|
| `mrope_section` | List of positive ints that partition half of the rotary head dimension (`head_dim // 2`) into one slice per axis. Length is usually `3` (time / height / width). The sum of the sections must match `head_dim // 2` (or the partial-rotary width when `partial_rotary_factor` is set). |
| `mrope_interleaved` | Optional bool. When true, some models interleave axis frequencies instead of concatenating contiguous sections. |

VL configs list these keys in `ignore_keys_at_rope_validation` so rope validation does not reject them as unknown fields for the chosen `rope_type`.

Example (shape matches Qwen2-VL-style defaults):

```python
config.rope_parameters = {
    "rope_type": "default",
    "mrope_section": [16, 24, 24],  # time, height, width — sums to head_dim // 2
}
```

At runtime the model builds per-axis cos/sin from multi-axis `position_ids`, then recomposes them with `mrope_section` before applying rotary embeddings to queries and keys. Families that use this pattern include Qwen2-VL / Qwen2.5-VL / Qwen3-VL, HunYuan-VL, GLM-V, and related multimodal variants.

Prefer documenting and configuring MRoPE through `mrope_section` (and `mrope_interleaved` when present). Do not invent a new `"mrope"` / `"axial"` `rope_type` unless upstream registers it in `ROPE_INIT_FUNCTIONS`.

## Configuration in Model Configs

To enable and customize rotary embeddings, add a `rope_parameters` field to your model’s configuration file (`config.json`). This field controls the RoPE behavior across model layers. Note that each RoPE variant defines its own set of expected keys and missing keys will raise an error. See the example below which creates a llama config with default RoPE parameters:

```python
from transformers import LlamaConfig

config = LlamaConfig()
config.rope_parameters = {
    "rope_type": "default", # type of RoPE to use
    # rope_theta is optional — omitting it uses the model’s default_theta (typically 10000.0)
}

# If we want to apply a scaled RoPE type, we need to pass extra parameters
config.rope_parameters = {
    "rope_type": "linear",
    "rope_theta": 10000.0,  # can be omitted to fall back to default_theta
    "factor": 8.0  # scale factor for context extension
}
```

## Per-Layer-Type RoPE Configuration

Some models such as Gemma-3 use different layer types with different attention mechanisms, i.e. "full attention" in some blocks and "sliding-window attention" in others. Transformers supports specifying distinct RoPE parameters per layer type for these models. In this case, `rope_parameters` should be a nested dictionary, where top-level keys correspond to `config.layer_types` and values are per-type RoPE parameters. During model initialization, each decoder layer will automatically look up the matching RoPE configuration based on its declared layer type.

```python
from transformers import Gemma3Config

config = Gemma3Config()
config.rope_parameters = {
    "full_attention": {
        "rope_type": "dynamic",
        "rope_theta": 1000000.0,
        "factor": 8.0,
        "original_max_position_embeddings": 8096,
    },
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }
}
```

## Utilities

[[autodoc]] RopeParameters
    - __call__
