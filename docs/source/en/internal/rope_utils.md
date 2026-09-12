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

## MRoPE

MRoPE is a type of existing `rope_type`, defined by `mrope_section`. It is not a separate entry in the `rope_type` table. You can still apply rope scaling (`linear`, `dynamic`) with it.

`mrope_section` sizes contiguous frequency bands for the temporal, height, and width axes. Those sizes sum to `head_dim // 2`. Frequencies are then repeated so the embedding spans the full `head_dim`. RoPE is applied in one shot (matmul or elementwise multiply of frequencies with positions). `mrope_section` only reorders those frequencies along `(t, h, w)` first, as in Qwen2-VL's `recomposition_frequencies`.

```python
from transformers import Qwen2VLConfig

config = Qwen2VLConfig()
config.text_config.rope_parameters = {
    "rope_type": "default",
    "rope_theta": 1000000.0,
    "mrope_section": [16, 24, 24],  # temporal, height, width frequency band sizes
}
```

Qwen2-VL uses `mrope_section = [16, 24, 24]` (16 temporal, 24 height, 24 width when `head_dim` is 128). Check `mrope_section` in the text config on models that use this layout (Qwen2-VL, GLM-4V).

## Axial RoPE

Separately, `"axial"` is a registered `rope_type`, but it is not listed `ROPE_INIT_FUNCTIONS`. Frequency setup stays on the model. Use it on vision encoders (Qwen2-VL vision model or other vision stacks such as Pixtral). It usually applies the same frequencies (`head_dim // 4` per spatial axis) for height and width positions and does not allow scaling on top. That is the vision-side path, while MRoPE's `mrope_section` is the text-side path on VL models.

## Utilities

[[autodoc]] RopeParameters
    - __call__
