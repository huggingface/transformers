<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Utilities for Rotary Embedding

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


# Configuration in Model Configs

To enable and customize rotary embeddings, add a `rope_parameters` field to your model’s configuration file (`config.json`). This field controls the RoPE behavior across model layers. Note that each RoPE variant defines its own set of expected keys and missing keys will raise an error. See the example below which creates a llama config with default RoPE parameters: 


```python
from transformers import LlamaConfig

config = LlamaConfig()
config.rope_parameters = {
    "rope_type": "default", # type of RoPE to use
    "rope_theta": 10000.0 # base frequency parameter
}

# If we want to apply a scaled RoPE type, we need to pass extra parameters
config.rope_parameters = {
    "rope_type": "linear",
    "rope_theta": 10000.0,
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

# Utilities

[[autodoc]] RopeParameters
    - __call__


