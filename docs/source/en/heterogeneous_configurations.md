<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Per-layer configurations

Some checkpoints are heterogeneous (not layer-uniform). A smaller MLP in one layer, fewer key-value heads in another, or a different
layout on selected layers means a single global config doesn't accurately describe the stack.

Use `per_layer_config` on [`~transformers.PreTrainedConfig`] to record those diffs when you're authoring or
inspecting a config, and when model code will consume them. Each entry stores only what differs from the
global configuration, the rest inherits.

> [!NOTE]
> Heterogeneous configurations are a power feature. If a heterogeneous layout becomes a common or prominent
> architecture, we will strive to model it explicitly in the architecture implementation rather than rely on
> `per_layer_config`. Prefer the explicit architecture when one exists.

The models below are heterogeneous checkpoints. Layers are not uniform across the stack. They use a dedicated architecture with `block_configs` and their own `model_type`, rather than `per_layer_config` on a standard architecture.

| Model | Derived from |
|---|---|
| [nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| [nvidia/Llama-3_1-Nemotron-Ultra-253B-v1](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) | [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) |
| [nvidia/gpt-oss-puzzle-88B](https://huggingface.co/nvidia/gpt-oss-puzzle-88B) | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| [nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16) | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) |

## Define per-layer overrides

Pass `per_layer_config` to [`~transformers.LlamaConfig`] as a mapping from layer indices to attribute overrides. Layer
indices are zero-based. Only attributes that differ from the global configuration need to be specified.

`per_layer_config` records and resolves configuration values. It does not by itself change the modules a model creates
or how those modules run. Applying a size override or a `skip` requires model code that reads the resolved per-layer
configuration when constructing or running each layer. The current `LlamaModel` constructs every `LlamaDecoderLayer`
with the global configuration, so the `skip` entries below remain configuration values rather than removing modules
after `from_pretrained`.

The following example records overrides for four layers: layer 5 uses a smaller MLP, layer 11 uses
fewer key-value heads, and layers 23 and 27 record `skip` values for architectures that support them.

```py
from transformers import LlamaConfig


config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    per_layer_config={
        # Use a smaller MLP in one layer.
        5: {"intermediate_size": 8192},

        # Use fewer key-value heads in another layer.
        11: {"num_key_value_heads": 4},

        # Record a request to skip the MLP in architectures that support it.
        23: {"skip": ["mlp"]},

        # Record a request to skip attention in architectures that support it.
        27: {"skip": ["attention"]},
    },
)
```

The submodules that an architecture can skip (for example, `"mlp"` and `"attention"`) are defined per architecture.
`skip` accepts a list, so a layer can record more than one submodule override.

Accessing `config.per_layer_config[layer_idx]` returns a resolved layer configuration. The resolved configuration
combines the global configuration with the overrides for that layer.

```py
# Layer 0 does not define overrides, so it inherits the global values.
config.per_layer_config[0].intermediate_size
# 14336

config.per_layer_config[0].num_key_value_heads
# 8

# Layer 5 overrides the MLP intermediate size.
config.per_layer_config[5].intermediate_size
# 8192

# Layer 11 overrides the number of key-value heads.
config.per_layer_config[11].num_key_value_heads
# 4

# Layer 23 records an MLP skip.
config.per_layer_config[23].skip
# ["mlp"]

# Layer 27 records an attention skip.
config.per_layer_config[27].skip
# ["attention"]
```

Configurations that use `per_layer_config` support the same [`~PreTrainedConfig.save_pretrained`] and
[`~PreTrainedConfig.from_pretrained`] round trip as other configurations.

Each architecture defines in its code which attributes it consumes at the layer level. `per_layer_config` provides the
mechanism for recording those layer-level differences and resolving them against the global config.

## Global attribute access

An attribute with per-layer overrides does not have a single model-wide value. `num_key_value_heads` may be `8` on
most layers and `4` on selected layers. Reading `config.num_key_value_heads` outside a layer context is ambiguous.

By default that access raises `AmbiguousGlobalPerLayerAttributeError` and points you to
`config.per_layer_config[layer_idx]`. The attribute still exists on the global config, so this is not an
`AttributeError`. Reading that global value without a layer index is still wrong. Code that builds a key-value cache from a global
`num_key_value_heads` would size the wrong layers incorrectly.

Set `allow_global_per_layer_attribute_access=True` only when you intentionally need the global fallback and can handle
heterogeneous configs. Global access is then allowed. A warning is emitted once.

```py
config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    allow_global_per_layer_attribute_access=True,
    per_layer_config={
        11: {"num_key_value_heads": 4},
    },
)

config.num_key_value_heads
# 8
# Emits a one-time warning because num_key_value_heads has a per-layer override.
```

## Serialization

`per_layer_config` serializes sparsely by default, and layers without overrides are omitted. Overridden attributes that
match the global value are omitted too.

```py
from transformers import LlamaConfig


config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=4,
    num_attention_heads=32,
    num_key_value_heads=8,
    per_layer_config={
        0: {"num_key_value_heads": 8},
        2: {"num_key_value_heads": 4},
    },
)

config.to_dict()["per_layer_config"]
# {"2": {"num_key_value_heads": 4}}
```

Set `serialize_explicit_per_layer_config=True` to include every layer for the attributes represented in
`per_layer_config`. That makes the layer layout easier to inspect when some values still match the global
configuration.

```py
explicit_config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=4,
    num_attention_heads=32,
    num_key_value_heads=8,
    serialize_explicit_per_layer_config=True,
    per_layer_config={
        0: {"num_key_value_heads": 8},
        2: {"num_key_value_heads": 4},
    },
)

serialized_per_layer_config = explicit_config.to_dict()["per_layer_config"]

serialized_per_layer_config
# {
#     "0": {"num_key_value_heads": 8},
#     "1": {"num_key_value_heads": 8},
#     "2": {"num_key_value_heads": 4},
#     "3": {"num_key_value_heads": 8},
# }
```

Use sparse serialization for compact configs, and explicit serialization when you need the full per-layer layout for
readability or tooling.
