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

# Heterogeneous model configurations

Most model configurations in Transformers describe a homogeneous stack: each layer has the same dimensions and contains
the same submodules. Some checkpoints do not follow this pattern. For example, a model may use a smaller MLP in one
layer, fewer key-value heads in another layer, or omit a submodule such as attention or the MLP from selected layers.

`per_layer_config` represents these layer-specific differences directly in the model configuration. Each entry stores
only the attributes that differ from the global configuration. Attributes that are not overridden inherit their value
from the global configuration.

This is useful for checkpoints that remain close to an existing architecture but are no longer layer-uniform, such as
pruned, distilled, or NAS-derived (Neural Architecture Search) models. Instead of defining a new architecture for every
such variant, `per_layer_config` records the layer-level differences in a few lines of config, at little to no
config-side cost.

> [!NOTE]
> Heterogeneous configurations are a power feature. If a heterogeneous layout becomes a common or prominent
> architecture, we will strive to model it explicitly in the architecture implementation rather than rely on
> `per_layer_config`. Prefer the explicit architecture when one exists.

Examples of heterogeneous checkpoints include:

| Model | Derived from |
|---|---|
| [`nvidia/Llama-3_3-Nemotron-Super-49B-v1_5`](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| [`nvidia/Llama-3_1-Nemotron-Ultra-253B-v1`](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) | [`meta-llama/Llama-3.1-405B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) |
| [`nvidia/gpt-oss-puzzle-88B`](https://huggingface.co/nvidia/gpt-oss-puzzle-88B) | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) |
| [`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16) | [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) |

## Define per-layer overrides

Pass `per_layer_config` to a configuration as a mapping from layer indices to attribute overrides. Layer indices are
zero-based. Only attributes that differ from the global configuration need to be specified.

The following example overrides four layers: layer 5 uses a smaller MLP, layer 11 uses
fewer key-value heads, layer 23 skips the MLP, and layer 27 skips attention.

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

        # Omit the MLP from a selected layer.
        23: {"skip": ["mlp"]},

        # Omit attention from a selected layer.
        27: {"skip": ["attention"]},
    },
)
```

The submodules that can be skipped (for example, `"mlp"` and `"attention"`) are defined per architecture. `skip`
accepts a list, so a layer can omit more than one submodule.

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

# Layer 23 skips the MLP.
config.per_layer_config[23].skip
# ["mlp"]

# Layer 27 skips attention.
config.per_layer_config[27].skip
# ["attention"]
```

Configurations that use `per_layer_config` support the same [`~PreTrainedConfig.save_pretrained`] and
[`~PreTrainedConfig.from_pretrained`] round trip as other configurations.

Each architecture defines in its code which attributes are used at the layer level. `per_layer_config` provides the mechanism for
recording those layer-level differences and resolving them against the global config.

## Global attribute access

In a heterogeneous configuration, an attribute with per-layer overrides no longer has a single model-wide value.
For example, `num_key_value_heads` may be `8` for most layers and `4` for selected layers, so reading
`config.num_key_value_heads` outside a layer-specific context is not well-defined.

This matters because consumers that read such an attribute globally would silently apply the wrong value to the
overridden layers. Code that allocates a key-value cache from a global `num_key_value_heads`, for instance,
would be incorrect for the layers that override it.

By default, an `AmbiguousGlobalPerLayerAttributeError` will be raised for this access pattern, directing callers to use
`config.per_layer_config[layer_idx]` instead. We raise this error instead of `AttributeError` because the attribute
exists on the global config, but reading it there is ambiguous without layer-specific context.

Set `allow_global_per_layer_attribute_access=True` only when the caller intentionally needs the global fallback value
and can safely handle heterogeneous configurations. In that case, global access is allowed, but a warning will be emitted once.

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
# Emits a warning_once message because num_key_value_heads has a per-layer override.
```

## Serialization

`per_layer_config` is serialized sparsely by default. Layers without overrides are omitted, and overridden attributes
that match the global value are also omitted.

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

Set `serialize_explicit_per_layer_config=True` when the serialized configuration should include every layer for the
attributes represented in `per_layer_config`. This can make the layer layout easier to inspect, even when some values
match the global configuration.

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

Use sparse serialization for compact configs. Use explicit serialization when readability or downstream tooling benefits
from seeing the full per-layer layout.
