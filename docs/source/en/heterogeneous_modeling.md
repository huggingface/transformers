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

# Heterogeneous modeling

[Heterogeneous configurations](./heterogeneous_configurations) record per-layer differences — attribute values that
differ from the global configuration, or submodules that are skipped entirely — in the model configuration. This guide covers the modeling side: how
an architecture declares support for those differences, and how to enable support for a model that does not have it
built in.

The architectures with built-in support are registered in
[`supported_models.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/heterogeneity/supported_models.py).
Any other model, including custom models with remote code, can enable support by attaching a spec to its model class,
without changes to Transformers.

## How it works

When a model is constructed with a heterogeneous configuration, Transformers patches the construction of the
architecture's layer class so that:

1. Each layer is built from its resolved layer configuration, `config.per_layer_config[layer_idx]`, so per-layer
   attribute overrides such as `intermediate_size` or `num_key_value_heads` apply naturally.
2. Each skip type in the layer's `skip` applies a skip descriptor from the spec, replacing the layer members it lists
   with modules that turn them into a no-op.
3. When a mask-affecting attribute (`sliding_window` or `attention_chunk_size`) varies across layers, the model builds
   one attention mask per distinct value, and each affected layer's forward is patched to select the mask matching its
   own value.

All of this is driven by a single declaration, the `HeterogeneousModelingSpec`.

## The heterogeneous modeling spec

A spec names the layer class to patch and how to find each layer's index during construction:

```py
from transformers.integrations.heterogeneity import HeterogeneousModelingSpec
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

HeterogeneousModelingSpec(
    layer_cls=LlamaDecoderLayer,
    layer_idx_variable_name="layer_idx",
)
```

- `layer_cls` is the architecture's repeated layer class (typically its decoder layer) whose differences `per_layer_config` describes.
- `layer_idx_variable_name` is the name of the layer-index argument of `layer_cls.__init__` (`layer_idx` in most
  models). When the layer class does not accept the index as an argument, it is the name of the loop variable used to
  construct the layers, which is then resolved from the call stack.


A spec without skip descriptors is enough as long as no layer has a skip. Supporting `skip` additionally
requires a skip descriptor for each skip type, defining its effect on the layer.

## Skip descriptors

The strings accepted in a configuration's per-layer `skip` lists are the keys of the spec's `skip_descriptors`. Each
`SkipDescriptor` declares which layer members the skip replaces, and whether one of them is the member that updates
the KV cache:

```py
import torch

from transformers.integrations.heterogeneity import HeterogeneousModelingSpec, ReturnEntry, SkipDescriptor, get_skip_replacement
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm


def identity(hidden_states):
    return hidden_states


HeterogeneousModelingSpec(
    layer_cls=LlamaDecoderLayer,
    layer_idx_variable_name="layer_idx",
    skip_descriptors={
        "attention": SkipDescriptor(
            replacements={
                "input_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=identity)
                ),
                "self_attn": get_skip_replacement(
                    LlamaAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
            replaces_kv_cache_updater=True,
        ),
        "mlp": SkipDescriptor(
            replacements={
                "post_attention_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=identity)
                ),
                "mlp": get_skip_replacement(LlamaMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)),
            },
            replaces_kv_cache_updater=False,
        ),
    },
)
```

### Replacements

The replaced members must make the layer's forward collapse to a no-op for that sublayer without editing the forward
itself. In a pre-norm residual layer, skipping attention means the norm passes the hidden states through unchanged and
the attention contributes zeros, so `residual + 0` leaves the residual stream untouched.

`get_skip_replacement` builds a factory for such a no-op module from the original class and a description of what
its forward should return:

- `None` returns nothing (for members whose output is ignored).
- A `ReturnEntry` returns one of the forward's arguments, transformed. `ReturnEntry(arg_name="hidden_states",
  transform=torch.zeros_like)` returns zeros shaped like the input; `transform=identity` passes the input through.
- A list returns a tuple, with `None` for positions that should be `None`. `LlamaAttention` returns
  `(hidden_states, attention_weights)`, so its replacement returns `(zeros, None)`.

The `arg_name` must be an argument of the original class's `forward`.

A replacement key is either a member name, or a `(member_name, member_class)`
tuple, which applies only when the member is an instance of that class and takes precedence over a plain member-name
key. This supports layers whose member class varies, like NemotronH's `mixer` — its built-in spec declares the skip
like this (excerpt):

```py
skip_descriptors={
    "mixer": SkipDescriptor(
        replacements={
            "norm": get_skip_replacement(NemotronHRMSNorm, ReturnEntry(arg_name="hidden_states", transform=identity)),
            ("mixer", NemotronHAttention): get_skip_replacement(
                NemotronHAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
            ),
            ("mixer", NemotronHMoE): get_skip_replacement(
                NemotronHMoE, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
            ),
            ("mixer", NemotronHMamba2Mixer): get_skip_replacement(
                NemotronHMamba2Mixer, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
            ),
        },
        replaces_kv_cache_updater=True,
    ),
},
```

### KV cache

`replaces_kv_cache_updater` states whether the skip replaces the member that updates the layer's KV cache — the module
that calls `past_key_values.update(...)`. It is `True` for an attention skip and `False` for an MLP skip. Layers whose
skips replace the KV cache updater never hold KV states, and this declaration is how caches and attention masks know to
read cache metadata from a different layer.

## The resulting model

A heterogeneous modeling spec and a configuration together determine the resulting model. In the following
gpt-oss model, layer 1 is built with smaller experts, layer 2 uses a shorter sliding window, layer 3's attention is
replaced with a no-op:

```py
from transformers import GptOssConfig, GptOssForCausalLM

config = GptOssConfig(
    num_hidden_layers=4,
    intermediate_size=128,
    sliding_window=16,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    per_layer_config={
        1: {"intermediate_size": 32},
        2: {"sliding_window": 8},
        3: {"skip": ["attention"]},
    },
)
model = GptOssForCausalLM(config)

# Expert weights have shape (num_local_experts, intermediate_size, hidden_size).
# Layer 0 uses the global intermediate size; layer 1 uses its override.
model.model.layers[0].mlp.experts.down_proj.shape
# torch.Size([128, 128, 2880])

model.model.layers[1].mlp.experts.down_proj.shape
# torch.Size([128, 32, 2880])

# Layer 0 uses the global sliding window; layer 2 uses its override. Internally,
# the sliding layers receive their attention masks as a dict keyed by window size,
# {16: <mask>, 8: <mask>}, and each layer's forward selects its own entry.
model.model.layers[0].self_attn.sliding_window
# 16

model.model.layers[2].self_attn.sliding_window
# 8

# Layer 3's attention is a no-op replacement.
type(model.model.layers[0].self_attn).__name__
# 'GptOssAttention'

type(model.model.layers[3].self_attn).__name__
# '_NoOpReplacement'

list(model.model.layers[3].self_attn.parameters())
# []

list(model.model.layers[0].self_attn.parameters())
# [tensor(...), tensor(...), ...]
```

Because the replacements hold no parameters, checkpoints of such models simply do not contain weights for the skipped
members, and [`~PreTrainedModel.save_pretrained`] and [`~PreTrainedModel.from_pretrained`] round trip them out of the
box.

## Enable a custom model

Set `_heterogeneous_modeling_spec` on the model's `PreTrainedModel` base class, so that every model class of the
architecture resolves the same spec:

```py
class MyModelPreTrainedModel(PreTrainedModel):
    ...


MyModelPreTrainedModel._heterogeneous_modeling_spec = spec
```

Nothing else is required: constructing any `MyModel*` class with a heterogeneous configuration applies the spec, and
[`~PreTrainedModel.from_pretrained`] works as usual.

Built-in support lives in `transformers.integrations.heterogeneity.supported_models`. To contribute support for a
Transformers architecture, add a spec factory to `MODEL_TO_SPEC_FACTORY` there instead of attaching an attribute.
