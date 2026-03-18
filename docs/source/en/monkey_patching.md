<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Monkey patching (experimental feature)

Monkey patching allows you to replace model components globally without modifying the original model code. Once registered, patches are automatically applied when loading any model with [`~PreTrainedModel.from_pretrained`] or [`~PreTrainedModel.from_config`]. This enables you to restructure models for specific requirements like quantization compatibility, apply optimizations, or experiment with architectural variants.

> [!WARNING]
> **Monkey patching should be used as a last resort** when you need to change the layout and structure of a module and or its weights. For many customization and optimization needs, try using the [Attention interface](./attention_interface), [Experts interface](./experts_interface), or [Kernels registry](./kernel_doc/overview) instead. Only use monkey patching when you need structural changes that can't be achieved through custom forward implementations alone (e.g., for quantization library compatibility, fusing layers, or architectural experiments).

## Quick start

Here's a simple example showing how to replace a model component:

```python
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.monkey_patching import register_patch_mapping


# Define your replacement class (must inherit from nn.Module)
class CustomLlamaAttention(LlamaAttention):
    def forward(self, *args, **kwargs):
        # Your custom implementation
        print("Using custom attention!")
        return super().forward(*args, **kwargs)


# Register the patch globally (only applies to transformers modeling modules)
register_patch_mapping(mapping={"LlamaAttention": CustomLlamaAttention})

# Load a model - the patch is automatically applied during initialization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# All LlamaAttention layers in the model are now CustomLlamaAttention instances
print(type(model.model.layers[0].self_attn))  # <class '__main__.CustomLlamaAttention'>
```

## How it works

Monkey patches work through a two-step process:

1. **Registration**: Call [`register_patch_mapping`] to add mappings to a global registry.

2. **Application**: Patches are automatically applied during model initialization:
   - **`from_pretrained` / `from_config`**: Patches are **automatically** applied via an internal context manager. No additional action needed!
   - **Manual construction** (e.g., `Model(config)`): You must use the [`apply_patches`] context manager manually.

Once patches are registered, they persist and affect all subsequent model loads until you clear them with [`clear_patch_mapping`].

**Important limitations**:

- Only classes in `transformers` modeling modules are allowed to be patched (e.g., `LlamaAttention`, `LlamaMLP`).
- The mapping keys can be either exact class names or regular expression patterns (see [Pattern matching](#pattern-matching) below).

## Global registration

Use [`register_patch_mapping`] to register mappings globally:

```python
from transformers.monkey_patching import register_patch_mapping

# Register a single patch
register_patch_mapping(
    mapping={"Qwen2MoeExperts": SequentialExperts}
)

# Register multiple patches at once
register_patch_mapping(
    mapping={
        "Qwen2MoeExperts": SequentialExperts,
        "Qwen2MoeAttention": CustomAttention,
    },
    # Overwrite existing patches if they exist
    overwrite=True,
)
```

## Pattern matching

You can use regular expressions to match multiple classes with a single pattern:

```python
from transformers.monkey_patching import register_patch_mapping

# Match all classes containing "Attention"
register_patch_mapping(
    mapping={".*Attention": CustomAttention}
)

# More examples
register_patch_mapping(
    mapping={
        ".*MoeExperts$": CustomExperts,           # Ends with "MoeExperts"
        "^Llama\\d+Attention$": CustomAttention,  # Llama2Attention, Llama3Attention, etc.
    }
)
```

**Important**: Exact matches take precedence over patterns. If you register both `"LlamaAttention"` and `".*Attention"`, classes named `LlamaAttention` will use the exact-match replacement, while other matching classes will use the pattern-match replacement.

> [!WARNING]
> **Regex patterns can silently break models.** A broad pattern like `".*Attention"` will match *every* class whose name contains "Attention" — including container classes that wrap the attention you actually want to replace. For example, BERT has three attention-related classes: `BertSelfAttention` and `BertCrossAttention` (the inner attention implementations) and `BertAttention` (an outer module that *contains* one of those inner classes). Patching all three with the same custom attention layer produces a broken model because the outer `BertAttention` no longer wraps the inner one — it *is* one, eliminating expected sub-modules like `self` and `output`. Prefer narrow patterns (e.g., `".*SelfAttention$"`) or exact class names to avoid unintended matches.

To unregister patches, use [`unregister_patch_mapping`]:

```python
from transformers.monkey_patching import unregister_patch_mapping

# Unregister a single patch (use exact name or pattern from registration)
unregister_patch_mapping(keys=["Qwen2MoeExperts"])

# Unregister multiple patches at once
unregister_patch_mapping(keys=["Qwen2MoeExperts", "Qwen2MoeAttention"])

# Unregister a pattern
unregister_patch_mapping(keys=[".*Attention"])
```

To clear all registered patches, use [`clear_patch_mapping`]:

```python
from transformers.monkey_patching import clear_patch_mapping

clear_patch_mapping()
```

To view currently registered patches, use [`get_patch_mapping`]:

```python
from transformers.monkey_patching import get_patch_mapping

current_patches = get_patch_mapping()
print(current_patches)
```

## Manual model construction

The [`apply_patches`] context manager is only needed when you're constructing models **manually** (e.g., `Model(config)`) without using `from_pretrained` or `from_config`:

```python
from transformers import LlamaModel, LlamaConfig
from transformers.monkey_patching import register_patch_mapping, apply_patches

# Register patch globally
register_patch_mapping(mapping={"LlamaAttention": CustomAttention})

# For manual construction, you need the context manager
with apply_patches():
    model = LlamaModel(LlamaConfig())  # Uses CustomAttention

# Without the context manager, manual construction uses original classes
model = LlamaModel(LlamaConfig())  # Uses LlamaAttention

# But from_pretrained and from_config will always apply registered patches
model = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")  # Uses CustomAttention
```

## Important notes

- **Weight handling**: Monkey patching only replaces classes, not weights. If your patched class has a different weights layout, you'll need to handle [weight conversions](./weightconverter) separately to ensure compatibility with pretrained weights. See the [Complete example](#complete-example) below for how to combine monkey patches with weight conversion mappings.

- **Global effect**: Patches registered with [`register_patch_mapping`] are applied globally to all models loaded after registration. Always use [`clear_patch_mapping`] to clean up when done, especially in tests, notebooks, or long-running applications.

- **Class validation**: The API automatically validates that replacement classes are `nn.Module` subclasses. If you pass an invalid class, you'll get a clear error message.

- **Thread safety**: All patching operations are thread-safe. You can safely register, unregister, and apply patches from multiple threads.

- **Matching behavior**: When using exact class names, they must exactly match the original class names as they appear in the model's source code (case-sensitive). When using regex patterns, they are matched against class names using `re.search()`.

## Troubleshooting

### My patch isn't being applied

**Check class name or pattern**: Ensure the class name or pattern in your mapping is correct:

```python
# For exact names - must match exactly (case-sensitive)
register_patch_mapping(mapping={"LlamaAttention": CustomAttention})

# For patterns - use valid regex
register_patch_mapping(mapping={".*Attention": CustomAttention})
```

**Verify registration**: Use [`get_patch_mapping`] to confirm your mapping is registered:

```python
print(get_patch_mapping())
# Shows all registered mappings: {'LlamaAttention': <class 'CustomAttention'>, '.*MLP': <class 'CustomMLP'>}
```

**Check model source**: Find the exact class name in the model's source:

```python
from transformers.models.llama import modeling_llama
print(dir(modeling_llama))  # Look for the class name
```

### How do I know if my patch is working?

Inspect the loaded model to verify the patch:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Check the type of a specific module
print(type(model.model.layers[0].self_attn))  # Should show your custom class

# Or iterate through all modules
for name, module in model.named_modules():
    if 'attention' in name.lower():
        print(f"{name}: {type(module)}")
```

### Weight shape mismatch errors

If your patched class has different weight shapes, register a weight conversion:

```python
from transformers.conversion_mapping import register_checkpoint_conversion_mapping, WeightConverter
from transformers.monkey_patching import register_patch_mapping

register_patch_mapping(
    mapping={
        "LlamaAttention": LlamaFusedAttention,
    }
)

register_checkpoint_conversion_mapping(
    model_type="llama",
    mapping=[
        WeightConverter(
            source_patterns=["q_proj", "k_proj", "v_proj"],
            target_patterns=["qkv_proj"],
            operations=[
                Concatenate(dim=0),
            ],
        )
    ],
    overwrite=True,
)
```

### Cleaning up patches

Always clean up patches when you're done to avoid affecting other code:

```python
from transformers.monkey_patching import register_patch_mapping, clear_patch_mapping

try:
    register_patch_mapping(mapping={"LlamaAttention": CustomAttention})
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # ... use model ...
finally:
    clear_patch_mapping()  # Always clean up
```

## Complete example

Here's a comprehensive example showing how to restructure both the experts and attention modules in a Mixture-of-Experts model (`qwen2_moe`) for optimization and quantization compatibility. This demonstrates:

1. Creating custom replacement classes that maintain the same interface
2. Registering monkey patches for multiple components
3. Handling weight conversions for the new structure

```python
from typing import Unpack

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, Concatenate, WeightConverter
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2_moe.modeling_qwen2_moe import apply_rotary_pos_emb
from transformers.monkey_patching import register_patch_mapping
from transformers.utils.generic import TransformersKwargs


class MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Adapted from the original Qwen2MoeExperts
class ModuleListExperts(nn.ModuleList):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(MoeMLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# Adapted from the original Qwen2MoeAttention
class FusedQKVAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        if self.config.layer_types[layer_idx] == "sliding_attention":
            self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.qkv_proj(hidden_states).chunk(3, dim=-1)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Registering monkey patches for the new attention and experts modules.
register_patch_mapping(
    mapping={
        "Qwen2MoeExperts": ModuleListExperts,
        "Qwen2MoeAttention": FusedQKVAttention,
    }
)

# Registering weight conversion mappings adapted for the new modules. This registration will:
# - Override the original conversion mapping for qwen2_moe which concatenated the experts into a single parameter format.
# - Concatenate the q_proj, k_proj, v_proj weights/biases into a single qkv_proj weight/bias for the new fused attention module.
register_checkpoint_conversion_mapping(
    model_type="qwen2_moe",
    mapping=[
        WeightConverter(
            source_patterns=["q_proj", "k_proj", "v_proj"],
            target_patterns=["qkv_proj"],
            operations=[Concatenate(dim=0)],
        ),
    ],
    overwrite=True,
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
```

## API reference

[[autodoc]] transformers.monkey_patching.register_patch_mapping

[[autodoc]] transformers.monkey_patching.unregister_patch_mapping

[[autodoc]] transformers.monkey_patching.clear_patch_mapping

[[autodoc]] transformers.monkey_patching.get_patch_mapping

[[autodoc]] transformers.monkey_patching.apply_patches
