<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Attention backends

All attention implementations perform the same computation. Every token is compared to every other token. The difference is *how* the computation is performed. Basic attention scales poorly because it materializes the full attention matrix in memory, creating bottlenecks that slow down inference. Optimized implementations rearrange the math to reduce memory traffic for faster, more affordable inference.

The [`AttentionInterface`] provides optimized attention implementations. It decouples the attention implementation from the model implementation to simplify experimentation with different functions. Add new backends easily with this consistent interface.

| attention backend | description |
|---|---|
| `"flash_attention_3"` | improves FlashAttention-2 by also overlapping operations and fusing forward and backward passes more tightly |
| `"flash_attention_2"` | tiles computations into smaller blocks and uses fast on-chip memory |
| `"flex_attention"` | framework for specifying custom attention patterns (sparse, block-local, sliding window) without writing low-level kernels by hand |
| `"sdpa"` | built-in PyTorch implementation of [scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| <code>"paged&#124;flash_attention_3"</code> | Paged version of FlashAttention-3 |
| <code>"paged&#124;flash_attention_2"</code> | Paged version of FlashAttention-2 |
| <code>"paged&#124;sdpa"</code> | Paged version of SDPA |
| <code>"paged&#124;eager"</code> | Paged version of eager |

## Set an attention backend

Use the `attn_implementation` argument in [`~PreTrainedModel.from_pretrained`] to instantiate a model with a specific attention function.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", attn_implementation="flash_attention_2"
)
```

Switch between attention backends at runtime without reloading the model using [`~PreTrainedModel.set_attn_implementation`].

```py
model.set_attn_implementation("sdpa")
```

### Kernels

Download and load compiled compute kernels directly from the [Hub](https://huggingface.co/models?other=kernels) at runtime with the [Kernels](https://huggingface.co/docs/kernels/index) library. This avoids packaging issues from mismatched PyTorch or CUDA versions.

Kernels automatically register to [`AttentionInterface`] upon detection. You don't need to install the FlashAttention package explicitly.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", attn_implementation="kernels-community/flash-attn2"
)
```

### SDPA context manager

PyTorch's scaled dot product attention (SDPA) selects the fastest attention function for CUDA backends automatically. It defaults to the PyTorch C++ implementation for other backends.

Force SDPA to use a specific implementation with the [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) context manager.

```py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", attn_implementation="sdpa"
)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)
```

## Backbone-specific attention

Multimodal models use different backbones for each modality. Optimize performance by assigning specific attention functions to each backbone. Some vision backbones perform better in fp32, for example, which FlashAttention does not support.

Map vision backbones to different attention functions with a dict while the text backbone continues to use FlashAttention. Keys in the attention implementation must match sub-config names.

```py
from transformers import AutoModelForImageTextToText

attention_implementation_per_backbone = {"vision_config": "sdpa", "text_config": "flash_attention_2"}

for key in attention_implementation_per_backbone:
    assert key in model.config.sub_configs, f"Invalid key in `attention_implementation`"

model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation=attention_implementation_per_backbone
)
```

Omit certain backbones from the dict to use the default attention function (SDPA).

```py
model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation={"text_config": "flash_attention_2"}
)
```

Set the same attention function for all backbones with a single string.

```py
model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation="eager"
)
```

Set the attention function globally with an empty key.

```py
model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation={"": "eager"}
)
```

## Create a new attention function

Customize or create new attention functions by adding them to the attention registry with [`AttentionInterface.register`]. Models use these functions through the `attn_implementation` argument.

This example customizes the attention function to print a statement for each layer.

```python
import torch
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

def my_new_sdpa(*args, **kwargs):
    print("I just entered the attention computation")
    return sdpa_attention_forward(*args, **kwargs)

AttentionInterface.register("my_new_sdpa", my_new_sdpa)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="my_new_sdpa")
model(torch.ones(1, 5, dtype=int))
```

You can also add new arguments to the attention function. Models supporting [`AttentionInterface`] propagate kwargs to attention layers and the attention function. Pass arguments as kwargs in the model's forward function. Custom attention functions must follow this signature and return format.

```python
import torch
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

def custom_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    a_new_kwargs = None,  # You can now add as many kwargs as you need
    another_new_kwargs = None,  # You can now add as many kwargs as you need
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
    ...  # do your magic!
    return attn_output, attn_weights  # attn_weights are optional here

AttentionInterface.register("custom", custom_attention)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)
```

Check a model's [modeling code](https://github.com/huggingface/transformers/tree/main/src/transformers/models) to confirm what arguments and kwargs it sends to the attention function.

### AttentionMaskInterface

Configure which key and value tokens queries attend to with [`AttentionMaskInterface`]. Some attention functions require this configuration. Customize the attention mask function and add it to the registry with [`AttentionMaskInterface.register`].

```python
import torch
from transformers import AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

def my_new_sdpa_mask(*args, **kwargs):
    print("I just entered the attention mask computation")
    return sdpa_mask(*args, **kwargs)

AttentionMaskInterface.register("my_new_sdpa_mask", my_new_sdpa_mask)
```

Registered attention masks automatically correct the mask format for the attention implementation. For example, FlexAttention uses a [BlockMask](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html?utm_source=chatgpt.com#torch.nn.attention.flex_attention.BlockMask) format, while SDPA uses a 4D tensor. Without a registered attention mask function, mask creation is skipped and `attention_mask=None` passes to the model's attention layers.

This is the default signature for an attention mask function.

```python
def custom_attention_mask(
    batch_size: int,  # required arg
    cache_position: torch.Tensor,  # required arg
    kv_length: int,  # required arg
    kv_offset: int = 0,  # required arg
    mask_function: Callable = causal_mask_function,  # required arg
    attention_mask: Optional[torch.Tensor] = None,  # required arg
    **kwargs,  # a few additional args may be passed as kwargs, especially the model's config is always passed
) -> Optional[torch.Tensor]:
```

The `mask_function` argument is a `Callable` that mimics PyTorch's [mask_mod](https://pytorch.org/blog/flexattention/) functions. It takes 4 indices as input and returns a boolean. This boolean indicates if the position contributes to the attention computation.

Use this [workaround](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py) for torch export if `mask_function` fails to create a mask.
