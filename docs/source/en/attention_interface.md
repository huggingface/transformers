<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Attention Interface

This page describes how to use the `AttentionInterface` in order to register custom attention functions to use with
supported models.

## Customizing attention function

Most recent models can now switch from one attention function used in the Attention layer to the other, thanks to a simple mapping.
By default, we provide the implementation for [`sdpa`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[`flash_attention_2`](https://github.com/Dao-AILab/flash-attention) and [`flex_attention`](https://pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)
as well as `eager`, which is a simple matrix multiplication without any optimization on top.  
This is the setting you can usually choose when instantiating a model:

```python
from transformers import AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"

# Here, using flash attention as an example
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
```

But what if you wanted to create your own attention function? Or simply play around with existing ones, adding
a few statements here and there? You can now do so with the `AttentionInterface`! Here is an example:

```python
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

model_id = "meta-llama/Llama-3.2-1B"

def my_new_sdpa(*args, **kwargs):
    print("I just entered the attention computation")
    return sdpa_attention_forward(*args, **kwargs)

AttentionInterface.register("my_new_sdpa", my_new_sdpa)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="my_new_sdpa")
# Try running the forward with the new attention function
model(torch.ones(1, 5, dtype=int))
```

You will see it prints "I just entered the attention computation" as many times as there are layers in the model (with this example, 16 times).

## Dynamically switching attention function

You could dynamically change the model's attention function as well, by overriding the `config._attn_implementation` field:

```python
# Back to use original sdpa implementation
model.config._attn_implementation = "sdpa"

model(torch.ones(1, 5, dtype=int))
```

and it will stop printing the statements, as it now uses the `sdpa` attention.  
This allows to quickly change an attention function, without needing to reload the model!

## What about new args needed in my custom attention function?

But indeed, what if the new function requires a new arg to be properly used? It's no issue! Models supporting the
`AttentionInterface` propagate kwargs all the way to the Attention layers, and to the used attention function. That way,
you can simply pass the arg (as a kwargs, i.e. you need to qualify the name of the arg) in the model's forward, and it will be correctly used in the attention. However, custom attention functions have some limitations. In particular, it must follow the signature and return format of other attention functions, i.e.

```python
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

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
# Forward pass with the new kwargs
model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)
```

If in doubt about what args/kwargs a given model sends to the attention function, simply check that model's modeling code on [GitHub](https://github.com/huggingface/transformers/tree/main/src/transformers/models)!

## Accessing current available implementations

Most of the time, you will simply need to `register` a new function. If, however, you need to access an existing one,
and/or perform a few checks, the preferred way is to use the global `ALL_ATTENTION_FUNCTIONS`. It behaves the same way you
would expect from a usual Python dictionary:

```python
>>> from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

>>> list(ALL_ATTENTION_FUNCTIONS.keys())
>>> ['flash_attention_2', 'flex_attention', 'sdpa']

>>> ALL_ATTENTION_FUNCTIONS["sdpa"]
>>> <function transformers.integrations.sdpa_attention.sdpa_attention_forward>

>>> ALL_ATTENTION_FUNCTIONS.get("sdpa", None)
>>> <function transformers.integrations.sdpa_attention.sdpa_attention_forward>

# You can also globally `register` a new function directly on it
>>> ALL_ATTENTION_FUNCTIONS.register("new_func", new_func)
```

## Attention Mask Interface

Having a new attention function may mean that you need a new format of attention mask to decide what key and value tokens
the query tokens should attend to. This is now possible with the `AttentionMaskInterface`! It works in the same way as
the `AttentionInterface`:

```python
from transformers import AttentionMaskInterface
from transformers.masking_utils import sdpa_mask
import torch

def my_new_sdpa_mask(*args, **kwargs):
    print("I just entered the attention mask computation")
    return sdpa_mask(*args, **kwargs)

AttentionMaskInterface.register("my_new_sdpa_mask", my_new_sdpa_mask)
```

The reason you have to register it is because we need to automatically correct your mask format based on the attention implementation (for example, flex attention uses a BlockMask format, while sdpa uses a 4D tensor).
By default, if you do not register an attention mask function along with your attention function, mask creation will be skipped
and `attention_mask=None` will be passed along to the Attention layers.

The default signature of the attention mask functions is the following:

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

It mostly works thanks to the `mask_function`, which is a `Callable` in the form of [torch's mask_mod functions](https://pytorch.org/blog/flexattention/), taking 4 indices as input and returning a boolean to indicate if this position should take part in the attention computation.

If you cannot use the `mask_function` to create your mask for some reason, you can try to work around it by doing something similar to our [torch export workaround](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py).