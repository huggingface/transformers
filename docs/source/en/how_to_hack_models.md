<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Customizing model components

Another way to customize a model is to modify their components, rather than writing a new model entirely, allowing you to tailor a model to your specific use case. For example, you can add new layers or optimize the attention mechanism of an architecture. Customizations are applied directly to a Transformers model so that you can continue to use features such as [`Trainer`], [`PreTrainedModel`], and the [PEFT](https://huggingface.co/docs/peft/en/index) library.

This guide will show you how to customize a models attention mechanism in order to apply [Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) to it.

> [!TIP]
> The [clear_import_cache](https://github.com/huggingface/transformers/blob/9985d06add07a4cc691dc54a7e34f54205c04d40/src/transformers/utils/import_utils.py#L2286) utility is very useful when you're iteratively modifying and developing model code. It removes all cached Transformers modules and allows Python to reload the modified code without constantly restarting your environment.
>
> ```py
> from transformers import AutoModel
> from transformers.utils.import_utils import clear_import_cache
>
> model = AutoModel.from_pretrained("bert-base-uncased")
> # modifications to model code
> # clear cache to reload modified code
> clear_import_cache()
> # re-import to use updated code
> model = AutoModel.from_pretrained("bert-base-uncased")
> ```

## Attention class

[Segment Anything](./model_doc/sam) is an image segmentation model, and it combines the query-key-value (`qkv`) projection in its attention mechanisms. To reduce the number of trainable parameters and computational overhead, you can apply LoRA to the `qkv` projection. This requires splitting the `qkv` projection so that you can separately target the `q` and `v` with LoRA.

1. Create a custom attention class, `SamVisionAttentionSplit`, by subclassing the original `SamVisionAttention` class. In the `__init__`, delete the combined `qkv` and create a separate linear layer for `q`, `k` and `v`.

```py
import torch
import torch.nn as nn
from transformers.models.sam.modeling_sam import SamVisionAttention

class SamVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        # remove combined qkv
        del self.qkv
        # separate q, k, v projections
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)
```

2. The `_split_qkv_load_hook` function splits the pretrained `qkv` weights into separate `q`, `k`, and `v` weights when loading the model to ensure compatibility with any pretrained model.

```py
    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # split q, k, v from the combined projection
                q, k, v = state_dict[key].chunk(3, dim=0)
                # replace with individual q, k, v projections
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # mark the old qkv key for deletion
                keys_to_delete.append(key)
        
        # remove old qkv keys
        for key in keys_to_delete:
            del state_dict[key]
```

3. In the `forward` pass, `q`, `k`, and `v` are computed separately while the rest of the attention mechanism remains the same.

```py
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
        query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        return outputs
```

Assign the custom `SamVisionAttentionSplit` class to the original models `SamVisionAttention` module to replace it. All instances of `SamVisionAttention` in the model is replaced with the split attention version.

Load the model with [`~PreTrainedModel.from_pretrained`].

```py
from transformers import SamModel

# load the pretrained SAM model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# replace the attention class in the vision_encoder module
for layer in model.vision_encoder.layers:
    if hasattr(layer, "attn"):
        layer.attn = SamVisionAttentionSplit(model.config.vision_config, model.config.vision_config.window_size)
```

## LoRA

With separate `q`, `k`, and `v` projections, apply LoRA to `q` and `v`.

Create a [LoraConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig) and specify the rank `r`, `lora_alpha`, `lora_dropout`, `task_type`, and most importantly, the modules to target.

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    # apply LoRA to q and v
    target_modules=["q", "v"],
    lora_dropout=0.1,
    task_type="FEATURE_EXTRACTION"
)
```

Pass the model and [LoraConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig) to [get_peft_model](https://huggingface.co/docs/peft/package_reference/peft_model#peft.get_peft_model) to apply LoRA to the model.

```py
model = get_peft_model(model, config)
```

Call [print_trainable_parameters](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftMixedModel.print_trainable_parameters) to view the number of parameters you're training as a result versus the total number of parameters.

```py
model.print_trainable_parameters()
"trainable params: 589,824 || all params: 94,274,096 || trainable%: 0.6256"
```