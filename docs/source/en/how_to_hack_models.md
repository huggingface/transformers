<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# How to Hack Any Transformers Model

The [🤗 Transformers](https://github.com/huggingface/transformers) library offers a collection of pre-trained models and tools for natural language processing, vision, and beyond. While these models cover a wide range of applications, you might encounter use cases that aren't supported out of the box. Customizing models can unlock new possibilities, such as adding new layers, altering architectures, or optimizing attention mechanisms. This guide will show you how to modify existing Transformers models to fit your specific needs. The great thing is, you don’t have to step away from the Transformers framework to make these changes. You can actually modify models directly in Transformers and still take advantage of features like the [Trainer API](https://huggingface.co/docs/transformers/main/en/main_classes/trainer), [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel), and efficient fine-tuning with tools like [PEFT](https://huggingface.co/docs/peft/index).

In this guide, we’ll walk you through how to customize existing Transformers models to meet your requirements—without losing the benefits of the ecosystem.

You'll learn how to:

- Modify a model's architecture by changing its attention mechanism.
- Apply techniques like Low-Rank Adaptation (LoRA) to specific model components.

We encourage you to contribute your own hacks and share them here with the community1

## Example: Modifying the Attention Mechanism in the Segment Anything Model (SAM)

The **Segment Anything Model (SAM)** is a state-of-the-art model for image segmentation. In its default implementation, SAM uses a combined query-key-value (`qkv`) projection in its attention mechanism. However, you might want to fine-tune only specific components of the attention mechanism, such as the query (`q`) and value (`v`) projections, to reduce the number of trainable parameters and computational resources required.

### Motivation

By splitting the combined `qkv` projection into separate `q`, `k`, and `v` projections, you can apply techniques like **LoRA** (Low-Rank Adaptation) to only the `q` and `v` projections. This approach allows you to:

- Fine-tune fewer parameters, reducing computational overhead.
- Potentially achieve better performance by focusing on specific components.
- Experiment with different adaptation strategies in the attention mechanism.

### Implementation

#### **Step 1: Create a Custom Attention Class**

Next, subclass the original `SamVisionAttention` class and modify it to have separate `q`, `k`, and `v` projections.

```python
import torch
import torch.nn as nn
from transformers.models.sam.modeling_sam import SamVisionAttention

class SamVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        del self.qkv
        # Separate q, k, v projections
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)

    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # Split q, k, v from the combined projection
                q, k, v = state_dict[key].chunk(3, dim=0)
                # Replace with individual q, k, v projections
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # Mark the old qkv key for deletion
                keys_to_delete.append(key)
        
        # Remove old qkv keys
        for key in keys_to_delete:
            del state_dict[key]

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
        query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

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

**Explanation:**

- **Separate Projections:** The combined `qkv` projection is removed, and separate `q`, `k`, and `v` linear layers are created.
- **Weight Loading Hook:** The `_split_qkv_load_hook` method splits the pre-trained `qkv` weights into separate `q`, `k`, and `v` weights when loading the model. This ensures compatibility with any pre-trained model.
- **Forward Pass:** Queries, keys, and values are computed separately, and the attention mechanism proceeds as usual.

#### **Step 2: Replace the Original Attention Class**

Replace the original `SamVisionAttention` class with your custom class so that the model uses the modified attention mechanism.

```python
from transformers import SamModel
from transformers.models.sam import modeling_sam

# Replace the attention class in the modeling_sam module
modeling_sam.SamVisionAttention = SamVisionAttentionSplit

# Load the pre-trained SAM model
model = SamModel.from_pretrained("facebook/sam-vit-base")
```

**Explanation:**

- **Class Replacement:** By assigning your custom class to `modeling_sam.SamVisionAttention`, any instances of `SamVisionAttention` in the model will use the modified version. Thus when you call `SamModel`, it will use the newly defined `SamVisionAttentionSplit`. 
- **Model Loading:** The model is loaded using `from_pretrained`, and the custom attention mechanism is integrated.

#### **Step 3: Apply LoRA to Specific Projections**

With separate `q`, `k`, and `v` projections, you can now apply LoRA to specific components, such as the `q` and `v` projections.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # Apply LoRA to q and v projections
    lora_dropout=0.1,
    task_type="mask-generation"
)

# Apply LoRA to the model
model = get_peft_model(model, config)
```

**Explanation:**

- **LoRA Configuration:** The `LoraConfig` specifies the rank `r`, scaling factor `lora_alpha`, target modules (`"q"` and `"v"`), dropout, and task type.
- **Applying LoRA:** The `get_peft_model` function applies LoRA to the specified modules in the model.
- **Parameter Reduction:** By focusing on `q` and `v`, you reduce the number of trainable parameters, leading to faster training and lower memory usage.

#### **Step 4: Verify the Number of Trainable Parameters**

It's simple to verify the number of trainable parameters and see what impact your modification had. 

```python
model.print_trainable_parameters()
```

**Expected Output:**

```
trainable params: 608,256 || all params: 94,343,728 || trainable%: 0.6447
trainable params: 912,384 || all params: 94,647,856 || trainable%: 0.9640 # with k 
```

## Contributing Your Own Hacks

Modifying pre-trained models can open up new avenues for research and application. By understanding and adjusting the internal mechanisms of models like SAM, you can tailor them to your specific needs, optimize performance, and experiment with new ideas.

If you've developed your own hacks for Transformers models and would like to share them, consider contributing to this doc.

- **Open a Pull Request:** Share your code changes and improvements directly in the repository.
- **Write Documentation:** Provide clear explanations and examples of your modifications.
- **Engage with the Community:** Discuss your ideas and get feedback from other developers and researchers by opening an issue.