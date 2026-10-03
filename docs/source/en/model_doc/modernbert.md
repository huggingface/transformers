<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2024-12-18 and contributed to Hugging Face Transformers on 2024-12-19.*

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# ModernBERT

[ModernBERT](https://huggingface.co/papers/2412.13663) is a modernized version of [`BERT`] trained on 2T tokens. It brings many improvements to the original architecture such as rotary positional embeddings to support sequences of up to 8192 tokens, unpadding to avoid wasting compute on padding tokens, GeGLU layers, and alternating attention.

You can find all the original ModernBERT checkpoints under the [ModernBERT](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb) collection.

> [!TIP]
> Click on the ModernBERT models in the right sidebar for more examples of how to apply ModernBERT to different language tasks.
>
> Set `use_kernels=True` in [`~PreTrainedModel.from_pretrained`] to replace supported layers with optimized kernels from the Hub. Refer to [Loading kernels](../kernel_doc/loading_kernels) to learn more.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


pipeline = pipeline(
    task="fill-mask",
    model="answerdotai/ModernBERT-base",
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "answerdotai/ModernBERT-base",
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
</hfoptions>

## Padding-free inference and training

ModernBERT supports padding-free inference and training. For example, you can leverage the [`DataCollatorWithFlattening`] to prepare your inputs:

> [!TIP]
> Padding-free inference and training requires `flash_attention_2` as the attention implementation. Since ModernBERT no longer defaults to FlashAttention2, you must explicitly set `attn_implementation="flash_attention_2"` when loading the model for padding-free usage.

```python
import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithFlattening


model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)


def prepare_text_for_padding_free(texts):
    # base tokenization with padding and subsequent flattening
    inputs_dict = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    flattened_features = collator(
        [
            {"input_ids": i[a.bool()].tolist()}
            for i, a in zip(inputs_dict["input_ids"], inputs_dict["attention_mask"])
        ]
    )

    for k, v in flattened_features.items():
        if isinstance(v, torch.Tensor):
            flattened_features[k] = v.to(model.device)

    return flattened_features


inputs = prepare_text_for_padding_free(
    ["The capital of France is [MASK].", "ModernBERT is a [MASK] model."]
)
model = AutoModelForMaskedLM.from_pretrained(
    model_id, attn_implementation="flash_attention_2", device_map="auto"
)

# Optional: use torch.compile for faster inference
# model.forward = torch.compile(model.forward, fullgraph=True)

out = model(**inputs)
```

## ModernBertConfig

[[autodoc]] ModernBertConfig

## ModernBertModel

[[autodoc]] ModernBertModel
    - forward

## ModernBertForMaskedLM

[[autodoc]] ModernBertForMaskedLM
    - forward

## ModernBertForSequenceClassification

[[autodoc]] ModernBertForSequenceClassification
    - forward

## ModernBertForTokenClassification

[[autodoc]] ModernBertForTokenClassification
    - forward

## ModernBertForMultipleChoice

[[autodoc]] ModernBertForMultipleChoice
    - forward

## ModernBertForQuestionAnswering

[[autodoc]] ModernBertForQuestionAnswering
    - forward

### Usage tips

The ModernBert model can be fine-tuned using the HuggingFace Transformers library with its [official script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) for question-answering tasks.







## Advanced: Custom Attention Masking for Shared Prefixes

ModernBERT's local attention layers use a sliding window mechanism. By default, this window is computed based on **sequence index** (the position in the input tensor), not on `position_ids`. 

This works perfectly for standard use cases. However, if you're using **non-monotonic position_ids** (e.g., packing multiple segments that share a common prefix and restart at the same position), you need to explicitly construct a position-aware sliding window mask.

### Example: Shared Prefix with Multiple Questions

Here's how to correctly handle a scenario where you have one document prefix and multiple independent questions that all start from the same position:

```python
import torch
from transformers import ModernBertConfig, ModernBertModel
from transformers.masking_utils import and_masks, create_bidirectional_mask, create_bidirectional_sliding_window_mask

# Configuration with local attention
config = ModernBertConfig(
    vocab_size=97,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=6,
    num_attention_heads=4,
    local_attention=16,  # Sliding window size
    global_attn_every_n_layers=3,
    max_position_embeddings=512,
    pad_token_id=0,
    attn_implementation="sdpa"  # or "eager"
)

model = ModernBertModel(config).eval()
sliding_window = config.sliding_window  # 8 in this example

# Create a shared prefix (40 tokens) and two question blocks (10 tokens each)
# Both questions restart at position 40 (non-monotonic position_ids)
prefix_len = 40
question_len = 10

input_ids = torch.cat([
    torch.randint(4, 97, (prefix_len,)),      # Shared prefix
    torch.randint(4, 97, (question_len,)),    # Question 1
    torch.randint(4, 97, (question_len,)),    # Question 2
])[None]

# Non-monotonic positions: both questions start at position 40
position_ids = torch.cat([
    torch.arange(prefix_len),                          # Positions 0-39
    torch.arange(prefix_len, prefix_len + question_len),  # Positions 40-49
    torch.arange(prefix_len, prefix_len + question_len),  # Positions 40-49 (repeated!)
])[None]

# Define which tokens can attend to which
# Each segment sees the prefix and itself
segment_ids = torch.tensor([0] * prefix_len + [1] * question_len + [2] * question_len)

def block_mask(batch_idx, head_idx, q_idx, kv_idx):
    """Prefix sees itself; each question sees prefix and itself."""
    return (segment_ids[kv] == 0) | (segment_ids[kv] == segment_ids[q_idx])

def position_based_window(batch_idx, head_idx, q_idx, kv_idx):
    """Sliding window based on position_ids, not sequence index."""
    return (position_ids[0, q_idx] - position_ids[0, kv_idx]).abs() <= sliding_window

# Create the correct mask: combine block logic WITH position-based window
attention_mask = {
    "full_attention": create_bidirectional_mask(
        config=config,
        inputs_embeds=model.embeddings(input_ids=input_ids),
        attention_mask=torch.ones_like(input_ids),
        and_mask_function=block_mask
    ),
    "sliding_attention": create_bidirectional_mask(
        config=config,
        inputs_embeds=model.embeddings(input_ids=input_ids),
        attention_mask=torch.ones_like(input_ids),
        and_mask_function=and_masks(block_mask, position_based_window)
    )
}

# Now run the model with the position-aware mask
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass the dict, not a single tensor
        position_ids=position_ids
    )

    