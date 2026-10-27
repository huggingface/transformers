<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-26.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Mistral Large 4

## Overview

Mistral Large 4 is a natively multimodal mixture-of-experts model with 1 trillion parameters and 49 billion active
parameters per token (52 billion including the embeddings and the output layer). It takes text and images as input,
produces text, and can answer directly or reason first, with the reasoning effort set per request.

The text decoder builds on the DeepSeek-V3.2 recipe (Multi-head Latent Attention made sparse by a DSA indexer) and adds
a few changes of its own:

- **Attention**: 72 layers of Deepseek Sparse Attention (DSA) with a 512-dimensional KV latent, made sparse by an
  indexer selecting the top 2048 key positions for each query. Queries use a direct projection, there is no Q-LoRA.
  Some layers compute their own selection, the others reuse the one of a previous layer (`"shared"`) like in 
  [GLM-5](./glm_moe_dsa).
- **NoPE layers**: every 4th layer uses no rotary embedding, in the attention and in the indexer.
- **KV sharing**: from layer 39 and on, every 4th layer does not compute a KV latent, instead attending to the one of
  layer 35. They keep their own query, KV up-projection and output weights.
- **Scaled softmax**: queries are scaled per head by `s * log(position + 1) + b`, with learned `s` and `b`, in the
  attention and in the indexer.
- **LeRoPE**: YaRN extends the 512-token base RoPE by a factor of 2048 (1,048,576 positions), and LeRoPE adds a
  learned multiplier per rotary frequency.
- **MoE**: the first layer is dense, every other layer routes each token to 4 of 128 experts, plus one shared expert.
- **Logits**: final logits are soft-capped with `30 * tanh(logits / 30)`.

The vision encoder and the multimodal projector are the same as in [Mistral 3](./mistral3) (Pixtral encoder with a
patch-merge projector).

## Usage examples

The checkpoint is stored in FP8 (128x128 block-scaled weights with dynamic activation scales) and is loaded as is.
The full model needs about 1 TB of GPU memory, so `device_map="auto"` is used to spread it across several GPUs.

The chat template accepts a `reasoning_effort` argument, set to `"none"` (default) or `"high"`. With `"high"`, the model
writes its reasoning between `[THINK]` and `[/THINK]` before answering.

<hfoptions id="usage">
<hfoption id="Image and text">

```python
from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration

model_id = "mistralai/Mistral-Large-4-1T-A52B"

processor = AutoProcessor.from_pretrained(model_id)
model = MistralLarge4ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

</hfoption>
<hfoption id="Reasoning">

```python
from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration

model_id = "mistralai/Mistral-Large-4-1T-A52B"

processor = AutoProcessor.from_pretrained(model_id)
model = MistralLarge4ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "How many prime numbers are there between 100 and 150?"}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="high",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
# Keep the special tokens to see the reasoning between [THINK] and [/THINK]
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False))
```

</hfoption>
</hfoptions>

## Notes

- The checkpoint also contains a draft head for speculative decoding (the `mtp_dspark.*` weights) using the DSpark
    speculative decoding method. <!-- TODO: implement DSpark or specify this is not supported -->
- The DSA indexer is applied as an attention mask on top of dense attention, with the `eager` and `sdpa` backends.
  It gives the same outputs as sparse attention but not its speed-up on long sequences.
<!-- 
TODO: ask tokenizer people about this
- The tokenizer reads a literal `[INST]` typed in a user message as the control token, while the reference
  [mistral-common](https://github.com/mistralai/mistral-common) tokenizer keeps it as plain text. -->

## MistralLarge4Config

[[autodoc]] MistralLarge4Config

## MistralLarge4TextConfig

[[autodoc]] MistralLarge4TextConfig

## MistralLarge4PreTrainedModel

[[autodoc]] MistralLarge4PreTrainedModel

## MistralLarge4TextModel

[[autodoc]] MistralLarge4TextModel
    - forward

## MistralLarge4Model

[[autodoc]] MistralLarge4Model
    - forward

## MistralLarge4ForConditionalGeneration

[[autodoc]] MistralLarge4ForConditionalGeneration
    - forward
